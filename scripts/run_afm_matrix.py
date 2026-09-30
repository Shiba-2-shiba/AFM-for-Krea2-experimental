"""Prepare or explicitly execute a bounded Krea2 AFM validation matrix.

Dry-run is the default. Pass ``--execute`` to contact ComfyUI. This client
records server-returned artifacts and links to server logs, but never invents
CUDA timing or peak-memory metrics in the client process.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_workflow import validate_workflow
from scripts.summarize_krea2_afm import load_reports

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _json_request(opener, url: str, *, payload=None, timeout: float):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)
    with opener(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_matrix(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("matrix must be a schema_version 1 object")
    runs = value.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("matrix must contain at least one run")
    identifiers = [run.get("id") for run in runs if isinstance(run, dict)]
    if len(identifiers) != len(runs) or any(
        not isinstance(item, str) or not item for item in identifiers
    ):
        raise ValueError("every matrix run needs a non-empty string id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("matrix run ids must be unique")
    if any(not _SAFE_ID.fullmatch(item) or item in (".", "..") for item in identifiers):
        raise ValueError("matrix run ids must be safe path components")
    return value


def _prepare_prompt(
    run: dict, matrix_dir: Path, args, server_log_relative: str
) -> tuple[dict, str]:
    workflow_path = (matrix_dir / run["workflow"]).resolve()
    graph = json.loads(workflow_path.read_text(encoding="utf-8"))
    graph = copy.deepcopy(graph)
    loader_overrides = {
        "UNETLoader": ("unet_name", args.unet_name),
        "CLIPLoader": ("clip_name", args.clip_name),
        "VAELoader": ("vae_name", args.vae_name),
    }
    for node in graph.values():
        override = loader_overrides.get(node.get("class_type"))
        if override and override[1] is not None:
            node.setdefault("inputs", {})[override[0]] = override[1]
        if node.get("class_type") == "KSampler" and "seed" in run:
            node.setdefault("inputs", {})["seed"] = run["seed"]
    for node_id, input_values in run.get("node_inputs", {}).items():
        if node_id not in graph or not isinstance(input_values, dict):
            raise ValueError(f"run {run['id']} has invalid node_inputs entry {node_id!r}")
        graph[node_id].setdefault("inputs", {}).update(input_values)
    for node in graph.values():
        if node.get("class_type") == "Krea2AFMModelPatch":
            node.setdefault("inputs", {})["jsonl_path"] = server_log_relative
    return graph, str(workflow_path)


def _history_entry(payload: dict, prompt_id: str):
    if prompt_id in payload:
        return payload[prompt_id]
    if payload.get("prompt_id") == prompt_id:
        return payload
    return None


def _poll_history(opener, base_url: str, prompt_id: str, timeout: float, interval: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = _json_request(
            opener, f"{base_url}/history/{prompt_id}", timeout=min(timeout, 30.0)
        )
        entry = _history_entry(payload, prompt_id)
        if entry is not None:
            return entry
        time.sleep(min(interval, max(deadline - time.monotonic(), 0.0)))
    raise TimeoutError(f"history timeout for prompt {prompt_id}")


def _collect_images(opener, base_url: str, history: dict, destination: Path, timeout: float):
    collected = []
    outputs = history.get("outputs", {}) if isinstance(history, dict) else {}
    for node_id, output in outputs.items():
        for index, image in enumerate(output.get("images", [])):
            if not isinstance(image, dict) or not isinstance(image.get("filename"), str):
                continue
            query = {
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
            request = Request(f"{base_url}/view?{urlencode(query)}")
            with opener(request, timeout=min(timeout, 30.0)) as response:
                content = response.read()
            safe_name = Path(image["filename"]).name
            safe_node_id = (
                re.sub(r"[^A-Za-z0-9._-]", "_", str(node_id)).strip(".") or "node"
            )
            local_name = f"{safe_node_id}-{index}-{safe_name}"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / local_name).write_bytes(content)
            collected.append(
                {**query, "local_path": str(destination / local_name), "bytes": len(content)}
            )
    return collected


def _execute_run(opener, base_url: str, graph: dict, run_dir: Path, args, object_info: dict):
    errors = validate_workflow(graph, object_info)
    if errors:
        return {"status": "object_info_failed", "validation_errors": errors}
    response = _json_request(
        opener,
        f"{base_url}/prompt",
        payload={"prompt": graph, "client_id": args.client_id},
        timeout=min(args.timeout, 30.0),
    )
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("ComfyUI /prompt response omitted prompt_id")
    history = _poll_history(opener, base_url, prompt_id, args.timeout, args.poll_interval)
    history_path = run_dir / "history.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    status = history.get("status", {}) if isinstance(history, dict) else {}
    status_str = str(status.get("status_str", "")).lower() if isinstance(status, dict) else ""
    if status_str in ("error", "failed"):
        return {
            "status": "execution_failed",
            "prompt_id": prompt_id,
            "history": str(history_path),
            "execution_messages": status.get("messages", []),
        }
    images = _collect_images(opener, base_url, history, run_dir / "images", args.timeout)
    return {
        "status": "completed",
        "prompt_id": prompt_id,
        "history": str(history_path),
        "images": images,
    }


def _attach_server_metrics(record: dict, log_path: Path) -> None:
    record["server_log"] = str(log_path)
    if record.get("status") != "completed" or not log_path.is_file():
        return
    try:
        reports = load_reports(log_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        record["server_log_error"] = str(exc)
        return
    complete = [report for report in reports if isinstance(report.get("summary"), dict)]
    if complete:
        record["server_metrics"] = complete


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-url", required=True)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--unet-name")
    parser.add_argument("--clip-name")
    parser.add_argument("--vae-name")
    parser.add_argument(
        "--server-log-dir",
        type=Path,
        help="mounted or synchronized ComfyUI output/krea2_afm directory",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--client-id", default=f"krea2-afm-{uuid.uuid4()}")
    return parser


def main(argv=None, *, opener=urlopen) -> int:
    args = _parser().parse_args(argv)
    if args.timeout <= 0 or args.poll_interval <= 0:
        raise ValueError("timeout and poll interval must be positive")
    matrix = _load_matrix(args.matrix.resolve())
    out_dir = args.out_dir.resolve()
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.comfy_url.rstrip("/")
    invocation_id = str(uuid.uuid4())
    manifest = {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "comfy_url": base_url,
        "matrix": str(args.matrix.resolve()),
        "server_log_dir": str(args.server_log_dir.resolve()) if args.server_log_dir else None,
        "runner_cuda_peak": None,
        "runs": [],
    }
    prepared = []
    exit_code = 0
    for run in matrix["runs"]:
        log_name = f"{run['id']}-{invocation_id}.jsonl"
        server_log_relative = f"krea2_afm/{log_name}"
        record = {
            "id": run["id"],
            "control": run.get("control"),
            "diagnostic_mode": run.get("diagnostic_mode"),
            "status": "not_submitted",
            "server_metrics": "unverified",
            "server_log_relative": server_log_relative,
        }
        log_path = (
            args.server_log_dir.resolve() / log_name
            if args.server_log_dir
            else None
        )
        if log_path is not None:
            record["server_log"] = str(log_path)
        manifest["runs"].append(record)
        try:
            graph, source = _prepare_prompt(
                run, args.matrix.resolve().parent, args, server_log_relative
            )
            prompt_path = prompt_dir / f"{run['id']}.json"
            prompt_path.write_text(
                json.dumps(graph, indent=2, allow_nan=False), encoding="utf-8"
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            record.update(status="preparation_failed", error=str(exc))
            exit_code = 2
            continue
        record.update(workflow_source=source, prompt=str(prompt_path))
        prepared.append((run, graph, record, log_path))

    if args.execute and prepared:
        try:
            object_info = _json_request(
                opener, f"{base_url}/object_info", timeout=min(args.timeout, 30.0)
            )
        except Exception as exc:
            for _, _, record, _ in prepared:
                record.update(status="object_info_failed", error=str(exc))
            exit_code = 2
        else:
            for run, graph, record, log_path in prepared:
                try:
                    result = _execute_run(
                        opener, base_url, graph, out_dir / "runs" / run["id"], args, object_info
                    )
                    record.update(result)
                    if log_path is not None:
                        _attach_server_metrics(record, log_path)
                    if result["status"] != "completed":
                        exit_code = 2
                except Exception as exc:
                    record.update(status="failed", error=str(exc))
                    exit_code = 2

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(str(manifest_path))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
