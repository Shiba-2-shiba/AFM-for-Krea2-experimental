"""Prepare or explicitly execute a bounded Krea2 AFM validation matrix.

Dry-run is the default. Pass ``--execute`` to contact ComfyUI. This client
records server-returned artifacts and links to server logs, but never invents
CUDA timing or peak-memory metrics in the client process.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
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


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nodes_by_type(graph: dict, class_type: str) -> list[tuple[str, dict]]:
    return sorted(
        ((str(node_id), node) for node_id, node in graph.items()
         if isinstance(node, dict) and node.get("class_type") == class_type),
        key=lambda item: item[0],
    )


def _input_reference(value):
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def _conditioning_texts(graph: dict, value, seen=None) -> list[str]:
    node_id = _input_reference(value)
    if node_id is None or node_id not in graph:
        return []
    seen = set() if seen is None else seen
    if node_id in seen:
        return []
    seen.add(node_id)
    node = graph[node_id]
    inputs = node.get("inputs", {})
    texts = []
    if node.get("class_type") == "CLIPTextEncode" and isinstance(inputs.get("text"), str):
        texts.append(inputs["text"])
    for item in inputs.values():
        texts.extend(_conditioning_texts(graph, item, seen))
    return texts


def _file_identity(matrix_dir: Path, value):
    if not isinstance(value, str) or not value:
        return value
    candidate = (matrix_dir / value).resolve()
    if candidate.is_file():
        return {"name": Path(value).name, "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
    return {"name": value, "sha256": None}


def _experiment_identity(graph: dict, matrix_dir: Path) -> dict:
    samplers = _nodes_by_type(graph, "KSampler")
    sampler = samplers[0][1].get("inputs", {}) if samplers else {}
    positive = _conditioning_texts(graph, sampler.get("positive"))
    negative = _conditioning_texts(graph, sampler.get("negative"))
    loaders = []
    for node_id, node in sorted(graph.items(), key=lambda item: str(item[0])):
        class_type = str(node.get("class_type", "")) if isinstance(node, dict) else ""
        if class_type in {"UNETLoader", "CLIPLoader", "VAELoader"} or "Lora" in class_type or "LoRA" in class_type:
            inputs = node.get("inputs", {})
            loaders.append({
                "class_type": class_type,
                "identity": {key: value for key, value in sorted(inputs.items())
                             if _input_reference(value) is None},
            })
    latent = _nodes_by_type(graph, "EmptyLatentImage") or _nodes_by_type(
        graph, "EmptySD3LatentImage"
    )
    latent_inputs = latent[0][1].get("inputs", {}) if latent else {}
    input_images = []
    for _, node in _nodes_by_type(graph, "LoadImage"):
        input_images.append(_file_identity(matrix_dir, node.get("inputs", {}).get("image")))
    return {
        "positive_text": positive,
        "negative_text": negative,
        "seed": sampler.get("seed"),
        "model_components": loaders,
        "sampler": {key: sampler.get(key) for key in
                    ("sampler_name", "scheduler", "steps", "cfg", "denoise")},
        "resolution": {key: latent_inputs.get(key) for key in ("width", "height", "batch_size")},
        "input_images": input_images,
    }


def _afm_inputs(graph: dict) -> dict:
    nodes = _nodes_by_type(graph, "Krea2AFMModelPatch")
    if not nodes:
        return {}
    values = copy.deepcopy(nodes[0][1].get("inputs", {}))
    values.pop("model", None)
    values.pop("jsonl_path", None)
    return values


def _parse_indices(value, *, default=(0,)) -> list[int]:
    if value in (None, ""):
        return list(default)
    result = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result


def _parse_scope(value) -> list[int]:
    if str(value or "").strip() == "all":
        return []  # Total block count is host metadata and cannot be guessed offline.
    result = set()
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item.strip()) for item in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def _expected_coverage(graph: dict, afm: dict) -> dict:
    samplers = _nodes_by_type(graph, "KSampler")
    sampler = samplers[0][1].get("inputs", {}) if samplers else {}
    latent = _nodes_by_type(graph, "EmptyLatentImage") or _nodes_by_type(
        graph, "EmptySD3LatentImage"
    )
    latent_inputs = latent[0][1].get("inputs", {}) if latent else {}
    steps = sampler.get("steps")
    batch_size = latent_inputs.get("batch_size", 1)
    if (sampler.get("sampler_name"), sampler.get("scheduler"), batch_size) != ("euler", "simple", 1):
        return {"protocol": None, "status": "unverified", "reason": "unsupported coverage protocol",
                "semantic_keys": []}
    if type(steps) is not int or steps <= 0:
        return {"protocol": None, "status": "unverified", "reason": "unknown step count",
                "semantic_keys": []}
    blocks = _parse_scope(afm.get("diagnostic_block_scope", afm.get("block_scope", "")))
    every = afm.get("diagnostic_every_n_steps", 1)
    if type(every) is not int or every <= 0 or not blocks:
        return {"protocol": None, "status": "unverified", "reason": "invalid diagnostic schedule",
                "semantic_keys": []}
    heads = _parse_indices(
        afm.get("diagnostic_head_indices"),
        default=range(int(afm.get("diagnostic_head_limit", 1))),
    )
    branch_mode = afm.get("branch_mode", "positive_only")
    if branch_mode != "positive_only":
        return {"protocol": None, "status": "unverified",
                "reason": "initial coverage protocol requires positive_only CFG selection",
                "semantic_keys": []}
    branches = {"positive_only": [0], "negative_only": [1], "both": [0, 1]}.get(branch_mode)
    if branches is None:
        return {"protocol": None, "status": "unverified", "reason": "unsupported branch mode",
                "semantic_keys": []}
    edit_all = str(afm.get("block_scope", "")).strip() == "all"
    edit_blocks = set(_parse_scope(afm.get("block_scope", "")))
    start = float(afm.get("start_percent", 0.0))
    end = float(afm.get("end_percent", 1.0))
    mode = afm.get("mode", "off")
    zero_original = afm.get("strength", 0.0) == 0 and afm.get("zero_strength_mode") == "original"
    keys = []
    for step in range(0, steps, every):
        progress = step / max(steps - 1, 1)
        for block in blocks:
            for branch in branches:
                for head in heads:
                    keys.append({
                        "step_index": step, "block_index": block, "branch_kind": branch,
                        "head_index": head, "sample_index": 0,
                        "expected_edit": bool(mode == "edit" and (edit_all or block in edit_blocks)
                                              and start <= progress <= end and not zero_original),
                        "normalization_mode": afm.get("normalization_mode", "joint"),
                        "expected_implementation": (
                            "original" if mode != "edit" or (not edit_all and block not in edit_blocks)
                            or not start <= progress <= end or zero_original
                            else "manual" if afm.get("normalization_mode") == "preserve_text_mass"
                            else afm.get("implementation", "auto")
                        ),
                    })
    return {"protocol": "euler_simple_single_eval_per_branch_v1", "status": "planned",
            "semantic_keys": keys}


def _apply_diagnostic_mode(run: dict, graph: dict) -> None:
    label = run.get("diagnostic_mode")
    mapping = {
        "off": {"debug_level": "off", "spectral_diag": "off", "compare_kernels": False},
        "summary": {"debug_level": "summary", "spectral_diag": "off",
                    "compare_kernels": False},
        "spectral": {"debug_level": "summary", "spectral_diag": "sampled",
                     "compare_kernels": False},
        "comparison": {"debug_level": "summary", "spectral_diag": "sampled",
                       "compare_kernels": True},
    }
    if label not in mapping:
        return
    for _, node in _nodes_by_type(graph, "Krea2AFMModelPatch"):
        inputs = node.setdefault("inputs", {})
        inputs.update(mapping[label])
        if "diagnostic_top_k" in run:
            inputs["diagnostic_top_k"] = run["diagnostic_top_k"]
        if "diagnostic_radial_bins" in run:
            inputs["diagnostic_radial_bins"] = run["diagnostic_radial_bins"]


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
    _apply_diagnostic_mode(run, graph)
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
        run_ids = {report.get("run_id") for report in complete}
        if len(run_ids) == 1 and None not in run_ids:
            record["runtime_run_id"] = next(iter(run_ids))
        implementations = sorted({
            implementation
            for report in complete
            for implementation in report.get("actual_implementations", [])
        })
        record["actual_implementation"] = (
            implementations[0] if len(implementations) == 1 else "mixed" if implementations else "unverified"
        )


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
        "schema_version": 2,
        "matrix_schema_version": matrix["schema_version"],
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
        try:
            identity = _experiment_identity(graph, args.matrix.resolve().parent)
            afm = _afm_inputs(graph)
            metadata = dict(
            workflow_source=source,
            prompt=str(prompt_path),
            pair_id=_canonical_sha256(identity),
            condition_id=_canonical_sha256(afm),
            repeat_index=run.get("repeat_index", 0),
            phase=run.get("phase", "unspecified"),
            control=run.get("control"),
            condition_role=run.get("condition_role", run.get("control")),
            prompt_text_sha256=_canonical_sha256({
                "positive_text": identity["positive_text"],
                "negative_text": identity["negative_text"],
            }),
            resolved_workflow_sha256=_canonical_sha256(graph),
            model_identity=identity["model_components"],
            case_identity=identity,
            afm_config=afm,
            comparison_group=run.get("comparison_group"),
            requested_implementation=afm.get("implementation", "original"),
            actual_implementation="unverified",
            normalization_mode=afm.get("normalization_mode", "joint"),
            normalization_version=afm.get("normalization_version", 1),
            metric_definition={
                "metric_version": "text_spectrum_v2",
                "diagnostic_top_k": afm.get("diagnostic_top_k"),
                "radial_bins": afm.get("diagnostic_radial_bins", 32),
                "cutoff": afm.get("cutoff"),
            },
            expected_coverage=_expected_coverage(graph, afm),
            runtime_run_id=None,
            )
        except (OSError, TypeError, ValueError) as exc:
            record.update(status="preparation_failed", error=str(exc))
            exit_code = 2
            continue
        record.update(metadata)
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
