import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_afm_matrix import main

ROOT = Path(__file__).resolve().parents[1]


def test_zero_controls_select_explicit_zero_kernels():
    matrix = json.loads((ROOT / "examples" / "validation_matrix.json").read_text())
    controls = {run["control"]: run for run in matrix["runs"]}
    assert controls["manual_zero"]["node_inputs"]["20"]["zero_strength_mode"] == "manual"
    assert controls["augmented_zero"]["node_inputs"]["20"]["zero_strength_mode"] == "augmented"


def _write_fixture(tmp_path: Path):
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "old-unet"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "old-clip"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "old-vae"}},
        "4": {"class_type": "KSampler", "inputs": {"seed": 1}},
    }
    (tmp_path / "workflow.json").write_text(json.dumps(workflow), encoding="utf-8")
    matrix = {
        "schema_version": 1,
        "runs": [
            {
                "id": "smoke",
                "workflow": "workflow.json",
                "seed": 123,
                "control": "baseline",
                "diagnostic_mode": "off",
            }
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    return matrix_path


def test_runner_is_dry_run_by_default_and_writes_reproducible_inputs(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"

    result = main(
        [
            "--comfy-url", "http://127.0.0.1:9",
            "--matrix", str(matrix_path),
            "--out-dir", str(out_dir),
            "--unet-name", "chosen-unet",
            "--clip-name", "chosen-clip",
            "--vae-name", "chosen-vae",
        ]
    )

    assert result == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    prompt = json.loads((out_dir / "prompts" / "smoke.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "dry_run"
    assert manifest["runs"][0]["status"] == "not_submitted"
    assert manifest["runs"][0]["server_metrics"] == "unverified"
    assert prompt["1"]["inputs"]["unet_name"] == "chosen-unet"
    assert prompt["2"]["inputs"]["clip_name"] == "chosen-clip"
    assert prompt["3"]["inputs"]["vae_name"] == "chosen-vae"
    assert prompt["4"]["inputs"]["seed"] == 123


def test_execute_validates_object_info_before_prompt_submission(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method()))
        return Response({})

    result = main(
        [
            "--comfy-url", "http://comfy.invalid",
            "--matrix", str(matrix_path),
            "--out-dir", str(tmp_path / "out"),
            "--execute",
        ],
        opener=opener,
    )

    assert result == 2
    assert calls == [("http://comfy.invalid/object_info", "GET")]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["runs"][0]["status"] == "object_info_failed"
    assert manifest["runs"][0]["server_metrics"] == "unverified"


def test_execute_collects_only_images_from_requested_prompt_history(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    workflow_path = tmp_path / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["5"] = {"class_type": "Krea2AFMModelPatch", "inputs": {"jsonl_path": "old.jsonl"}}
    workflow_path.write_text(json.dumps(workflow))
    calls = []
    log_dir = tmp_path / "server-logs"
    log_dir.mkdir()
    (log_dir / "smoke.jsonl").write_text(
        json.dumps({"run_id": "stale-run", "event": "run_summary", "status": "ok",
                    "observed_calls": 99, "edited_calls": 99, "fallback_calls": 0,
                    "nan_events": 0, "oom_events": 0, "cuda_synchronized": False,
                    "measurement_source": "server", "diagnostics_enabled": False}) + "\n"
    )
    schemas = {
        name: {"input": {"required": {field: ["STRING"]}}, "output": []}
        for name, field in (
            ("UNETLoader", "unet_name"),
            ("CLIPLoader", "clip_name"),
            ("VAELoader", "vae_name"),
            ("KSampler", "seed"),
            ("Krea2AFMModelPatch", "jsonl_path"),
        )
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            if isinstance(self.payload, bytes):
                return self.payload
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method()))
        if request.full_url.endswith("/object_info"):
            return Response(schemas)
        if request.full_url.endswith("/prompt"):
            submitted = json.loads(request.data)["prompt"]
            relative_log = submitted["5"]["inputs"]["jsonl_path"]
            (log_dir / Path(relative_log).name).write_text(
                json.dumps({"run_id": "server-run", "event": "run_summary", "status": "ok",
                            "observed_calls": 1, "edited_calls": 1, "fallback_calls": 0,
                            "nan_events": 0, "oom_events": 0, "cuda_synchronized": False,
                            "measurement_source": "server",
                            "diagnostics_enabled": False}) + "\n"
            )
            return Response({"prompt_id": "requested-id"})
        if request.full_url.endswith("/history/requested-id"):
            return Response(
                {
                    "requested-id": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "../safe.png",
                                        "subfolder": "run",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    },
                    "unrelated-id": {"outputs": {"9": {"images": [{"filename": "other.png"}]}}},
                }
            )
        if "/view?" in request.full_url:
            return Response(b"synthetic-image")
        raise AssertionError(request.full_url)

    out_dir = tmp_path / "out"
    result = main(
        [
            "--comfy-url", "http://comfy.invalid",
            "--matrix", str(matrix_path),
            "--out-dir", str(out_dir),
            "--server-log-dir", str(log_dir),
            "--execute",
        ],
        opener=opener,
    )

    assert result == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    run = manifest["runs"][0]
    assert run["prompt_id"] == "requested-id"
    assert run["server_metrics"][0]["summary"]["edited_calls"] == 1
    assert len(run["images"]) == 1
    image_path = Path(run["images"][0]["local_path"])
    assert image_path.parent == out_dir / "runs" / "smoke" / "images"
    assert image_path.read_bytes() == b"synthetic-image"
    assert not any("other.png" in url for url, _ in calls)


def test_history_error_is_not_completed_and_does_not_download_images(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    calls = []
    schemas = {
        name: {"input": {"required": {field: ["STRING"]}}, "output": []}
        for name, field in (("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"),
                            ("VAELoader", "vae_name"), ("KSampler", "seed"))
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def read(self):
            return json.dumps(self.payload).encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None

    def opener(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/object_info"):
            return Response(schemas)
        if request.full_url.endswith("/prompt"):
            return Response({"prompt_id": "failed-id"})
        if request.full_url.endswith("/history/failed-id"):
            return Response(
                {
                    "failed-id": {
                        "status": {"status_str": "error", "messages": ["boom"]},
                        "outputs": {"9": {"images": [{"filename": "bad.png"}]}},
                    }
                }
            )
        raise AssertionError(request.full_url)

    out_dir = tmp_path / "out"
    result = main(["--comfy-url", "http://comfy.invalid", "--matrix", str(matrix_path),
                   "--out-dir", str(out_dir), "--execute"], opener=opener)

    assert result == 2
    run = json.loads((out_dir / "manifest.json").read_text())["runs"][0]
    assert run["status"] == "execution_failed"
    assert run["server_metrics"] == "unverified"
    assert not any("/view?" in url for url in calls)


def test_matrix_ids_must_be_safe_path_components(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    matrix = json.loads(matrix_path.read_text())
    matrix["runs"][0]["id"] = "../escape"
    matrix_path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="safe path"):
        main(["--comfy-url", "http://x", "--matrix", str(matrix_path),
              "--out-dir", str(tmp_path / "out")])


def test_runner_injects_unique_server_log_and_links_matching_summary(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    matrix = json.loads(matrix_path.read_text())
    matrix["runs"][0]["node_inputs"] = {"5": {"jsonl_path": "stale.jsonl"}}
    matrix_path.write_text(json.dumps(matrix))
    workflow_path = tmp_path / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["5"] = {"class_type": "Krea2AFMModelPatch", "inputs": {"jsonl_path": "old.jsonl"}}
    workflow_path.write_text(json.dumps(workflow))
    log_dir = tmp_path / "server-logs"
    log_dir.mkdir()
    out_dir = tmp_path / "out"
    assert main(["--comfy-url", "http://127.0.0.1:9", "--matrix", str(matrix_path),
                 "--out-dir", str(out_dir), "--server-log-dir", str(log_dir)]) == 0

    prompt = json.loads((out_dir / "prompts" / "smoke.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    run = manifest["runs"][0]
    expected_name = f"smoke-{manifest['invocation_id']}.jsonl"
    assert prompt["5"]["inputs"]["jsonl_path"] == f"krea2_afm/{expected_name}"
    assert run["server_log_relative"] == f"krea2_afm/{expected_name}"
    assert run["server_log"] == str(log_dir / expected_name)
    assert run["server_metrics"] == "unverified"


def test_repeated_prepares_use_distinct_logs_and_ignore_stale_previous_log(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    workflow_path = tmp_path / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["5"] = {"class_type": "Krea2AFMModelPatch", "inputs": {"jsonl_path": "old.jsonl"}}
    workflow_path.write_text(json.dumps(workflow))
    log_dir = tmp_path / "server-logs"
    log_dir.mkdir()

    manifests = []
    for name in ("first", "second"):
        out_dir = tmp_path / name
        assert main(["--comfy-url", "http://127.0.0.1:9", "--matrix", str(matrix_path),
                     "--out-dir", str(out_dir), "--server-log-dir", str(log_dir)]) == 0
        manifests.append(json.loads((out_dir / "manifest.json").read_text()))

    first_run = manifests[0]["runs"][0]
    second_run = manifests[1]["runs"][0]
    assert manifests[0]["invocation_id"] != manifests[1]["invocation_id"]
    assert first_run["server_log"] != second_run["server_log"]
    Path(first_run["server_log"]).write_text(
        json.dumps({"run_id": "old", "event": "run_summary", "status": "ok",
                    "observed_calls": 1, "edited_calls": 1, "fallback_calls": 0,
                    "nan_events": 0, "oom_events": 0}) + "\n"
    )
    assert not Path(second_run["server_log"]).exists()
    assert second_run["server_metrics"] == "unverified"


def test_execute_links_each_run_to_its_own_mapped_log(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    matrix = json.loads(matrix_path.read_text())
    matrix["runs"].append({**matrix["runs"][0], "id": "second", "seed": 456})
    matrix_path.write_text(json.dumps(matrix))
    workflow_path = tmp_path / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["5"] = {"class_type": "Krea2AFMModelPatch", "inputs": {"jsonl_path": "old.jsonl"}}
    workflow_path.write_text(json.dumps(workflow))
    log_dir = tmp_path / "server-logs"
    log_dir.mkdir()
    prompt_ids = []
    schemas = {
        name: {"input": {"required": {field: ["STRING"]}}, "output": []}
        for name, field in (
            ("UNETLoader", "unet_name"),
            ("CLIPLoader", "clip_name"),
            ("VAELoader", "vae_name"),
            ("KSampler", "seed"),
            ("Krea2AFMModelPatch", "jsonl_path"),
        )
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def opener(request, timeout):
        if request.full_url.endswith("/object_info"):
            return Response(schemas)
        if request.full_url.endswith("/prompt"):
            prompt_id = f"prompt-{len(prompt_ids) + 1}"
            prompt_ids.append(prompt_id)
            submitted = json.loads(request.data)["prompt"]
            log_name = Path(submitted["5"]["inputs"]["jsonl_path"]).name
            (log_dir / log_name).write_text(
                json.dumps({"run_id": prompt_id, "event": "run_summary", "status": "ok",
                            "observed_calls": 1, "edited_calls": len(prompt_ids),
                            "fallback_calls": 0, "nan_events": 0, "oom_events": 0}) + "\n"
            )
            return Response({"prompt_id": prompt_id})
        if "/history/" in request.full_url:
            prompt_id = request.full_url.rsplit("/", 1)[-1]
            return Response({prompt_id: {"outputs": {}}})
        raise AssertionError(request.full_url)

    out_dir = tmp_path / "out"
    assert main(["--comfy-url", "http://comfy.invalid", "--matrix", str(matrix_path),
                 "--out-dir", str(out_dir), "--server-log-dir", str(log_dir),
                 "--execute"], opener=opener) == 0

    runs = json.loads((out_dir / "manifest.json").read_text())["runs"]
    assert [run["server_metrics"][0]["summary"]["edited_calls"] for run in runs] == [1, 2]


def test_runner_supports_documented_direct_script_invocation():
    result = subprocess.run(
        [sys.executable, "scripts/run_afm_matrix.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_workflow_is_recorded_as_unverified_preparation_failure(tmp_path):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "id": "missing",
                        "workflow": "not-there.json",
                        "control": "manual",
                        "diagnostic_mode": "comparison",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = main(
        [
            "--comfy-url", "http://127.0.0.1:9",
            "--matrix", str(matrix_path),
            "--out-dir", str(out_dir),
        ]
    )

    assert result == 2
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["runs"][0]["status"] == "preparation_failed"
    assert manifest["runs"][0]["server_metrics"] == "unverified"


def test_manifest_v2_pair_identity_ignores_afm_and_output_inputs(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    workflow_path = tmp_path / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["5"] = {
        "class_type": "Krea2AFMModelPatch",
        "inputs": {
            "model": ["1", 0], "normalization_mode": "joint", "strength": 0.05,
            "jsonl_path": "old.jsonl", "spectral_diag": "off",
        },
    }
    workflow["6"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["4", 0], "filename_prefix": "old/prefix"},
    }
    workflow_path.write_text(json.dumps(workflow))
    matrix = json.loads(matrix_path.read_text())
    matrix["runs"] = [
        {**matrix["runs"][0], "id": "joint", "phase": "pilot", "repeat_index": 0,
         "node_inputs": {"5": {"normalization_mode": "joint", "strength": 0.05}}},
        {**matrix["runs"][0], "id": "preserve", "phase": "pilot", "repeat_index": 0,
         "node_inputs": {"5": {"normalization_mode": "preserve_text_mass",
                                "implementation": "manual", "strength": 0.2,
                                "jsonl_path": "ignored.jsonl"},
                         "6": {"filename_prefix": "different/prefix"}}},
    ]
    matrix_path.write_text(json.dumps(matrix))

    out_dir = tmp_path / "out"
    assert main(["--comfy-url", "http://127.0.0.1:9", "--matrix", str(matrix_path),
                 "--out-dir", str(out_dir)]) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    first, second = manifest["runs"]

    assert manifest["schema_version"] == 2
    assert first["pair_id"] == second["pair_id"]
    assert first["condition_id"] != second["condition_id"]
    assert first["resolved_workflow_sha256"] != second["resolved_workflow_sha256"]
    assert first["phase"] == "pilot" and first["repeat_index"] == 0


def test_diagnostic_label_updates_real_afm_node_inputs(tmp_path):
    matrix_path = _write_fixture(tmp_path)
    workflow_path = tmp_path / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["5"] = {
        "class_type": "Krea2AFMModelPatch",
        "inputs": {"model": ["1", 0], "spectral_diag": "off", "diagnostic_top_k": 4},
    }
    workflow_path.write_text(json.dumps(workflow))
    matrix = json.loads(matrix_path.read_text())
    matrix["runs"][0]["diagnostic_mode"] = "spectral"
    matrix["runs"][0]["diagnostic_top_k"] = 8
    matrix_path.write_text(json.dumps(matrix))

    out_dir = tmp_path / "out"
    assert main(["--comfy-url", "http://127.0.0.1:9", "--matrix", str(matrix_path),
                 "--out-dir", str(out_dir)]) == 0
    prompt = json.loads((out_dir / "prompts" / "smoke.json").read_text())

    assert prompt["5"]["inputs"]["spectral_diag"] == "sampled"
    assert prompt["5"]["inputs"]["diagnostic_top_k"] == 8


def test_initial_coverage_protocol_does_not_guess_unconditional_evaluations():
    from scripts.run_afm_matrix import _expected_coverage
    graph = {
        "3": {"class_type": "KSampler", "inputs": {
            "sampler_name": "euler", "scheduler": "simple", "steps": 8, "cfg": 1}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": 512, "height": 512, "batch_size": 1}},
    }
    for branch_mode in ("negative_only", "both"):
        coverage = _expected_coverage(graph, {
            "mode": "edit", "branch_mode": branch_mode,
            "block_scope": "0", "diagnostic_block_scope": "0"})
        assert coverage["status"] == "unverified"
        assert coverage["protocol"] is None
        assert coverage["semantic_keys"] == []
