"""Future, explicitly opted-in server integration. Never part of CPU tests."""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_explicit_krea2_server_matrix(tmp_path):
    if os.environ.get("KREA2_AFM_RUN_REAL_MODEL") != "1":
        pytest.skip("real-model execution requires KREA2_AFM_RUN_REAL_MODEL=1")
    url = os.environ.get("KREA2_AFM_COMFY_URL")
    if not url:
        pytest.skip("set KREA2_AFM_COMFY_URL to the explicitly selected server")
    from scripts.run_afm_matrix import main
    root = Path(__file__).resolve().parents[2]
    args = ["--comfy-url", url, "--matrix", str(root / "examples/validation_matrix.json"),
            "--out-dir", str(tmp_path), "--execute"]
    for env, option in (("KREA2_AFM_UNET", "--unet-name"), ("KREA2_AFM_CLIP", "--clip-name"),
                        ("KREA2_AFM_VAE", "--vae-name"), ("KREA2_AFM_SERVER_LOG_DIR", "--server-log-dir")):
        if os.environ.get(env):
            args.extend([option, os.environ[env]])
    assert main(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["mode"] == "execute"
    assert manifest["runs"] and all(run["status"] == "completed" for run in manifest["runs"])
    # Image generation alone is not proof that AFM edited any call. Qualification
    # remains a separate check against the server JSONL and numerical/benchmark evidence.
