"""Scalar-only diagnostic records. No tensors or cross-run caches."""
from dataclasses import asdict
import importlib
import json
import logging
from pathlib import Path
import platform
import subprocess

import torch

from .metrics import ValidationMetrics  # Public plan interface; implementation shared with offline tools.

LOGGER = logging.getLogger("krea2_afm")


def environment_metadata() -> dict:
    commit = "unknown"
    try:
        comfy_file = importlib.import_module("comfy").__file__
        if comfy_file:
            result = subprocess.run(
                ["git", "-C", str(Path(comfy_file).resolve().parent), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
    except (ImportError, AttributeError, OSError, subprocess.TimeoutExpired):
        pass
    cuda_ready = torch.cuda.is_initialized()
    return {
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "comfyui_commit": commit,
        "gpu": torch.cuda.get_device_name() if cuda_ready else "not_initialized",
        "model_filename": "unknown", "model_filename_source": "not_exposed_by_model_patcher",
        "model_family": "krea2", "sdpa_kernel": "unknown",
    }


def emit_record(settings, run_id: str, event: str, **fields) -> dict:
    row = {"schema_version": 4, "model_family": "krea2", "run_id": run_id,
           "event": event, **fields}
    # Reject non-finite scalars and tensors rather than stringifying huge data.
    line = json.dumps(row, ensure_ascii=False, allow_nan=False)
    if settings.jsonl_path:
        destination = Path(settings.jsonl_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    if settings.debug_level == "verbose" or (
        settings.debug_level == "summary" and event in ("run_start", "run_summary")
    ):
        LOGGER.info("[Krea2 AFM] %s", line)
    return row


def configuration_metadata(config) -> dict:
    values = asdict(config)
    # Keep logs portable without exposing a user's full destination path.
    values.pop("jsonl_path")
    return values
