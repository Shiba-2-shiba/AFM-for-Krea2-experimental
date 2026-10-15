"""Explicit edit-path selection. Unqualified auto always uses the manual kernel."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .config import AFMConfig, validate_normalization_options
from .runtime import AFMUnsupportedError

SIGNATURE_KEYS = (
    "python", "pytorch", "cuda", "gpu", "compute_capability", "device_type",
    "comfyui_commit", "baseline_backend", "sdpa_settings", "activation_dtype",
    "original_head_dim", "sdpa_explicit_scale", "filter_signature",
)
RANGE_KEYS = ("text_tokens", "image_tokens", "batch", "heads", "grid_h", "grid_w",
              "head_chunk_size", "query_chunk_size", "strength")


@dataclass(frozen=True)
class BackendSelection:
    implementation: str
    reason: str
    profile_id: str | None = None


def load_qualified_profiles() -> list[dict]:
    profiles = json.loads(Path(__file__).with_name("qualified_profiles.json").read_text(encoding="utf-8"))
    if not isinstance(profiles, list):
        raise ValueError("qualified_profiles.json must contain a list")
    return profiles


def _matches(signature: dict, profile: dict) -> bool:
    if not isinstance(profile, dict) or profile.get("qualified") is not True or not profile.get("id"):
        return False
    evidence = profile.get("evidence", {})
    if not all(isinstance(evidence.get(key), str) and evidence[key].strip()
               for key in ("numerical_report", "benchmark_manifest")):
        return False
    expected = profile.get("signature", {})
    for candidate in (signature, expected):
        if not isinstance(candidate, dict):
            return False
        filter_spec = candidate.get("filter_signature", {})
        if (not isinstance(filter_spec, dict)
                or filter_spec.get("normalization_mode") not in ("joint", "preserve_text_mass")
                or filter_spec.get("normalization_version") != 1):
            return False
    if signature.get("device_type") != "cuda":
        return False
    for key in SIGNATURE_KEYS:
        value = signature.get(key)
        if value is None or value == "unknown" or key not in expected or expected[key] != value:
            return False
    ranges = profile.get("ranges", {})
    for key in RANGE_KEYS:
        bounds, value = ranges.get(key), signature.get(key)
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(type(item) in (int, float) and math.isfinite(item) for item in bounds)
                or type(value) not in (int, float) or not math.isfinite(value)
                or not bounds[0] <= value <= bounds[1]):
            return False
    return True


def select_implementation(config: AFMConfig, runtime_signature: dict,
                          profiles: list[dict]) -> BackendSelection:
    validate_normalization_options(config)
    if config.mode != "edit":
        return BackendSelection("original", "diagnostic_passthrough")
    requested = config.zero_strength_mode if config.strength == 0 else config.implementation
    if requested == "original":
        return BackendSelection("original", "zero_original")
    if requested == "manual":
        return BackendSelection("manual", "zero_control" if config.strength == 0 else "explicit_manual")
    if requested == "augmented":
        if not runtime_signature.get("sdpa_explicit_scale", False):
            raise AFMUnsupportedError("unsupported_augmented_backend", "SDPA explicit scale is required")
        return BackendSelection("augmented", "zero_control" if config.strength == 0 else "explicit_augmented")
    if requested != "auto":
        raise ValueError(f"Unknown implementation {requested!r}")
    if runtime_signature.get("sdpa_explicit_scale", False):
        for profile in profiles:
            if _matches(runtime_signature, profile):
                return BackendSelection("augmented", "qualified_profile", profile["id"])
    return BackendSelection("manual", "profile_not_qualified")
