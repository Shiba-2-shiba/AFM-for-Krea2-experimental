"""Immutable options shared by nodes, runtime and CPU tests."""
from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capabilities import Krea2Capabilities


@dataclass(frozen=True)
class AFMConfig:
    mode: str = "observe"
    implementation: str = "auto"
    strength: float = 0.05
    schedule: str = "curve"
    cutoff: float = 0.25
    preserve_dc: bool = True
    soft_mask: bool = True
    mask_width: float = 0.05
    branch_mode: str = "positive_only"
    block_scope: str = "0-3"
    start_percent: float = 0.0
    end_percent: float = 1.0
    zero_strength_mode: str = "original"
    spectral_diag: str = "off"
    debug_level: str = "summary"
    fail_mode: str = "raise"
    head_chunk_size: int = 2
    query_chunk_size: int = 128
    max_workspace_mib: float = 512.0
    jsonl_path: str = ""
    augmented_precision: str = "activation"
    diagnostic_block_scope: str = "0"
    diagnostic_every_n_steps: int = 1
    diagnostic_head_limit: int = 1
    diagnostic_query_chunk_size: int = 128
    diagnostic_top_k: int = 4
    compare_kernels: bool = False
    benchmark: bool = False
    normalization_mode: str = "joint"
    diagnostic_head_indices: str = ""
    diagnostic_radial_bins: int = 32


def parse_block_scope(scope: str, block_count: int) -> frozenset[int]:
    if scope.strip() == "all":
        return frozenset(range(block_count))
    result = set()
    for part in scope.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if not match:
            raise ValueError(f"Invalid block_scope: {scope!r}; use all, 0-3 or 0,2")
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if not 0 <= start <= end < block_count:
            raise ValueError(f"block_scope {part!r} is outside 0..{block_count - 1}")
        result.update(range(start, end + 1))
    return frozenset(result)


def validate_config(config: AFMConfig, caps: "Krea2Capabilities") -> None:
    enums = {
        "mode": ("off", "discover", "observe", "edit"),
        "normalization_mode": ("joint", "preserve_text_mass"),
        "implementation": ("auto", "manual", "augmented"),
        "schedule": ("curve", "frequency_neutral"),
        "branch_mode": ("positive_only", "negative_only", "both"),
        "zero_strength_mode": ("original", "manual", "augmented"),
        "spectral_diag": ("off", "sampled"),
        "augmented_precision": ("activation", "float32"),
        "debug_level": ("off", "summary", "verbose"),
        "fail_mode": ("raise", "fallback_original"),
    }
    for name, allowed in enums.items():
        if getattr(config, name) not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    for name in ("strength", "cutoff", "mask_width", "start_percent", "end_percent",
                 "max_workspace_mib"):
        if not math.isfinite(getattr(config, name)):
            raise ValueError(f"{name} must be finite")
    if not 0 <= config.start_percent <= config.end_percent <= 1:
        raise ValueError("Require 0 <= start_percent <= end_percent <= 1")
    if not 0 < config.cutoff < 1 or not 0 < config.mask_width <= 1:
        raise ValueError("cutoff must be in (0,1); mask_width must be in (0,1]")
    for name in ("head_chunk_size", "query_chunk_size", "diagnostic_every_n_steps",
                 "diagnostic_head_limit", "diagnostic_query_chunk_size", "diagnostic_top_k",
                 "diagnostic_radial_bins"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.diagnostic_radial_bins > 256:
        raise ValueError("diagnostic_radial_bins must be within 1..256")
    if config.max_workspace_mib <= 0:
        raise ValueError("max_workspace_mib must be positive")
    if config.benchmark and (config.spectral_diag != "off" or config.compare_kernels):
        raise ValueError("Disable spectral/comparison diagnostics for benchmark measurements")
    validate_normalization_options(config)
    parse_diagnostic_heads(config.diagnostic_head_indices, caps.query_heads, config.diagnostic_head_limit)
    parse_block_scope(config.block_scope, caps.block_count)
    parse_block_scope(config.diagnostic_block_scope, caps.block_count)


def parse_diagnostic_heads(indices: str, head_count: int, limit: int) -> tuple[int, ...]:
    """Return physical head indices, independent of processing chunk size."""
    if type(head_count) is not int or head_count <= 0 or type(limit) is not int or limit <= 0:
        raise ValueError("head_count and limit must be positive integers")
    if not isinstance(indices, str):
        raise ValueError("diagnostic_head_indices must be a comma-separated string")
    if not indices.strip():
        return tuple(range(min(limit, head_count)))
    parts = indices.split(",")
    if any(re.fullmatch(r"\s*\d+\s*", part) is None for part in parts):
        raise ValueError("diagnostic_head_indices must contain non-negative integer indices")
    selected = tuple(int(part) for part in parts)
    if len(set(selected)) != len(selected):
        raise ValueError("diagnostic_head_indices must not contain duplicates")
    if any(index >= head_count for index in selected):
        raise ValueError(f"diagnostic_head_indices must be within 0..{head_count - 1}")
    return selected


def validate_normalization_options(config: AFMConfig) -> None:
    """Reject unsupported semantics before any edit/comparison kernel executes."""
    if config.normalization_mode not in ("joint", "preserve_text_mass"):
        raise ValueError("normalization_mode must be joint or preserve_text_mass")
    if (config.mode == "off"
            or (config.mode == "edit" and config.strength == 0
                and config.zero_strength_mode == "original")):
        return
    if config.normalization_mode == "preserve_text_mass":
        from .runtime import AFMUnsupportedError
        if config.compare_kernels:
            raise AFMUnsupportedError(
                "unsupported_normalization_comparison",
                "manual/augmented comparisons require joint normalization",
            )
        requested = config.zero_strength_mode if config.strength == 0 else config.implementation
        if config.mode == "edit" and requested == "augmented":
            raise AFMUnsupportedError(
                "unsupported_normalization_implementation",
                "preserve_text_mass is implemented only by the manual kernel",
            )
