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
        "implementation": ("auto", "manual", "augmented"),
        "schedule": ("curve",),
        "branch_mode": ("positive_only", "negative_only", "both"),
        "zero_strength_mode": ("original", "manual", "augmented"),
        "spectral_diag": ("off",),
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
    for name in ("head_chunk_size", "query_chunk_size"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.max_workspace_mib <= 0:
        raise ValueError("max_workspace_mib must be positive")
    parse_block_scope(config.block_scope, caps.block_count)
