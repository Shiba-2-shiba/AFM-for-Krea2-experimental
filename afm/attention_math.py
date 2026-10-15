"""Shared validation and FP32 normalization for AFM attention kernels."""
import math

import torch

from .memory import ChunkPlan
from .runtime import AFMUnsupportedError


def validate_joint_inputs(q, k, v, layout, chunks, scale, implementation):
    if (not all(isinstance(value, torch.Tensor) and value.ndim == 4 for value in (q, k, v))
            or q.shape != k.shape or q.shape != v.shape):
        raise AFMUnsupportedError("unsupported_layout", "QKV must share shape [B,H,N,d]")
    batch, heads, tokens, dim = q.shape
    if (min(batch, heads, tokens, dim) <= 0
            or layout.text_slice != (0, layout.image_slice[0])
            or layout.text_slice[1] <= 0
            or layout.image_slice[1] != tokens or layout.total_tokens != tokens
            or layout.image_slice[1] - layout.image_slice[0] != math.prod(layout.grid)):
        raise AFMUnsupportedError("unsupported_layout", "invalid text/image slices")
    if not (q.device == k.device == v.device):
        raise AFMUnsupportedError("unsupported_layout", "QKV devices differ")
    if not (q.dtype == k.dtype == v.dtype):
        raise AFMUnsupportedError("unsupported_layout", "QKV dtype must match")
    if not all(value.is_floating_point() for value in (q, k, v)):
        raise AFMUnsupportedError("unsupported_layout", "QKV must be floating point")
    if (not isinstance(chunks, ChunkPlan) or chunks.implementation != implementation
            or chunks.head_chunk_size <= 0 or chunks.query_chunk_size <= 0):
        raise ValueError(f"invalid {implementation} ChunkPlan")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise AFMUnsupportedError("unsupported_scale", "scale must be positive and finite")
    require_finite(q, "Q")
    require_finite(k, "K")
    require_finite(v, "V")
    return batch, heads, tokens, dim


def reject_attention_options(mask, is_causal):
    if mask is not None:
        raise AFMUnsupportedError("unsupported_mask")
    if is_causal:
        raise AFMUnsupportedError("unsupported_causal")


def require_finite(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {label}")


def preserve_text_partition(
    edited_logits: torch.Tensor, original_log_partition: torch.Tensor
) -> torch.Tensor:
    """Restore the text partition, leaving its conditional distribution intact.

    Both inputs are FP32 and use already-scaled scores. No input is mutated.
    The caller computes the original partition before filtering the text logits.
    """
    if (not isinstance(edited_logits, torch.Tensor) or edited_logits.ndim != 4
            or edited_logits.shape[-1] <= 0
            or not isinstance(original_log_partition, torch.Tensor)
            or original_log_partition.shape != (*edited_logits.shape[:-1], 1)):
        raise ValueError("expected edited [B,H,I,T] and log partition [B,H,I,1]")
    if (edited_logits.dtype != torch.float32 or original_log_partition.dtype != torch.float32
            or edited_logits.device != original_log_partition.device):
        raise ValueError("text partition correction requires same-device float32 (FP32) tensors")
    require_finite(edited_logits, "edited text logits")
    require_finite(original_log_partition, "original text log partition")
    with torch.autocast(device_type=edited_logits.device.type, enabled=False):
        correction = original_log_partition - edited_logits.logsumexp(dim=-1, keepdim=True)
        corrected = edited_logits + correction
    require_finite(corrected, "partition-corrected text logits")
    return corrected
