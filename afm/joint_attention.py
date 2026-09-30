"""Chunked manual joint-attention kernel for selected image query rows."""
import math

import torch

from .config import AFMConfig
from .memory import ChunkPlan
from .runtime import AFMUnsupportedError, ProgressInfo, TokenLayout
from .spectral import edit_image_text_logits, schedule_alphas


def joint_image_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: TokenLayout,
    config: AFMConfig,
    progress: ProgressInfo,
    chunks: ChunkPlan,
    scale: float,
    *,
    mask=None,
    is_causal: bool = False,
) -> torch.Tensor:
    _reject_attention_options(mask, is_causal)
    batch, heads, tokens, dim = _validate_inputs(q, k, v, layout, chunks, scale, "manual")
    alpha_lf, alpha_hf = schedule_alphas(config, progress.progress)
    text = slice(*layout.text_slice)
    image = slice(*layout.image_slice)
    image_tokens = image.stop - image.start
    result = torch.empty((batch, heads, image_tokens, dim), device=v.device, dtype=v.dtype)

    for sample in range(batch):
        for head_start in range(0, heads, chunks.head_chunk_size):
            head_end = min(head_start + chunks.head_chunk_size, heads)
            with torch.autocast(device_type=q.device.type, enabled=False):
                q_image = q[sample:sample + 1, head_start:head_end, image, :].float()
                k_heads = k[sample:sample + 1, head_start:head_end].float()
                v_heads = v[sample:sample + 1, head_start:head_end].float()
                text_logits = torch.matmul(
                    q_image, k_heads[..., text, :].transpose(-2, -1)
                )
                text_logits.mul_(scale)
                text_logits = edit_image_text_logits(
                    text_logits, layout.grid, alpha_lf, alpha_hf, config
                )
                _require_finite(text_logits, "edited logits")
                for query_start in range(0, image_tokens, chunks.query_chunk_size):
                    query_end = min(query_start + chunks.query_chunk_size, image_tokens)
                    scores = torch.matmul(
                        q_image[..., query_start:query_end, :], k_heads.transpose(-2, -1)
                    )
                    scores.mul_(scale)
                    scores[..., text] = text_logits[..., query_start:query_end, :]
                    probabilities = torch.softmax(scores, dim=-1)
                    output = torch.matmul(probabilities, v_heads)
                    _require_finite(output, "attention output")
                    result[sample:sample + 1, head_start:head_end,
                           query_start:query_end, :] = output.to(v.dtype)
                    del scores, probabilities, output
                del q_image, k_heads, v_heads, text_logits
    _require_finite(result, "attention result")
    return result


def _validate_inputs(q, k, v, layout, chunks, scale, implementation):
    if (not all(isinstance(value, torch.Tensor) and value.ndim == 4 for value in (q, k, v))
            or q.shape != k.shape or q.shape != v.shape):
        raise AFMUnsupportedError("unsupported_layout", "QKV must share [B,H,N,d]")
    batch, heads, tokens, dim = q.shape
    if (layout.text_slice != (0, layout.image_slice[0])
            or layout.image_slice[1] != tokens or layout.total_tokens != tokens
            or layout.image_slice[1] - layout.image_slice[0] != math.prod(layout.grid)):
        raise AFMUnsupportedError("unsupported_layout", "invalid text/image slices")
    if not (q.device == k.device == v.device):
        raise AFMUnsupportedError("unsupported_layout", "QKV devices differ")
    if not all(value.is_floating_point() for value in (q, k, v)):
        raise AFMUnsupportedError("unsupported_layout", "QKV must be floating point")
    if (not isinstance(chunks, ChunkPlan) or chunks.implementation != implementation
            or chunks.head_chunk_size <= 0 or chunks.query_chunk_size <= 0):
        raise ValueError(f"invalid {implementation} ChunkPlan")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise AFMUnsupportedError("unsupported_scale", "scale must be finite")
    _require_finite(q, "Q")
    _require_finite(k, "K")
    _require_finite(v, "V")
    return batch, heads, tokens, dim


def _reject_attention_options(mask, is_causal):
    if mask is not None:
        raise AFMUnsupportedError("unsupported_mask")
    if is_causal:
        raise AFMUnsupportedError("unsupported_causal")


def _require_finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {label}")
