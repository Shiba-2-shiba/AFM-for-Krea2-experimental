"""Equivalent AFM kernel using augmented Q/K/V and bounded SDPA calls."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import AFMConfig
from .attention_math import reject_attention_options, require_finite, validate_joint_inputs
from .memory import ChunkPlan
from .runtime import AFMUnsupportedError, ProgressInfo, TokenLayout
from .spectral import apply_spatial_filter, schedule_alphas


@dataclass(frozen=True)
class AugmentedKV:
    k: torch.Tensor
    v: torch.Tensor
    original_head_dim: int


def build_augmented_kv(k: torch.Tensor, v: torch.Tensor, layout: TokenLayout) -> AugmentedKV:
    if (not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor)
            or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape
            or k.shape[2] != layout.total_tokens):
        raise AFMUnsupportedError("unsupported_layout", "K/V must share [B,H,N,d]")
    if k.device != v.device or k.dtype != v.dtype:
        raise AFMUnsupportedError("unsupported_layout", "K/V device and dtype must match")
    if not k.is_floating_point() or not v.is_floating_point():
        raise AFMUnsupportedError("unsupported_layout", "K/V must be floating point")
    require_finite(k, "K")
    require_finite(v, "V")
    extra_k = torch.zeros_like(k)
    extra_k[..., slice(*layout.text_slice), :] = k[..., slice(*layout.text_slice), :]
    return AugmentedKV(
        k=torch.cat((k, extra_k), dim=-1),
        v=torch.cat((v, torch.zeros_like(v)), dim=-1),
        original_head_dim=k.shape[-1],
    )


def strip_augmented_heads(
    out: torch.Tensor, original_head_dim: int, heads: int, merged: bool
) -> torch.Tensor:
    if type(original_head_dim) is not int or original_head_dim <= 0 or heads <= 0:
        raise ValueError("head dimensions must be positive")
    augmented_dim = 2 * original_head_dim
    if merged:
        if out.ndim != 3 or out.shape[-1] != heads * augmented_dim:
            raise ValueError("merged output must have shape [B,Q,H*2d]")
        separated = out.reshape(out.shape[0], out.shape[1], heads, augmented_dim)
        return separated[..., :original_head_dim].reshape(
            out.shape[0], out.shape[1], heads * original_head_dim
        )
    if out.ndim != 4 or out.shape[1] != heads or out.shape[-1] != augmented_dim:
        raise ValueError("head output must have shape [B,H,Q,2d]")
    return out[..., :original_head_dim]


def augmented_image_rows(
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
    if config.normalization_mode != "joint":
        raise AFMUnsupportedError("unsupported_normalization_implementation",
                                  "augmented attention only implements joint normalization")
    reject_attention_options(mask, is_causal)
    batch, heads, _, dim = validate_joint_inputs(
        q, k, v, layout, chunks, scale, "augmented"
    )
    alpha_lf, alpha_hf = schedule_alphas(config, progress.progress)
    image = slice(*layout.image_slice)
    image_tokens = image.stop - image.start
    result = torch.empty((batch, heads, image_tokens, dim), device=v.device, dtype=v.dtype)
    precision = getattr(config, "augmented_precision", "activation")
    if precision not in ("activation", "float32"):
        raise ValueError("augmented_precision must be activation or float32")

    for sample in range(batch):
        for head_start in range(0, heads, chunks.head_chunk_size):
            head_end = min(head_start + chunks.head_chunk_size, heads)
            with torch.autocast(device_type=q.device.type, enabled=False):
                q_image_activation = q[sample:sample + 1, head_start:head_end, image, :]
                filtered = apply_spatial_filter(
                    q_image_activation, layout.grid, alpha_lf, alpha_hf, config
                )
                delta = filtered - q_image_activation.float()
                require_finite(delta, "augmented Q delta")
                if precision == "float32":
                    q_image = q_image_activation.float()
                    k_heads = k[sample:sample + 1, head_start:head_end].float()
                    v_heads = v[sample:sample + 1, head_start:head_end].float()
                else:
                    q_image = q_image_activation
                    k_heads = k[sample:sample + 1, head_start:head_end]
                    v_heads = v[sample:sample + 1, head_start:head_end]
                kv = build_augmented_kv(k_heads, v_heads, layout)
                for query_start in range(0, image_tokens, chunks.query_chunk_size):
                    query_end = min(query_start + chunks.query_chunk_size, image_tokens)
                    query = torch.cat((
                        q_image[..., query_start:query_end, :],
                        delta[..., query_start:query_end, :].to(q_image.dtype),
                    ), dim=-1)
                    try:
                        output = F.scaled_dot_product_attention(
                            query, kv.k, kv.v, dropout_p=0.0, is_causal=False,
                            scale=float(scale),
                        )
                    except TypeError as exc:
                        raise AFMUnsupportedError(
                            "unsupported_scale", "SDPA does not accept an explicit scale"
                        ) from exc
                    require_finite(output, "augmented attention output")
                    stripped = strip_augmented_heads(output, dim, head_end - head_start, False)
                    result[sample:sample + 1, head_start:head_end,
                           query_start:query_end, :] = stripped.to(v.dtype)
                    del query, output, stripped
                del q_image_activation, filtered, delta, q_image, k_heads, v_heads, kv
    require_finite(result, "augmented attention result")
    return result
