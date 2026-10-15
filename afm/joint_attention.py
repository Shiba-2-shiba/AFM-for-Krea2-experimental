"""Chunked manual joint-attention kernel for selected image query rows."""

import torch

from .config import AFMConfig, validate_normalization_options
from .attention_math import (
    validate_joint_inputs, reject_attention_options, require_finite, preserve_text_partition,
)
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
    validate_normalization_options(config)
    reject_attention_options(mask, is_causal)
    batch, heads, tokens, dim = validate_joint_inputs(q, k, v, layout, chunks, scale, "manual")
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
                original_partition = (text_logits.logsumexp(dim=-1, keepdim=True)
                                      if config.normalization_mode == "preserve_text_mass" else None)
                text_logits = edit_image_text_logits(
                    text_logits, layout.grid, alpha_lf, alpha_hf, config
                )
                if original_partition is not None:
                    text_logits = preserve_text_partition(text_logits, original_partition)
                    del original_partition
                require_finite(text_logits, "edited logits")
                for query_start in range(0, image_tokens, chunks.query_chunk_size):
                    query_end = min(query_start + chunks.query_chunk_size, image_tokens)
                    scores = torch.matmul(
                        q_image[..., query_start:query_end, :], k_heads.transpose(-2, -1)
                    )
                    scores.mul_(scale)
                    scores[..., text] = text_logits[..., query_start:query_end, :]
                    probabilities = torch.softmax(scores, dim=-1)
                    output = torch.matmul(probabilities, v_heads)
                    require_finite(output, "attention output")
                    result[sample:sample + 1, head_start:head_end,
                           query_start:query_end, :] = output.to(v.dtype)
                    del scores, probabilities, output
                del q_image, k_heads, v_heads, text_logits
    require_finite(result, "attention result")
    return result
