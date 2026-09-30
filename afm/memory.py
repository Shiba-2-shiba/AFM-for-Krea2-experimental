"""Conservative AFM workspace estimates and chunk selection."""
from dataclasses import dataclass
import math

import torch

from .config import AFMConfig
from .runtime import AFMUnsupportedError, TokenLayout

_MIB = 2 ** 20


@dataclass(frozen=True)
class ChunkPlan:
    head_chunk_size: int
    query_chunk_size: int
    estimated_workspace_mib: float
    implementation: str


def estimate_full_joint_logits_mib(
    batch: int, heads: int, image_tokens: int, text_tokens: int
) -> float:
    _positive_dimensions(batch, heads, image_tokens, text_tokens)
    total = image_tokens + text_tokens
    return batch * heads * total * total * 4 / _MIB


def choose_chunks(
    q_shape: tuple,
    k_shape: tuple,
    layout: TokenLayout,
    config: AFMConfig,
    implementation: str,
    activation_dtype: torch.dtype,
) -> ChunkPlan:
    if implementation not in ("manual", "augmented"):
        raise ValueError("implementation must be manual or augmented")
    if len(q_shape) != 4 or tuple(q_shape) != tuple(k_shape):
        raise AFMUnsupportedError("unsupported_layout", "Q/K shapes must match [B,H,N,d]")
    batch, heads, tokens, dim = map(int, q_shape)
    _positive_dimensions(batch, heads, tokens, dim)
    text_tokens, image_tokens = _validate_layout(layout, tokens)
    try:
        activation_bytes = torch.empty((), dtype=activation_dtype).element_size()
    except (TypeError, RuntimeError) as exc:
        raise AFMUnsupportedError("unsupported_layout", "invalid activation dtype") from exc
    if not torch.empty((), dtype=activation_dtype).is_floating_point():
        raise AFMUnsupportedError("unsupported_layout", "activation dtype must be floating point")

    max_heads = min(config.head_chunk_size, heads)
    max_queries = min(config.query_chunk_size, image_tokens)
    query_floor = min(32, max_queries)
    query_candidates = [max_queries]
    while query_candidates[-1] > query_floor:
        query_candidates.append(max(query_floor, query_candidates[-1] // 2))

    budget = config.max_workspace_mib
    augmented_bytes = (4 if implementation == "augmented"
                       and getattr(config, "augmented_precision", "activation") == "float32"
                       else activation_bytes)
    caller_kv_conversion_bytes = (
        4 if implementation == "augmented"
        and getattr(config, "augmented_precision", "activation") == "float32"
        and activation_dtype != torch.float32 else 0
    )
    diagnostic_heads = min(max(int(getattr(config, "diagnostic_head_limit", 1)), 1), heads)
    diagnostic_queries = min(
        max(int(getattr(config, "diagnostic_query_chunk_size", 128)), 1), image_tokens
    )
    diagnostics_enabled = (getattr(config, "spectral_diag", "off") != "off"
                           or bool(getattr(config, "compare_kernels", False)))
    compare_kernels = bool(getattr(config, "compare_kernels", False))
    smallest = None
    for head_chunk in range(max_heads, 0, -1):
        for query_chunk in query_candidates:
            estimate = _workspace_bytes(
                batch, heads, tokens, dim, image_tokens, text_tokens,
                head_chunk, query_chunk, activation_bytes, augmented_bytes,
                caller_kv_conversion_bytes, implementation,
                diagnostic_heads, diagnostic_queries, diagnostics_enabled, compare_kernels,
            ) / _MIB
            smallest = estimate if smallest is None else min(smallest, estimate)
            if estimate <= budget:
                return ChunkPlan(head_chunk, query_chunk, estimate, implementation)
    raise AFMUnsupportedError(
        "memory_guard",
        f"minimum estimated workspace {smallest:.3f} MiB exceeds {budget:.3f} MiB",
    )


def _workspace_bytes(batch, heads, tokens, dim, image_tokens, text_tokens,
                     head_chunk, query_chunk, activation_bytes, augmented_bytes,
                     caller_kv_conversion_bytes, implementation, diagnostic_heads,
                     diagnostic_queries, diagnostics_enabled, compare_kernels):
    # These tensors remain live for the full input batch even though kernels loop by sample.
    retained_qkv = 3 * batch * heads * tokens * dim * activation_bytes
    baseline_and_clone = 2 * batch * heads * tokens * dim * activation_bytes
    image_rows = batch * heads * image_tokens * dim * activation_bytes
    fixed = retained_qkv + baseline_and_clone + image_rows

    if implementation == "manual":
        scratch = _manual_scratch(
            head_chunk, query_chunk, tokens, dim, image_tokens, text_tokens
        )
    else:
        scratch = _augmented_scratch(
            head_chunk, query_chunk, tokens, dim, image_tokens, augmented_bytes,
            caller_kv_conversion_bytes,
        )
    diagnostic = 0
    if diagnostics_enabled:
        # One sampled image: original FP32 logits, two complex FFT buffers,
        # restored logits, and bounded full-key probability diagnostics.
        diagnostic += diagnostic_heads * image_tokens * text_tokens * (4 + 24 + 4)
        diagnostic += image_tokens * 4 * 4  # radius, low, high and gain masks
        diagnostic += 2 * diagnostic_heads * diagnostic_queries * tokens * 4
    if compare_kernels:
        # Metrics retain outputs, differences and effects in float64. Sampled
        # FP32 QKV and the larger comparison kernel scratch can coexist.
        diagnostic += 12 * 8 * diagnostic_heads * image_tokens * dim
        diagnostic += 3 * diagnostic_heads * tokens * dim * 4
        diagnostic += max(
            _manual_scratch(diagnostic_heads, diagnostic_queries, tokens, dim,
                            image_tokens, text_tokens),
            _augmented_scratch(diagnostic_heads, diagnostic_queries, tokens, dim,
                               image_tokens, 4, 0),
        )
    return fixed + scratch + diagnostic


def _manual_scratch(heads, queries, tokens, dim, image_tokens, text_tokens):
    image_q_fp32 = heads * image_tokens * dim * 4
    text_logits = heads * image_tokens * text_tokens * 4
    # Forward spectrum, multiplied spectrum, inverse complex result, real output.
    fft = heads * image_tokens * text_tokens * (24 + 4)
    masks = image_tokens * 4 * 4
    key_value_fp32 = 2 * heads * tokens * dim * 4
    scores_probabilities = 2 * heads * queries * tokens * 4
    chunk_output = heads * queries * dim * 4
    return (image_q_fp32 + text_logits + fft + masks + key_value_fp32
            + scores_probabilities + chunk_output)


def _augmented_scratch(heads, queries, tokens, dim, image_tokens, augmented_bytes,
                       caller_kv_conversion_bytes):
    image_q_fp32 = heads * image_tokens * dim * 4
    fft = heads * image_tokens * dim * (24 + 4)
    masks = image_tokens * 4 * 4
    delta = heads * image_tokens * dim * 4
    # build_augmented_kv retains extra_k while allocating augmented K, then
    # evaluates zero-V and augmented V before returning. Peak: K/V finals
    # (4*N*d) plus both N*d construction temporaries.
    augmented_kv = 6 * heads * tokens * dim * augmented_bytes
    # DeltaQ dtype conversion remains live while torch.cat allocates [Q,DeltaQ].
    augmented_q = 3 * heads * queries * dim * augmented_bytes
    # float32 diagnostic precision converts the current head group's K and V
    # in the caller; these copies coexist with augmentation and SDPA buffers.
    caller_kv = 2 * heads * tokens * dim * caller_kv_conversion_bytes
    scores_probabilities = 2 * heads * queries * tokens * 4
    # Conservative math-SDPA FP32 copies of Q/K/V plus its FP32 output.
    math_qkv = heads * (queries + 2 * tokens) * (2 * dim) * 4
    math_output = heads * queries * (2 * dim) * 4
    return (image_q_fp32 + fft + masks + delta + augmented_kv + augmented_q + caller_kv
            + scores_probabilities + math_qkv + math_output)


def _validate_layout(layout, tokens):
    if not isinstance(layout, TokenLayout):
        raise AFMUnsupportedError("unsupported_layout", "missing TokenLayout")
    text_start, text_end = layout.text_slice
    image_start, image_end = layout.image_slice
    image_tokens = math.prod(layout.grid)
    if (text_start != 0 or image_start != text_end or image_end != tokens
            or layout.total_tokens != tokens or image_end - image_start != image_tokens):
        raise AFMUnsupportedError("unsupported_layout", "non-contiguous text/image layout")
    return text_end, image_tokens


def _positive_dimensions(*values):
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("dimensions must be positive integers")
