"""Diagnostic attention interception with exact-once baseline dispatch."""
import math

import torch

from .config import AFMConfig, parse_block_scope
from .diagnostics import emit_record
from .runtime import (
    AFMUnsupportedError, BACKENDS_KEY, EVAL_KEY, LAYOUT_KEY, TokenLayout,
    _require_state, branch_selection, progress_from_sigmas,
)


def _is_container(value) -> bool:
    return (not isinstance(value, torch.Tensor) and callable(getattr(value, "peek", None))
            and callable(getattr(value, "take", None)))


def dispatch_baseline(backend, q, k, v, heads: int, kwargs: dict) -> torch.Tensor:
    """Dispatch an already selected backend; never retry a consumed container."""
    options = dict(kwargs)
    transformer_options = dict(options.get("transformer_options") or {})
    transformer_options.pop("optimized_attention_override", None)
    options["transformer_options"] = transformer_options
    options.pop("preferred_attention", None)
    options["_inside_attn_wrapper"] = True
    containers = tuple(_is_container(value) for value in (q, k, v))
    if any(containers):
        if not all(containers):
            raise TypeError("q, k and v must all be attention containers")
        entry = getattr(backend, "container_function", None)
        if callable(entry):
            return entry(q, k, v, heads, **options)
        q, k, v = q.take(), k.take(), v.take()
    return backend(q, k, v, heads, **options)


def _backend_name(backend) -> str:
    return f"{getattr(backend, '__module__', 'unknown')}.{getattr(backend, '__name__', type(backend).__name__)}"


def _validate_attention(q, k, v, heads, options, kwargs) -> TokenLayout:
    layout = options.get(LAYOUT_KEY)
    if not isinstance(layout, TokenLayout):
        raise AFMUnsupportedError("unsupported_layout", "missing validated post_input layout")
    block = options.get("block_index")
    total = options.get("total_blocks")
    if (options.get("block_type") != "single" or type(block) is not int
            or type(total) is not int or not 0 <= block < total
            or len(options.get(BACKENDS_KEY, ())) != total):
        raise AFMUnsupportedError("unsupported_layout", "not a native Krea2 body block")
    if tuple(options.get("img_slice", ())) != layout.image_slice:
        raise AFMUnsupportedError("unsupported_layout", "img_slice mismatch")
    if (not kwargs.get("skip_reshape", False)
            or not all(isinstance(value, torch.Tensor) and value.ndim == 4 for value in (q, k, v))):
        raise AFMUnsupportedError("unsupported_layout", "expected post-RoPE head-format QKV")
    if (q.shape != k.shape or q.shape != v.shape or q.shape[1] != heads
            or q.shape[2] != layout.total_tokens or q.shape[0] <= 0 or q.shape[3] <= 0):
        raise AFMUnsupportedError("unsupported_layout", "joint QKV or expanded GQA shape mismatch")
    if not (q.device == k.device == v.device and q.dtype == k.dtype == v.dtype):
        raise AFMUnsupportedError("unsupported_layout", "QKV dtype/device mismatch")
    return layout


class Krea2AFMAttentionOverride:
    def __init__(self, config: AFMConfig):
        if config.mode not in ("discover", "observe"):
            raise NotImplementedError("Attention override is diagnostic-only")
        self.config = config

    def __call__(self, original_func, q, k, v, heads, **kwargs):
        return self._observe(original_func, q, k, v, heads, kwargs)

    def container_function(self, q, k, v, heads, **kwargs):
        # Current ComfyUI's container override entry does not pass original_func
        # or preferred_attention. The diffusion wrapper supplies the live module
        # backend; this global function is used only if the module has none.
        from comfy.ldm.modules.attention import optimized_attention_masked
        return self._observe(optimized_attention_masked, q, k, v, heads, kwargs)

    def _observe(self, original_func, q, k, v, heads, kwargs):
        options = kwargs.get("transformer_options") or {}
        state = _require_state(options)
        state.attention_calls += 1
        block = options.get("block_index")
        backends = options.get(BACKENDS_KEY, ())
        preferred = backends[block] if type(block) is int and 0 <= block < len(backends) else None
        backend = preferred if preferred is not None else original_func
        if not callable(backend):
            raise RuntimeError("No original Krea2 attention backend available")
        # Only scalar metadata survives inspection. Containers stay backend-owned.
        tensors = tuple(value.peek() if _is_container(value) else value for value in (q, k, v))
        try:
            layout = _validate_attention(*tensors, heads, options, kwargs)
        except AFMUnsupportedError as exc:
            state.reasons[exc.reason] += 1
            if self.config.fail_mode == "raise":
                raise
            state.reasons["fallback_original"] += 1
            del tensors
            output = dispatch_baseline(backend, q, k, v, heads, kwargs)
            emit_record(self.config, state.run_id, "attention_skipped", block_index=block,
                        model_eval_index=options.get(EVAL_KEY), reason=exc.reason,
                        detail=str(exc), edit_applied=False, baseline_backend=_backend_name(backend))
            return output
        in_scope = self.config.mode == "discover" or block in parse_block_scope(
            self.config.block_scope, options["total_blocks"])
        if not in_scope:
            state.reasons["target_skipped"] += 1
            del tensors
            return dispatch_baseline(backend, q, k, v, heads, kwargs)
        query, key, value = tensors
        selected, branch_reason = branch_selection(options.get("cond_or_uncond"), query.shape[0],
                                                    self.config.branch_mode)
        if branch_reason:
            state.reasons[branch_reason] += 1
        progress = progress_from_sigmas(options)
        in_window = progress is not None and (
            self.config.start_percent <= progress.progress <= self.config.end_percent)
        if progress is not None and not in_window:
            state.reasons["outside_window"] += 1
        mask_present = kwargs.get("mask") is not None
        if mask_present:
            state.reasons["unsupported_mask"] += 1
        original_scale = kwargs.get("scale")
        if original_scale is None:
            original_scale = query.shape[-1] ** -.5
        original_scale = float(original_scale)
        if not math.isfinite(original_scale):
            raise AFMUnsupportedError("unsupported_scale", "nonfinite attention scale")
        metadata = {
            "mode": self.config.mode, "block_index": block, "total_blocks": options["total_blocks"],
            "step_index": progress.step_index if progress else None,
            "sigma": progress.sigma if progress else None,
            "progress": progress.progress if progress else None,
            "model_eval_index": options.get(EVAL_KEY),
            "q_shape": list(query.shape), "k_shape": list(key.shape), "v_shape": list(value.shape),
            "activation_dtype": str(query.dtype), "device": str(query.device),
            "text_tokens": layout.text_slice[1], "image_tokens": math.prod(layout.grid),
            "grid": list(layout.grid), "img_slice": list(layout.image_slice),
            "gqa_expanded": query.shape[1] == key.shape[1] == value.shape[1],
            "cond_or_uncond": options.get("cond_or_uncond"), "selected_batch_rows": list(selected),
            "branch_reason": branch_reason, "in_window": in_window, "mask_present": mask_present,
            "scope": self.config.block_scope, "edit_applied": False,
            "baseline_backend": _backend_name(backend), "preferred_backend": preferred is not None,
            "requested_implementation": self.config.implementation,
            "selected_implementation": "original", "selection_reason": "diagnostic_passthrough",
            "edit_backend": None, "profile_id": None, "original_head_dim": query.shape[-1],
            "augmented_head_dim": None, "original_scale": original_scale,
            "fft_dtype": None, "sdpa_dtype": None, "sdpa_kernel": "unknown",
            "alpha_lf": None, "alpha_hf": None, "max_logits_delta": None,
            "rho_text_conditional": None, "text_attention_mass": None,
            "head_chunk_size": None, "query_chunk_size": None, "estimated_workspace_mib": None,
        }
        del tensors, query, key, value
        output = dispatch_baseline(backend, q, k, v, heads, kwargs)
        state.observed_calls += 1
        emit_record(self.config, state.run_id, "attention", **metadata)
        return output
