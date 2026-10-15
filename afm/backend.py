"""Joint AFM attention editing with exact-once baseline and atomic replacement."""
import math
import time

import torch

from .config import AFMConfig, parse_block_scope, validate_normalization_options
from .diagnostics import emit_record
from .runtime import (
    AFMUnsupportedError, BACKENDS_KEY, CAPS_KEY, EVAL_KEY, LAYOUT_KEY, TokenLayout,
    _require_state, branch_selection, progress_from_sigmas,
)


def joint_image_rows(*args, **kwargs):
    from .joint_attention import joint_image_rows as kernel
    return kernel(*args, **kwargs)


def augmented_image_rows(*args, **kwargs):
    from .augmented_attention import augmented_image_rows as kernel
    return kernel(*args, **kwargs)


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
        if config.mode not in ("discover", "observe", "edit"):
            raise ValueError("Attention override requires discover, observe or edit")
        validate_normalization_options(config)
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
            state.fallback_calls += 1
            del tensors
            output = dispatch_baseline(backend, q, k, v, heads, kwargs)
            emit_record(self.config, state.run_id, "attention_skipped", block_index=block,
                        model_eval_index=options.get(EVAL_KEY), reason=exc.reason,
                        detail=str(exc), edit_applied=False, baseline_backend=_backend_name(backend))
            return output
        in_scope = self.config.mode == "discover" or block in parse_block_scope(
            self.config.block_scope, options["total_blocks"])
        progress = progress_from_sigmas(options)
        needs_diagnostics = False
        if self.config.spectral_diag != "off" or self.config.compare_kernels:
            from .observation import diagnostic_requested
            needs_diagnostics = diagnostic_requested(self.config, options, progress)
        if not in_scope:
            state.reasons["target_skipped"] += 1
        if not in_scope and not needs_diagnostics:
            del tensors
            return dispatch_baseline(backend, q, k, v, heads, kwargs)
        query, key, value = tensors
        selected, branch_reason = branch_selection(options.get("cond_or_uncond"), query.shape[0],
                                                    self.config.branch_mode)
        # Unrecognized metadata is never guessed to be positive, nor serialized
        # verbatim (it can contain tensors or arbitrary host objects).
        branch_metadata = (None if branch_reason == "unsupported_branch_layout"
                           else list(options["cond_or_uncond"]))
        in_window = progress is not None and (
            self.config.start_percent <= progress.progress <= self.config.end_percent)
        if progress is not None and not in_window:
            state.reasons["outside_window"] += 1
        mask_present = kwargs.get("mask") is not None or kwargs.get("attn_mask") is not None
        original_scale = kwargs.get("scale")
        if original_scale is None:
            original_scale = query.shape[-1] ** -.5
        if (isinstance(original_scale, bool) or not isinstance(original_scale, (int, float))
                or not math.isfinite(original_scale) or original_scale <= 0):
            original_scale = None
        if original_scale is None and self.config.mode != "edit":
            raise AFMUnsupportedError("unsupported_scale", "invalid attention scale")
        metadata = {
            "mode": self.config.mode, "normalization_mode": self.config.normalization_mode,
            "normalization_version": 1, "filter_signature": _filter_signature(self.config),
            "batch_size": query.shape[0], "block_index": block, "total_blocks": options["total_blocks"],
            "step_index": progress.step_index if progress else None,
            "sigma": progress.sigma if progress else None,
            "progress": progress.progress if progress else None,
            "model_eval_index": options.get(EVAL_KEY),
            "q_shape": list(query.shape), "k_shape": list(key.shape), "v_shape": list(value.shape),
            "activation_dtype": str(query.dtype), "device": str(query.device),
            "text_tokens": layout.text_slice[1], "image_tokens": math.prod(layout.grid),
            "grid": list(layout.grid), "img_slice": list(layout.image_slice),
            "gqa_expanded": query.shape[1] == key.shape[1] == value.shape[1],
            "cond_or_uncond": branch_metadata, "selected_batch_rows": list(selected),
            "branch_reason": branch_reason, "in_window": in_window, "mask_present": mask_present,
            "scope": self.config.block_scope, "edit_applied": False,
            "baseline_backend": _backend_name(backend), "preferred_backend": preferred is not None,
            "requested_implementation": self.config.implementation,
            "selected_implementation": "original", "selection_reason": "diagnostic_passthrough",
            "edit_backend": None, "profile_id": None, "original_head_dim": query.shape[-1],
            "augmented_head_dim": None, "original_scale": original_scale,
            "fft_dtype": None, "sdpa_dtype": None, "sdpa_kernel": "unknown",
            "alpha_lf": None, "alpha_hf": None, "max_logits_delta": None,
            "text_attention_mass": None,
            "head_chunk_size": None, "query_chunk_size": None, "estimated_workspace_mib": None,
        }
        skip_edit = (not in_scope or branch_reason == "branch_not_selected"
                     or (progress is not None and not in_window)
                     or (self.config.strength == 0 and self.config.zero_strength_mode == "original"))
        if branch_reason and (self.config.mode != "edit" or skip_edit):
            state.reasons[branch_reason] += 1
        if mask_present and (self.config.mode != "edit" or skip_edit) and not needs_diagnostics:
            state.reasons["unsupported_mask"] += 1
        state.observed_calls += 1
        if self.config.mode == "edit" and not skip_edit:
            return self._edit(backend, q, k, v, heads, kwargs, tensors, layout,
                              progress, selected, branch_reason, state, metadata, needs_diagnostics)
        if self.config.mode == "edit":
            metadata["selection_reason"] = branch_reason or "outside_scope_window_or_zero_original"
        if not needs_diagnostics:
            del tensors, query, key, value
        output = dispatch_baseline(backend, q, k, v, heads, kwargs)
        if needs_diagnostics and selected:
            try:
                _validate_edit_options(kwargs)  # diagnostics must not FFT a mask/bias
                self._emit_diagnostics(tensors, layout, progress, selected, state,
                                       metadata, output, output, kwargs)
            except Exception as exc:
                self._record_failure(state, exc, metadata)
                if self.config.fail_mode == "raise":
                    raise
        emit_record(self.config, state.run_id, "attention", **metadata)
        return output

    def _edit(self, backend, q, k, v, heads, kwargs, tensors, layout, progress,
              selected, branch_reason, state, metadata, needs_diagnostics):
        from .spectral import schedule_alphas
        query, key, value = tensors
        options = kwargs.get("transformer_options") or {}
        # Preflight errors may fallback, but no edit kernel is tried until the
        # full live-tensor/chunk budget has been checked.
        try:
            _validate_edit_options(kwargs)
            if metadata["original_scale"] is None:
                raise AFMUnsupportedError("unsupported_scale", "invalid attention scale")
            if branch_reason:
                raise AFMUnsupportedError(branch_reason)
            if progress is None:
                raise AFMUnsupportedError("unsupported_progress")
            signature = _runtime_signature(query, layout, self.config, state, options, metadata)
            choice, chunks = _select_edit_plan(self.config, signature, state.profiles,
                                               tuple(query.shape), tuple(key.shape), layout, query.dtype)
            state.reasons[choice.reason] += 1
            metadata.update(selected_implementation=choice.implementation,
                            selection_reason=choice.reason, profile_id=choice.profile_id)
            if query.device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(query.device)
                if chunks.estimated_workspace_mib * 1024 ** 2 > free_bytes:
                    raise AFMUnsupportedError("memory_guard", "estimated AFM workspace exceeds free GPU memory")
            alpha_lf, alpha_hf = schedule_alphas(self.config, progress.progress)
            metadata.update(
                alpha_lf=alpha_lf, alpha_hf=alpha_hf, fft_dtype="torch.float32",
                augmented_head_dim=2 * query.shape[-1] if choice.implementation == "augmented" else None,
                sdpa_dtype=("torch.float32" if self.config.augmented_precision == "float32" else str(query.dtype))
                           if choice.implementation == "augmented" else None,
                edit_backend="torch_sdpa" if choice.implementation == "augmented" else "manual_fp32",
                head_chunk_size=chunks.head_chunk_size, query_chunk_size=chunks.query_chunk_size,
                estimated_workspace_mib=chunks.estimated_workspace_mib,
            )
        except Exception as exc:
            self._record_failure(state, exc, metadata)
            if self.config.fail_mode == "raise":
                raise
            baseline = dispatch_baseline(backend, q, k, v, heads, kwargs)
            emit_record(self.config, state.run_id, "attention", **metadata)
            return baseline

        # A backend failure is not an AFM failure and must never be retried.
        baseline = dispatch_baseline(backend, q, k, v, heads, kwargs)
        candidate = None
        try:
            if state.benchmark_device is not None:
                torch.cuda.synchronize(state.benchmark_device)
            edit_started = time.perf_counter()
            merged = not kwargs.get("skip_output_reshape", False)
            batch, head_count, tokens, dim = query.shape
            expected_shape = (batch, tokens, head_count * dim) if merged else tuple(query.shape)
            if not isinstance(baseline, torch.Tensor) or tuple(baseline.shape) != expected_shape:
                raise AFMUnsupportedError("unsupported_output_layout", "baseline shape does not match heads")
            if not torch.isfinite(baseline).all():
                raise FloatingPointError("nonfinite original attention output")
            candidate = baseline.clone()
            kernel = augmented_image_rows if choice.implementation == "augmented" else joint_image_rows
            start, end = layout.image_slice
            for row in selected:
                image_rows = kernel(query[row:row + 1], key[row:row + 1], value[row:row + 1],
                                    layout, self.config, progress, chunks, metadata["original_scale"])
                expected = (1, head_count, end - start, dim)
                if tuple(image_rows.shape) != expected:
                    raise AFMUnsupportedError("unsupported_output_layout", "edit kernel returned wrong shape")
                if not torch.isfinite(image_rows).all():
                    raise FloatingPointError("nonfinite edited image rows")
                if merged:
                    candidate[row:row + 1, start:end] = image_rows.transpose(1, 2).reshape(1, end - start, -1)
                else:
                    candidate[row:row + 1, :, start:end] = image_rows
                del image_rows
            if needs_diagnostics:
                self._emit_diagnostics(tensors, layout, progress, selected, state,
                                       metadata, baseline, candidate, kwargs, chunks=chunks)
            if state.benchmark_device is not None:
                torch.cuda.synchronize(state.benchmark_device)
                metadata["edit_seconds"] = time.perf_counter() - edit_started
                state.edit_seconds += metadata["edit_seconds"]
        except Exception as exc:
            candidate = None  # discard every partial edit; keep only original baseline
            self._record_failure(state, exc, metadata)
            if self.config.fail_mode == "raise":
                raise
            emit_record(self.config, state.run_id, "attention", **metadata)
            return baseline
        state.edited_calls += 1
        metadata["edit_applied"] = True
        emit_record(self.config, state.run_id, "attention", **metadata)
        return candidate

    def _emit_diagnostics(self, tensors, layout, progress, selected, state,
                          metadata, baseline, candidate, kwargs, chunks=None):
        from .memory import choose_chunks
        from .observation import collect_attention_diagnostics
        query, key, value = tensors
        if chunks is None:
            chunks = choose_chunks(tuple(query.shape), tuple(key.shape), layout, self.config,
                                   "augmented" if self.config.compare_kernels else "manual", query.dtype)
        if query.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(query.device)
            if chunks.estimated_workspace_mib * 1024 ** 2 > free_bytes:
                raise AFMUnsupportedError("memory_guard", "diagnostic workspace exceeds free GPU memory")
        edited = candidate is not baseline
        preserved = True
        if edited:
            for row in range(query.shape[0]):
                if row not in selected:
                    preserved &= torch.equal(candidate[row], baseline[row])
                elif kwargs.get("skip_output_reshape", False):
                    preserved &= torch.equal(candidate[row, :, :layout.image_slice[0]],
                                             baseline[row, :, :layout.image_slice[0]])
                else:
                    preserved &= torch.equal(candidate[row, :layout.image_slice[0]],
                                             baseline[row, :layout.image_slice[0]])
        records = collect_attention_diagnostics(*tensors, layout, self.config, progress, chunks,
                                                metadata["original_scale"], selected, edit_applied=edited)
        common = {name: metadata[name] for name in (
            "step_index", "model_eval_index", "block_index", "mode", "progress",
            "normalization_mode", "normalization_version", "selected_implementation",
            "filter_signature", "cond_or_uncond", "selected_batch_rows", "batch_size",
        )}
        branches = metadata["cond_or_uncond"]
        samples_per_branch = query.shape[0] // len(branches)
        for record in records:
            event = record.pop("event")
            row = record["branch"]
            fields = {**common, **record, "edit_applied": edited,
                      "unselected_rows_preserved": bool(preserved),
                      "branch_kind": branches[row // samples_per_branch],
                      "sample_index": row % samples_per_branch}
            emit_record(self.config, state.run_id, event, **fields)

    def _record_failure(self, state, error, metadata):
        if isinstance(error, AFMUnsupportedError):
            reason = error.reason
        elif isinstance(error, torch.OutOfMemoryError):
            reason = "oom"
            state.oom_events += 1
        elif isinstance(error, FloatingPointError):
            reason = "numerical_failure"
            state.nan_events += 1
        else:
            reason = "kernel_failure"
        state.reasons[reason] += 1
        metadata.update(edit_applied=False, failure_reason=reason, error_type=type(error).__name__)
        if self.config.fail_mode == "fallback_original":
            state.reasons["fallback_original"] += 1
            state.fallback_calls += 1
            metadata["fallback_original"] = True


def _validate_edit_options(kwargs):
    if kwargs.get("mask") is not None or kwargs.get("attn_mask") is not None:
        raise AFMUnsupportedError("unsupported_mask")
    if kwargs.get("is_causal", False):
        raise AFMUnsupportedError("unsupported_causal")
    if kwargs.get("dropout_p", 0) != 0:
        raise AFMUnsupportedError("unsupported_dropout")
    if kwargs.get("enable_gqa", False):
        raise AFMUnsupportedError("unsupported_gqa", "Krea2 already expands KV heads")
    allowed = {"mask", "attn_mask", "is_causal", "dropout_p", "enable_gqa", "scale",
               "skip_reshape", "skip_output_reshape", "attn_precision", "transformer_options",
               "_inside_attn_wrapper", "preferred_attention", "low_precision_attention"}
    if set(kwargs) - allowed:
        raise AFMUnsupportedError("unsupported_attention_options", ", ".join(sorted(set(kwargs) - allowed)))


def _select_edit_plan(config, signature, profiles, q_shape, k_shape, layout, dtype):
    from .memory import choose_chunks
    from .selection import BackendSelection, select_implementation
    is_auto = (config.strength != 0 and config.implementation == "auto"
               and config.normalization_mode == "joint")
    if is_auto and profiles and signature.get("sdpa_explicit_scale", False):
        try:
            augmented_plan = choose_chunks(q_shape, k_shape, layout, config, "augmented", dtype)
        except AFMUnsupportedError as exc:
            if exc.reason != "memory_guard":
                raise
            # A preflight budget rejection is not a mid-kernel retry.
            choice = BackendSelection("manual", "augmented_memory_guard")
        else:
            actual_signature = {**signature, "head_chunk_size": augmented_plan.head_chunk_size,
                                "query_chunk_size": augmented_plan.query_chunk_size}
            choice = select_implementation(config, actual_signature, profiles)
            if choice.implementation == "augmented":
                return choice, augmented_plan
    else:
        choice = select_implementation(config, signature, profiles)
    return choice, choose_chunks(q_shape, k_shape, layout, config, choice.implementation, dtype)


def _runtime_signature(query, layout, config, state, options, metadata):
    from .capabilities import _sdpa_has_explicit_scale
    signature = dict(state.environment)
    caps = options.get(CAPS_KEY)
    signature.update(
        sdpa_explicit_scale=caps.sdpa_explicit_scale if caps is not None else _sdpa_has_explicit_scale(),
        device_type=query.device.type, activation_dtype=str(query.dtype),
        original_head_dim=query.shape[-1], baseline_backend=metadata["baseline_backend"],
        text_tokens=layout.text_slice[1], image_tokens=math.prod(layout.grid),
        batch=query.shape[0], heads=query.shape[1], grid_h=layout.grid[0], grid_w=layout.grid[1],
        head_chunk_size=config.head_chunk_size, query_chunk_size=config.query_chunk_size,
        strength=config.strength,
        filter_signature=_filter_signature(config),
    )
    signature["sdpa_settings"] = {
        name: bool(getattr(torch.backends.cuda, name + "_sdp_enabled")())
        for name in ("flash", "math", "mem_efficient", "cudnn")
        if callable(getattr(torch.backends.cuda, name + "_sdp_enabled", None))
    }
    if query.device.type == "cuda":
        signature["gpu"] = torch.cuda.get_device_name(query.device)
        signature["compute_capability"] = list(torch.cuda.get_device_capability(query.device))
    return signature


def _filter_signature(config):
    return {"version": "radial_v1", "cutoff": config.cutoff, "soft_mask": config.soft_mask,
            "mask_width": config.mask_width, "preserve_dc": config.preserve_dc,
            "schedule": config.schedule, "augmented_precision": config.augmented_precision,
            "normalization_mode": config.normalization_mode, "normalization_version": 1}
