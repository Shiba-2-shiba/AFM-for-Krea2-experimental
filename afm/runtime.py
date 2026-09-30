"""Run-local wrappers and verified Krea2 token layouts."""
from collections import Counter
from dataclasses import dataclass, field
import time
from uuid import uuid4

import torch

from .capabilities import CONFIG_KEY, WRAPPER_KEY, detect_krea2, detect_patch_conflicts
from .config import AFMConfig, validate_config
from .diagnostics import configuration_metadata, emit_record, environment_metadata

RUN_KEY = "krea2_afm_run"
LAYOUT_KEY = "krea2_afm_layout"
BACKENDS_KEY = "krea2_afm_backends"
EVAL_KEY = "krea2_afm_model_eval_index"
SKIP_KEY = "krea2_afm_skip"


class AFMUnsupportedError(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class TokenLayout:
    text_slice: tuple[int, int]
    image_slice: tuple[int, int]
    grid: tuple[int, int]
    total_tokens: int
    reference_tokens: int = 0


@dataclass(frozen=True)
class ProgressInfo:
    step_index: int
    num_steps: int
    progress: float
    sigma: float


@dataclass
class RunState:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    model_evaluations: int = 0
    attention_calls: int = 0
    observed_calls: int = 0
    edited_calls: int = 0
    reasons: Counter = field(default_factory=Counter)
    evaluations_per_step: Counter = field(default_factory=Counter)
    finished: bool = False

    def next_evaluation(self, progress: ProgressInfo | None) -> int:
        key = progress.step_index if progress else -1
        index = self.evaluations_per_step[key]
        self.evaluations_per_step[key] += 1
        self.model_evaluations += 1
        return index


def layout_from_latent(x: torch.Tensor, patch_size: int, text_tokens: int) -> TokenLayout:
    if x.ndim not in (4, 5):
        raise AFMUnsupportedError("unsupported_layout", "expected 4D or single-frame 5D latent")
    if x.ndim == 5 and x.shape[2] != 1:
        raise AFMUnsupportedError("multiple_frames")
    if type(patch_size) is not int or patch_size <= 0 or text_tokens <= 0:
        raise AFMUnsupportedError("unsupported_layout", "invalid patch or text length")
    if any(dimension <= 0 for dimension in x.shape):
        raise AFMUnsupportedError("unsupported_layout", "empty latent dimension")
    h, w = ((int(size) + patch_size - 1) // patch_size for size in x.shape[-2:])
    return TokenLayout((0, text_tokens), (text_tokens, text_tokens + h * w),
                       (h, w), text_tokens + h * w)


def validate_post_input(payload: dict, layout: TokenLayout) -> TokenLayout:
    if layout.reference_tokens or payload.get("transformer_options", {}).get(
        "reference_image_num_tokens"
    ):
        raise AFMUnsupportedError("reference_not_supported")
    img, txt, ids, txt_ids = (payload.get(key) for key in ("img", "txt", "img_ids", "txt_ids"))
    if not all(isinstance(value, torch.Tensor) and value.ndim == 3
               for value in (img, txt, ids, txt_ids)):
        raise AFMUnsupportedError("unsupported_layout", "post_input requires rank-3 tensors")
    batch, image_tokens, features = img.shape
    h, w = layout.grid
    text_tokens = layout.text_slice[1]
    if (image_tokens != h * w or txt.shape != (batch, text_tokens, features)
            or ids.shape != (batch, h * w, 3) or txt_ids.shape != (batch, text_tokens, 3)):
        raise AFMUnsupportedError("unsupported_layout", "post_input shape/token mismatch")
    if not torch.isfinite(ids).all() or not torch.isfinite(txt_ids).all():
        raise AFMUnsupportedError("unsupported_layout", "nonfinite position ids")
    if torch.any(ids[..., 0] != 0):
        raise AFMUnsupportedError("reference_not_supported", "nonzero source index")
    rows = torch.arange(h, device=ids.device, dtype=ids.dtype).repeat_interleave(w)
    cols = torch.arange(w, device=ids.device, dtype=ids.dtype).repeat(h)
    if (not torch.equal(ids[..., 1], rows.expand(batch, -1))
            or not torch.equal(ids[..., 2], cols.expand(batch, -1))
            or torch.any(txt_ids != 0)):
        raise AFMUnsupportedError("unsupported_layout", "position ids are not native row-major order")
    return layout


def progress_from_sigmas(options: dict) -> ProgressInfo | None:
    try:
        schedule = torch.as_tensor(options["sample_sigmas"]).detach().to("cpu", torch.float64)
        sigma = torch.as_tensor(options["sigmas"]).detach().to("cpu", torch.float64).flatten()
        if (schedule.ndim != 1 or schedule.numel() < 2 or sigma.numel() == 0
                or not torch.isfinite(schedule).all() or not torch.isfinite(sigma).all()
                or not torch.all(schedule[:-1] > schedule[1:])
                or not torch.all(sigma == sigma[0])):
            return None
        matches = torch.isclose(schedule[:-1], sigma[0], rtol=1e-5, atol=1e-6).nonzero().flatten()
        if matches.numel() != 1:
            return None
        index = int(matches[0])
        steps = schedule.numel() - 1
        return ProgressInfo(index, steps, index / max(steps - 1, 1), float(sigma[0]))
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None


def branch_selection(branches, batch: int, mode: str) -> tuple[tuple[int, ...], str | None]:
    if (not isinstance(branches, (list, tuple)) or not branches
            or any(type(value) is not int or value not in (0, 1) for value in branches)
            or batch <= 0 or batch % len(branches)):
        return (), "unsupported_branch_layout"
    chunk = batch // len(branches)
    wanted = {"positive_only": (0,), "negative_only": (1,), "both": (0, 1)}[mode]
    indices = tuple(row for index, branch in enumerate(branches) if branch in wanted
                    for row in range(index * chunk, (index + 1) * chunk))
    return indices, None if indices else "branch_not_selected"


def _require_state(options: dict) -> RunState:
    state = options.get(RUN_KEY)
    if not isinstance(state, RunState) or state.finished:
        raise RuntimeError("Krea2 AFM requires an active OUTER_SAMPLE run")
    return state


def make_outer_sample_wrapper(config: AFMConfig):
    def outer_sample(executor, *args, **kwargs):
        guider = executor.class_obj
        original = guider.model_options
        if RUN_KEY in original.get("transformer_options", {}):
            raise RuntimeError("Nested Krea2 AFM sampling is unsupported")
        if original.get("multigpu_thread_pool") is not None:
            raise AFMUnsupportedError("multi_gpu_not_supported")
        # Only this run's guider gets state. Native nested dict/list clones
        # intentionally carry the same RunState object through sigma/CFG calls.
        options = original.copy()
        options["transformer_options"] = original.get("transformer_options", {}).copy()
        state = RunState()
        options["transformer_options"][RUN_KEY] = state
        guider.model_options = options
        started = time.perf_counter()
        status, error_type = "error", None
        try:
            emit_record(config, state.run_id, "run_start", environment=environment_metadata(),
                        config=configuration_metadata(config))
            result = executor(*args, **kwargs)
            status = "ok"
            return result
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            # Restoration precedes logging, including if the log destination fails.
            guider.model_options = original
            options["transformer_options"].pop(RUN_KEY, None)
            state.finished = True
            emit_record(config, state.run_id, "run_summary", status=status, error_type=error_type,
                        model_evaluations=state.model_evaluations,
                        attention_calls=state.attention_calls, observed_calls=state.observed_calls,
                        edited_calls=state.edited_calls, reasons=dict(state.reasons),
                        wall_seconds=time.perf_counter() - started,
                        timing_scope="outer_sample_unsynchronized_diagnostic",
                        peak_allocated_bytes=None, peak_reserved_bytes=None,
                        memory_measurement="not_measured")
    return outer_sample


def make_diffusion_wrapper(config: AFMConfig):
    def diffusion(executor, x, timesteps, context, attention_mask=None, ref_latents=None,
                  transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        state = _require_state(options)
        progress = progress_from_sigmas(options)
        options[EVAL_KEY] = state.next_evaluation(progress)
        if progress is None:
            state.reasons["unsupported_progress"] += 1
        model = executor.class_obj
        # Backend function may be populated when weights load; resolve per forward.
        options[BACKENDS_KEY] = tuple(
            getattr(getattr(block.attn, "comfy_attention", None), "function", None)
            for block in model.blocks
        )
        try:
            if ref_latents is not None and len(ref_latents) > 0:
                raise AFMUnsupportedError("reference_not_supported")
            if options.get("reference_image_num_tokens"):
                raise AFMUnsupportedError("reference_not_supported")
            if "optimized_attention_override" in options:
                raise ValueError("Krea2 AFM conflict: runtime optimized_attention_override")
            if context.ndim != 3 or context.shape[0] != x.shape[0]:
                raise AFMUnsupportedError("unsupported_layout", "conditioning shape")
            options[LAYOUT_KEY] = layout_from_latent(x, model.patch, context.shape[1])
        except AFMUnsupportedError as exc:
            state.reasons[exc.reason] += 1
            if config.fail_mode == "raise":
                raise
            options[SKIP_KEY] = exc.reason
            state.reasons["fallback_original"] += 1
            emit_record(config, state.run_id, "evaluation_skipped", reason=exc.reason,
                        edit_applied=False, model_eval_index=options[EVAL_KEY])
        return executor(x, timesteps, context, attention_mask, ref_latents, options, **kwargs)
    return diffusion


def make_post_input_patch(config: AFMConfig):
    def post_input(payload):
        options = payload["transformer_options"]
        state = _require_state(options)
        if options.get(SKIP_KEY):
            return payload
        try:
            layout = validate_post_input(payload, options[LAYOUT_KEY])
        except AFMUnsupportedError as exc:
            state.reasons[exc.reason] += 1
            if config.fail_mode == "raise":
                raise
            state.reasons["fallback_original"] += 1
            options[SKIP_KEY] = exc.reason
            emit_record(config, state.run_id, "evaluation_skipped", reason=exc.reason,
                        edit_applied=False, model_eval_index=options.get(EVAL_KEY))
            return payload
        if options.get("optimized_attention_override") is not None:
            raise ValueError("Krea2 AFM conflict: runtime optimized_attention_override")
        from .backend import Krea2AFMAttentionOverride
        options[LAYOUT_KEY] = layout
        options["optimized_attention_override"] = Krea2AFMAttentionOverride(config)
        return payload
    return post_input


def install_krea2_afm(model_patcher, config: AFMConfig):
    caps = detect_krea2(model_patcher)
    validate_config(config, caps)
    if config.mode == "edit":
        raise NotImplementedError("This diagnostic release supports off, discover and observe only")
    if config.mode == "off":
        return model_patcher.clone()
    conflicts = detect_patch_conflicts(model_patcher)
    if conflicts:
        raise ValueError("Krea2 AFM conflict: " + ", ".join(conflicts))
    from comfy.patcher_extension import WrappersMP
    clone = model_patcher.clone()
    clone.model_options["transformer_options"][CONFIG_KEY] = config
    clone.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, make_outer_sample_wrapper(config))
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, make_diffusion_wrapper(config))
    clone.set_model_post_input_patch(make_post_input_patch(config))
    return clone
