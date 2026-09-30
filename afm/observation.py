"""Opt-in bounded diagnostics. Never used by the normal edit kernel path."""
from dataclasses import asdict, replace

import torch

from .augmented_attention import augmented_image_rows
from .config import parse_block_scope
from .joint_attention import joint_image_rows
from .metrics import compare_attention_outputs
from .spectral import radial_low_high_masks, edit_image_text_logits, schedule_alphas


def diagnostic_requested(config, options, progress) -> bool:
    if (config.spectral_diag == "off" and not config.compare_kernels) or progress is None:
        return False
    return (options["block_index"] in parse_block_scope(config.diagnostic_block_scope, options["total_blocks"])
            and progress.step_index % config.diagnostic_every_n_steps == 0)


def _rho_text_conditional(logits, layout, config) -> float:
    probabilities = logits.softmax(dim=-1)
    concentration = probabilities.topk(min(config.diagnostic_top_k, logits.shape[-1]), dim=-1).values.mean(-1)
    maps = concentration.reshape(*concentration.shape[:-1], *layout.grid)
    maps = maps - maps.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.fft2(maps).abs().square()
    _, high = radial_low_high_masks(layout.grid, config.cutoff, False, config.mask_width, logits.device)
    ratio = (power * high).sum(dim=(-2, -1)) / power.sum(dim=(-2, -1)).clamp_min(1e-12)
    return float(ratio.mean().item())


def _spectral_record(q, k, layout, config, progress, query_chunk, scale, edit_applied):
    start, end = layout.image_slice
    query = q[..., start:end, :].float()
    keys = k.float()
    raw = query @ keys[..., :start, :].transpose(-2, -1) * scale
    if edit_applied:
        filtered = edit_image_text_logits(raw, layout.grid, *schedule_alphas(config, progress.progress), config)
    else:
        filtered = raw
    before_mass = after_mass = 0.0
    count = 0
    for offset in range(0, end - start, query_chunk):
        stop = min(offset + query_chunk, end - start)
        scores = query[..., offset:stop, :] @ keys.transpose(-2, -1) * scale
        before_mass += float(scores.softmax(-1)[..., :start].sum().item())
        if edit_applied:
            scores[..., :start] = filtered[..., offset:stop, :]
            after_mass += float(scores.softmax(-1)[..., :start].sum().item())
        count += scores.shape[0] * scores.shape[1] * scores.shape[2]
    if not edit_applied:
        after_mass = before_mass
    return {
        "event": "spectral_diagnostic", "edit_applied": edit_applied,
        "rho_text_conditional_before": _rho_text_conditional(raw, layout, config),
        "rho_text_conditional_after": _rho_text_conditional(filtered, layout, config),
        "text_attention_mass_before": before_mass / count,
        "text_attention_mass_after": after_mass / count,
        "max_logits_delta": float((filtered - raw).abs().max().item()),
        "diagnostic_top_k": min(config.diagnostic_top_k, start),
    }


def _comparison_record(q, k, v, layout, config, progress, chunks, scale):
    fp32 = tuple(value.float() for value in (q, k, v))
    zero = replace(config, strength=0.0)
    manual_chunks = replace(chunks, implementation="manual")
    augmented_chunks = replace(chunks, implementation="augmented")
    reference = joint_image_rows(*fp32, layout, config, progress, manual_chunks, scale)
    reference_zero = joint_image_rows(*fp32, layout, zero, progress, manual_chunks, scale)
    candidate = augmented_image_rows(q, k, v, layout, config, progress, augmented_chunks, scale)
    candidate_zero = augmented_image_rows(q, k, v, layout, zero, progress, augmented_chunks, scale)
    metrics = compare_attention_outputs(candidate, reference, candidate_zero, reference_zero)
    if not metrics.finite:
        raise FloatingPointError("nonfinite same-QKV comparison")
    return {"event": "kernel_comparison", "same_qkv": True,
            "candidate_implementation": "augmented", "reference_implementation": "manual",
            "dtype": str(q.dtype).removeprefix("torch."), "strength": config.strength,
            "augmented_precision": config.augmented_precision,
            "reference_precision": "float32", "metrics": asdict(metrics)}


def collect_attention_diagnostics(q, k, v, layout, config, progress, chunks, scale,
                                  rows, *, edit_applied: bool) -> list[dict]:
    heads = min(q.shape[1], config.diagnostic_head_limit, chunks.head_chunk_size)
    query_chunk = min(config.diagnostic_query_chunk_size, chunks.query_chunk_size)
    diagnostic_chunks = replace(chunks, head_chunk_size=heads, query_chunk_size=query_chunk)
    records = []
    # All operations are local; ambient model autocast must not change reference math.
    with torch.autocast(device_type=q.device.type, enabled=False):
        for row in rows:
            qs, ks, vs = (value[row:row + 1, :heads] for value in (q, k, v))
            common = {"branch": int(row), "diagnostic_heads": list(range(heads)),
                      "diagnostic_query_chunk_size": query_chunk}
            if config.spectral_diag == "sampled":
                records.append({**common, **_spectral_record(qs, ks, layout, config, progress,
                                                            query_chunk, scale, edit_applied)})
            if config.compare_kernels:
                records.append({**common, **_comparison_record(qs, ks, vs, layout, config, progress,
                                                              diagnostic_chunks, scale)})
    return records
