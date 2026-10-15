"""Bounded, opt-in diagnostics with explicit spectral definitions."""
from dataclasses import asdict, replace
import math

import torch

from .attention_math import preserve_text_partition, require_finite
from .augmented_attention import augmented_image_rows
from .config import parse_block_scope, parse_diagnostic_heads, validate_normalization_options
from .joint_attention import joint_image_rows
from .metrics import compare_attention_outputs
from .spectral import edit_image_text_logits, schedule_alphas

METRIC_VERSION = "text_spectrum_v2"


def diagnostic_requested(config, options, progress) -> bool:
    if (config.spectral_diag == "off" and not config.compare_kernels) or progress is None:
        return False
    return (options["block_index"] in parse_block_scope(
                config.diagnostic_block_scope, options["total_blocks"])
            and progress.step_index % config.diagnostic_every_n_steps == 0)


def text_spectral_metrics(logits, layout, config, radial_bins=32) -> dict:
    """Compute each ratio for a single batch row and physical attention head."""
    if (logits.ndim != 4 or logits.shape[:2] != (1, 1)
            or logits.shape[2] != math.prod(layout.grid) or logits.shape[-1] <= 0):
        raise ValueError("spectral metrics require [1,1,grid_h*grid_w,T]")
    if type(radial_bins) is not int or not 1 <= radial_bins <= 256:
        raise ValueError("radial_bins must be within 1..256")
    require_finite(logits, "spectral logits")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = logits.float().softmax(dim=-1)
        top_k = min(config.diagnostic_top_k, logits.shape[-1])
        concentration = probabilities.topk(top_k, dim=-1).values.mean(-1)
        spatial = concentration.reshape(*layout.grid)
        raw_power = torch.fft.fft2(spatial).abs().square().to(torch.float64)
        centered = spatial - spatial.mean()
        centered_power = torch.fft.fft2(centered).abs().square().to(torch.float64)
        raw_total = float(raw_power.sum().item())
        centered_total = float(centered_power.sum().item())
        h, w = layout.grid
        fy = torch.fft.fftfreq(h, device=logits.device, dtype=torch.float32)[:, None]
        fx = torch.fft.fftfreq(w, device=logits.device, dtype=torch.float32)[None, :]
        radius = (fy.square() + fx.square()).sqrt() / math.sqrt(0.5)
        high = radius > config.cutoff
        raw_defined = raw_total > 0 and math.isfinite(raw_total)
        constant_topk = top_k == logits.shape[-1]
        centered_defined = (raw_defined and not constant_topk
                            and math.isfinite(centered_total)
                            and centered_total > raw_total * 1e-12)
        raw_ratio = binned_ratio = centered_ratio = None
        reasons = {}
        if raw_defined:
            normalized = raw_power / raw_total
            raw_ratio = float(normalized[high].sum().item())
            bin_index = (radius * radial_bins).long().clamp(max=radial_bins - 1)
            energy = torch.zeros(radial_bins, device=logits.device, dtype=torch.float64)
            energy.scatter_add_(0, bin_index.reshape(-1), normalized.reshape(-1))
            bin_centers = (torch.arange(radial_bins, device=logits.device) + 0.5) / radial_bins
            binned_ratio = float(energy[bin_centers >= config.cutoff].sum().item())
        else:
            reasons["raw"] = reasons["radial_binned"] = "zero_or_nonfinite_power"
        if centered_defined:
            centered_ratio = float((centered_power / centered_total)[high].sum().item())
        else:
            reasons["centered"] = "constant_topk" if constant_topk else "centered_noise_floor"
    return {
        "rho_text_raw": raw_ratio,
        "rho_text_centered": centered_ratio,
        "rho_text_radial_binned": binned_ratio,
        "metric_defined": {"raw": raw_defined, "centered": centered_defined,
                           "radial_binned": raw_defined},
        "constant_topk": constant_topk,
        "undefined_reasons": reasons,
    }


def _rho_text_conditional(logits, layout, config):
    """Legacy internal helper; schema 5 never emits this ambiguous field name."""
    return text_spectral_metrics(logits, layout, config)["rho_text_centered"]


def _spectral_record(q, k, layout, config, progress, query_chunk, scale, edit_applied):
    start, end = layout.image_slice
    query = q[..., start:end, :].float()
    keys = k.float()
    raw = query @ keys[..., :start, :].transpose(-2, -1) * scale
    filtered = raw
    if edit_applied:
        filtered = edit_image_text_logits(
            raw, layout.grid, *schedule_alphas(config, progress.progress), config)
        if config.normalization_mode == "preserve_text_mass":
            filtered = preserve_text_partition(filtered, raw.logsumexp(-1, keepdim=True))
    before_mass = after_mass = 0.0
    mass_delta = image_delta = 0.0
    count = 0
    for offset in range(0, end - start, query_chunk):
        stop = min(offset + query_chunk, end - start)
        scores = query[..., offset:stop, :] @ keys.transpose(-2, -1) * scale
        require_finite(scores, "diagnostic joint logits")
        before = scores.softmax(-1)
        before_text = before[..., :start].sum(-1)
        before_mass += float(before_text.sum().item())
        if edit_applied:
            scores[..., :start] = filtered[..., offset:stop, :]
            after = scores.softmax(-1)
            after_text = after[..., :start].sum(-1)
            after_mass += float(after_text.sum().item())
            mass_delta = max(mass_delta, float((after_text - before_text).abs().max().item()))
            image_delta = max(image_delta, float(
                (after[..., start:] - before[..., start:]).abs().max().item()))
            del after, after_text
        else:
            after_mass += float(before_text.sum().item())
        count += before_text.numel()
        del scores, before, before_text
    bins = config.diagnostic_radial_bins
    before_metric = text_spectral_metrics(raw, layout, config, bins)
    after_metric = text_spectral_metrics(filtered, layout, config, bins)
    values = {}
    for name in ("rho_text_raw", "rho_text_centered", "rho_text_radial_binned"):
        values[name + "_before"] = before_metric[name]
        values[name + "_after"] = after_metric[name]
    before_raw, after_raw = before_metric["rho_text_raw"], after_metric["rho_text_raw"]
    return {
        "event": "spectral_diagnostic", "edit_applied": edit_applied, "finite": True,
        "metric_version": METRIC_VERSION, "radial_bins": bins,
        "cutoff": config.cutoff, "grid": list(layout.grid),
        "radius_normalization": "nyquist_corner",
        "raw_cutoff_rule": "radius>cutoff", "binned_cutoff_rule": "bin_center>=cutoff",
        "metric_defined": {key: before_metric["metric_defined"][key]
                          and after_metric["metric_defined"][key]
                          for key in before_metric["metric_defined"]},
        "metric_defined_before": before_metric["metric_defined"],
        "metric_defined_after": after_metric["metric_defined"],
        "constant_topk": before_metric["constant_topk"],
        "undefined_reasons": {"before": before_metric["undefined_reasons"],
                             "after": after_metric["undefined_reasons"]},
        "direct_delta_rho": None if before_raw is None or after_raw is None else after_raw - before_raw,
        "text_attention_mass_before": before_mass / count,
        "text_attention_mass_after": after_mass / count,
        "text_mass_max_abs_delta": mass_delta,
        "image_probability_max_abs_delta": image_delta,
        "max_logits_delta": float((filtered - raw).abs().max().item()),
        "diagnostic_top_k": min(config.diagnostic_top_k, start),
        "diagnostic_top_k_requested": config.diagnostic_top_k,
        **values,
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
    validate_normalization_options(config)
    selected_heads = parse_diagnostic_heads(
        config.diagnostic_head_indices, q.shape[1], config.diagnostic_head_limit)
    query_chunk = min(config.diagnostic_query_chunk_size, chunks.query_chunk_size)
    # Spectra and comparisons are emitted per physical head. This keeps metrics
    # independent of processing chunks and limits the live diagnostic workspace.
    diagnostic_chunks = replace(chunks, head_chunk_size=1, query_chunk_size=query_chunk)
    records = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        for row in rows:
            for head in selected_heads:
                qs, ks, vs = (value[row:row + 1, head:head + 1] for value in (q, k, v))
                common = {"branch": int(row), "head_index": head,
                          "diagnostic_heads": [head],
                          "diagnostic_query_chunk_size": query_chunk,
                          "normalization_mode": config.normalization_mode,
                          "normalization_version": 1}
                if config.spectral_diag == "sampled":
                    records.append({**common, **_spectral_record(
                        qs, ks, layout, config, progress, query_chunk, scale, edit_applied)})
                if config.compare_kernels:
                    records.append({**common, **_comparison_record(
                        qs, ks, vs, layout, config, progress, diagnostic_chunks, scale)})
    return records
