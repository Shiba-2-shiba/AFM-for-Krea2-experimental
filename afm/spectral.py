"""Shared FP32 spatial-frequency operator for AFM logits and queries."""
import math

import torch

from .config import AFMConfig


def schedule_alphas(config: AFMConfig, progress: float) -> tuple[float, float]:
    """Return gains without changing the common DC or normalization semantics.

    The opt-in frequency-neutral control removes LF/HF scheduling by applying
    1 + strength / 2 to both bands. It does not match total edit magnitude or
    implement a global attention-temperature change.
    """
    if config.schedule not in ("curve", "frequency_neutral"):
        raise ValueError(f"unsupported AFM schedule: {config.schedule!r}")
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be finite and in [0, 1]")
    if config.schedule == "frequency_neutral":
        gain = 1.0 + config.strength / 2.0
        return gain, gain
    return (1.0 + config.strength * (1.0 - progress),
            1.0 + config.strength * progress)


def radial_low_high_masks(
    grid: tuple[int, int],
    cutoff: float,
    soft_mask: bool,
    mask_width: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = _validate_grid(grid)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must be finite and in (0, 1)")
    if not math.isfinite(mask_width) or not 0.0 < mask_width <= 1.0:
        raise ValueError("mask_width must be finite and in (0, 1]")
    fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32)[:, None]
    fx = torch.fft.fftfreq(w, device=device, dtype=torch.float32)[None, :]
    # Normalize the radius by the two-dimensional Nyquist-corner radius.
    radius = torch.sqrt(fy.square() + fx.square()) / math.sqrt(0.5)
    if soft_mask:
        transition = ((radius - (cutoff - mask_width / 2.0)) / mask_width).clamp(0.0, 1.0)
        low = 0.5 * (1.0 + torch.cos(torch.pi * transition))
    else:
        low = (radius <= cutoff).to(torch.float32)
    return low, 1.0 - low


def apply_spatial_filter(
    x: torch.Tensor,
    grid: tuple[int, int],
    alpha_lf: float,
    alpha_hf: float,
    config: AFMConfig,
) -> torch.Tensor:
    h, w = _validate_grid(grid)
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[2] != h * w:
        raise ValueError("x must have shape [B, H, grid_h * grid_w, C]")
    if not x.is_floating_point():
        raise TypeError("x must be floating point")
    if not math.isfinite(alpha_lf) or not math.isfinite(alpha_hf):
        raise FloatingPointError("nonfinite AFM gain")
    signal = x.float()
    if not torch.isfinite(signal).all():
        raise FloatingPointError("nonfinite spatial-filter input")
    low, high = radial_low_high_masks(
        grid, config.cutoff, config.soft_mask, config.mask_width, signal.device
    )
    gain = alpha_lf * low + alpha_hf * high
    if config.preserve_dc:
        gain = gain.clone()
        gain[0, 0] = 1.0
    spatial = signal.reshape(signal.shape[0], signal.shape[1], h, w, signal.shape[3])
    spatial = spatial.permute(0, 1, 4, 2, 3)
    spectrum = torch.fft.fft2(spatial, dim=(-2, -1))
    filtered = torch.fft.ifft2(spectrum * gain, dim=(-2, -1)).real
    result = filtered.permute(0, 1, 3, 4, 2).reshape_as(signal)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite spatial-filter output")
    return result


def edit_image_text_logits(
    logits: torch.Tensor,
    grid: tuple[int, int],
    alpha_lf: float,
    alpha_hf: float,
    config: AFMConfig,
) -> torch.Tensor:
    return apply_spatial_filter(logits, grid, alpha_lf, alpha_hf, config)


def _validate_grid(grid: tuple[int, int]) -> tuple[int, int]:
    if (not isinstance(grid, tuple) or len(grid) != 2
            or any(type(value) is not int or value <= 0 for value in grid)):
        raise ValueError("grid must contain two positive integers")
    return grid
