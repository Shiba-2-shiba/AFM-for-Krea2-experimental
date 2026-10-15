import math

import torch


def _schedule_alphas(config, progress):
    """Test-only copy of the curve contract; deliberately independent of afm.spectral."""
    return (
        1.0 + config.strength * (1.0 - progress),
        1.0 + config.strength * progress,
    )


def _frequency_axis(size, *, device):
    indices = torch.arange(size, dtype=torch.float64, device=device)
    return torch.where(indices <= size // 2, indices, indices - size) / size


def _explicit_dft2(signal, grid):
    """Tiny-grid FP64 DFT written without torch.fft or production helpers."""
    height, width = grid
    spatial = signal.double().reshape(*signal.shape[:-2], height, width, signal.shape[-1])
    spatial = spatial.permute(0, 1, 4, 2, 3).to(torch.complex128)
    y = torch.arange(height, dtype=torch.float64, device=signal.device)
    x = torch.arange(width, dtype=torch.float64, device=signal.device)
    wy = torch.exp(-2j * torch.pi * y[:, None] * y[None, :] / height)
    wx = torch.exp(-2j * torch.pi * x[:, None] * x[None, :] / width)
    return torch.einsum("uy,bhcyx,vx->bhcuv", wy, spatial, wx)


def _explicit_idft2(spectrum, grid):
    height, width = grid
    y = torch.arange(height, dtype=torch.float64, device=spectrum.device)
    x = torch.arange(width, dtype=torch.float64, device=spectrum.device)
    wy = torch.exp(2j * torch.pi * y[:, None] * y[None, :] / height)
    wx = torch.exp(2j * torch.pi * x[:, None] * x[None, :] / width)
    spatial = torch.einsum("yu,bhcuv,xv->bhcyx", wy, spectrum, wx) / (height * width)
    return spatial.real.permute(0, 1, 3, 4, 2).reshape(
        spectrum.shape[0], spectrum.shape[1], height * width, spectrum.shape[2]
    )


def independent_edit_logits(logits, grid, config, progress):
    """Independent FP64 spectral oracle for small test grids."""
    height, width = grid
    fy = _frequency_axis(height, device=logits.device)[:, None]
    fx = _frequency_axis(width, device=logits.device)[None, :]
    radius = (fy.square() + fx.square()).sqrt() / math.sqrt(0.5)
    if config.soft_mask:
        transition = (
            (radius - (config.cutoff - config.mask_width / 2.0)) / config.mask_width
        ).clamp(0.0, 1.0)
        low = 0.5 * (1.0 + torch.cos(torch.pi * transition))
    else:
        low = (radius <= config.cutoff).double()
    alpha_lf, alpha_hf = _schedule_alphas(config, progress)
    gain = alpha_lf * low + alpha_hf * (1.0 - low)
    if config.preserve_dc:
        gain = gain.clone()
        gain[0, 0] = 1.0
    return _explicit_idft2(_explicit_dft2(logits, grid) * gain, grid)


def independent_spectral_ratios(concentration, grid, cutoff, radial_bins):
    """Independent FP64 raw/centered/binned ratios for a scalar spatial map."""
    height, width = grid
    signal = concentration.double().reshape(1, 1, height * width, 1)
    raw_power = _explicit_dft2(signal, grid).abs().square().reshape(height, width)
    centered_signal = signal - signal.mean()
    centered_power = _explicit_dft2(centered_signal, grid).abs().square().reshape(
        height, width
    )
    fy = _frequency_axis(height, device=signal.device)[:, None]
    fx = _frequency_axis(width, device=signal.device)[None, :]
    radius = (fy.square() + fx.square()).sqrt() / math.sqrt(0.5)
    raw_distribution = raw_power / raw_power.sum()
    raw = raw_distribution[radius > cutoff].sum()
    centered = None
    if centered_power.sum() > raw_power.sum() * 1e-12:
        centered = (centered_power / centered_power.sum())[radius > cutoff].sum()
    indices = (radius * radial_bins).long().clamp(max=radial_bins - 1)
    energy = torch.zeros(radial_bins, dtype=torch.float64, device=signal.device)
    energy.scatter_add_(0, indices.reshape(-1), raw_distribution.reshape(-1))
    centers = (torch.arange(radial_bins, device=signal.device) + 0.5) / radial_bins
    binned = energy[centers >= cutoff].sum()
    return float(raw), None if centered is None else float(centered), float(binned)


def dense_joint_probabilities(q, k, layout, config, progress, scale):
    """Return FP64 full-key probabilities for selected image-query rows."""
    logits = torch.matmul(q.double(), k.double().transpose(-2, -1)) * scale
    image = slice(*layout.image_slice)
    text = slice(*layout.text_slice)
    original_text = logits[..., image, text].clone()
    edited_text = independent_edit_logits(
        original_text, layout.grid, config, progress.progress
    )
    if getattr(config, "normalization_mode", "joint") == "preserve_text_mass":
        correction = (
            original_text.logsumexp(dim=-1, keepdim=True)
            - edited_text.logsumexp(dim=-1, keepdim=True)
        )
        edited_text = edited_text + correction
    logits[..., image, text] = edited_text
    return torch.softmax(logits[..., image, :], dim=-1)


def dense_joint_image_rows(q, k, v, layout, config, progress, scale):
    """Independent small CPU oracle; production kernels must not call this."""
    probabilities = dense_joint_probabilities(q, k, layout, config, progress, scale)
    return torch.matmul(probabilities, v.double()).to(v.dtype)
