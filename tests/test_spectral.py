from dataclasses import replace

import pytest
import torch

from afm.config import AFMConfig
from afm.spectral import (
    apply_spatial_filter,
    edit_image_text_logits,
    radial_low_high_masks,
    schedule_alphas,
)


def test_curve_schedule_endpoints():
    config = AFMConfig(strength=0.1)
    assert schedule_alphas(config, 0.0) == pytest.approx((1.1, 1.0))
    assert schedule_alphas(config, 1.0) == pytest.approx((1.0, 1.1))
    assert schedule_alphas(config, 0.5) == pytest.approx((1.05, 1.05))


def test_fft_identity_rectangular():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 15, 4)
    actual = apply_spatial_filter(x, (3, 5), 1.0, 1.0, AFMConfig())
    assert actual.dtype == torch.float32
    assert torch.max(torch.abs(actual - x)) < 1e-5


def test_preserve_dc_keeps_spatial_mean():
    torch.manual_seed(2)
    x = torch.randn(2, 2, 15, 3)
    config = AFMConfig(preserve_dc=True, soft_mask=False, cutoff=0.3)
    actual = apply_spatial_filter(x, (3, 5), 0.25, 2.0, config)
    torch.testing.assert_close(actual.mean(dim=2), x.mean(dim=2), atol=1e-5, rtol=1e-5)


def test_frequency_selectivity_on_rectangular_non_power_of_two_grid():
    grid = (5, 7)
    rows = torch.arange(grid[0], dtype=torch.float32)[:, None]
    cols = torch.arange(grid[1], dtype=torch.float32)[None, :]
    low_signal = torch.cos(2 * torch.pi * cols / grid[1]).expand(grid)
    high_signal = torch.cos(2 * torch.pi * 3 * cols / grid[1]).expand(grid)
    config = AFMConfig(soft_mask=False, cutoff=0.4, preserve_dc=False)
    low = apply_spatial_filter(low_signal.reshape(1, 1, -1, 1), grid, 2.0, 1.0, config)
    high = apply_spatial_filter(high_signal.reshape(1, 1, -1, 1), grid, 1.0, 3.0, config)
    torch.testing.assert_close(low, 2.0 * low_signal.reshape(1, 1, -1, 1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(high, 3.0 * high_signal.reshape(1, 1, -1, 1), atol=1e-5, rtol=1e-5)


def test_common_operator_is_linear_and_logits_wrapper_uses_it():
    torch.manual_seed(3)
    x, y = torch.randn(1, 2, 15, 4), torch.randn(1, 2, 15, 4)
    config = replace(AFMConfig(), cutoff=0.35, mask_width=0.08)
    combined = apply_spatial_filter(1.7 * x - 0.4 * y, (3, 5), 1.08, 1.14, config)
    separate = (1.7 * apply_spatial_filter(x, (3, 5), 1.08, 1.14, config)
                - 0.4 * apply_spatial_filter(y, (3, 5), 1.08, 1.14, config))
    torch.testing.assert_close(combined, separate, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        edit_image_text_logits(x, (3, 5), 1.08, 1.14, config),
        apply_spatial_filter(x, (3, 5), 1.08, 1.14, config),
    )


def test_masks_are_complementary_normalized_radial_weights():
    low, high = radial_low_high_masks((3, 5), 0.25, True, 0.05, torch.device("cpu"))
    assert low.shape == high.shape == (3, 5)
    torch.testing.assert_close(low + high, torch.ones_like(low))
    assert low[0, 0] > low[1, 2]


def test_soft_mask_uses_finite_cosine_transition_edges():
    radius = 0.25 / (0.5 ** 0.5)
    low_edge, _ = radial_low_high_masks((4, 4), radius + 0.05, True, 0.1,
                                        torch.device("cpu"))
    high_edge, _ = radial_low_high_masks((4, 4), radius - 0.05, True, 0.1,
                                         torch.device("cpu"))
    assert low_edge[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert high_edge[0, 1].item() == pytest.approx(0.0, abs=1e-6)
