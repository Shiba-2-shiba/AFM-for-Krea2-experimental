from dataclasses import replace

import pytest
import torch

from afm.config import AFMConfig
from afm.memory import choose_chunks
from afm.runtime import layout_from_latent, ProgressInfo
from afm.runtime import AFMUnsupportedError
from afm.observation import (
    collect_attention_diagnostics,
    diagnostic_requested,
    text_spectral_metrics,
)
from reference import independent_spectral_ratios


def fixture():
    torch.manual_seed(16)
    qkv = [torch.randn(2, 3, 18, 8) for _ in range(3)]
    layout = layout_from_latent(torch.empty(2, 16, 5, 9), 2, 3)
    config = AFMConfig(mode="edit", spectral_diag="sampled", diagnostic_head_limit=1,
                       diagnostic_query_chunk_size=4, head_chunk_size=2, query_chunk_size=5)
    # This tiny grid has no non-DC LF bins below cutoff=.25; use the HF endpoint.
    progress = ProgressInfo(1, 2, 1., 1.)
    chunks = choose_chunks(qkv[0].shape, qkv[1].shape, layout, config, "manual", qkv[0].dtype)
    return qkv, layout, config, progress, chunks


def test_diagnostics_are_bounded_and_text_mass_is_joint():
    qkv, layout, config, progress, chunks = fixture()
    records = collect_attention_diagnostics(*qkv, layout, config, progress, chunks, .2, [0],
                                            edit_applied=False)
    assert len(records) == 1
    record = records[0]
    assert record["diagnostic_heads"] == [0]
    assert record["diagnostic_query_chunk_size"] == 4
    logits = qkv[0][0, :1, 3:] @ qkv[1][0, :1].transpose(-2, -1) * .2
    expected = logits.softmax(-1)[..., :3].sum(-1).mean().item()
    assert record["text_attention_mass_before"] == pytest.approx(expected, abs=1e-6)
    assert record["text_attention_mass_after"] == record["text_attention_mass_before"]
    assert record["head_index"] == 0
    assert 0 <= record["rho_text_raw_before"] <= 1
    assert 0 <= record["rho_text_radial_binned_before"] <= 1
    assert "rho_text_conditional_before" not in record
    assert set(record["metric_defined"]) == {"raw", "centered", "radial_binned"}


def test_comparison_uses_same_qkv_and_resolves_nonzero_effect():
    qkv, layout, config, progress, chunks = fixture()
    config = replace(config, compare_kernels=True, strength=.2)
    records = collect_attention_diagnostics(*qkv, layout, config, progress, chunks, .2, [0],
                                            edit_applied=True)
    comparisons = [record for record in records if record["event"] == "kernel_comparison"]
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["same_qkv"] is True
    assert comparison["branch"] == 0
    assert comparison["metrics"]["finite"]
    assert comparison["metrics"]["effect_resolved"]
    assert comparison["metrics"]["rel_l2"] < 1e-5
    assert comparison["metrics"]["effect_rel_l2"] < .001


def test_observation_scope_is_independent_and_interval_is_honored():
    config = AFMConfig(mode="edit", block_scope="0", diagnostic_block_scope="2",
                       spectral_diag="sampled", diagnostic_every_n_steps=2)
    assert diagnostic_requested(config, {"block_index": 2, "total_blocks": 4}, ProgressInfo(0, 3, 0, 1))
    assert not diagnostic_requested(config, {"block_index": 2, "total_blocks": 4}, ProgressInfo(1, 3, .5, .5))
    assert not diagnostic_requested(config, {"block_index": 0, "total_blocks": 4}, ProgressInfo(0, 3, 0, 1))
    assert not diagnostic_requested(replace(config, spectral_diag="off"),
                                    {"block_index": 2, "total_blocks": 4}, ProgressInfo(0, 3, 0, 1))


def _logits_for_concentration(concentration):
    concentration = torch.as_tensor(concentration, dtype=torch.float64)
    log_odds = torch.log(concentration / (1.0 - concentration))
    return torch.stack((log_odds, torch.zeros_like(log_odds)), dim=-1).reshape(
        1, 1, concentration.numel(), 2
    )


@pytest.mark.parametrize("radial_bins", [16, 32, 64])
def test_spectral_metrics_match_independent_fp64_dft(radial_bins):
    grid = (4, 4)
    concentration = torch.tensor(
        [[0.55, 0.70, 0.55, 0.70],
         [0.80, 0.60, 0.80, 0.60],
         [0.55, 0.70, 0.55, 0.70],
         [0.80, 0.60, 0.80, 0.60]],
        dtype=torch.float64,
    )
    layout = layout_from_latent(torch.empty(1, 16, 8, 8), 2, 2)
    config = AFMConfig(diagnostic_top_k=1, cutoff=0.25)
    actual = text_spectral_metrics(
        _logits_for_concentration(concentration), layout, config, radial_bins=radial_bins
    )
    raw, centered, binned = independent_spectral_ratios(
        concentration, grid, config.cutoff, radial_bins
    )
    assert actual["rho_text_raw"] == pytest.approx(raw, abs=1e-10)
    assert actual["rho_text_centered"] == pytest.approx(centered, abs=1e-10)
    assert actual["rho_text_radial_binned"] == pytest.approx(binned, abs=1e-10)
    assert actual["metric_defined"] == {"raw": True, "centered": True, "radial_binned": True}


def test_checkerboard_regression_separates_raw_dc_from_centered_power():
    grid = (8, 8)
    checkerboard = (torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2
    logits = torch.zeros(1, 1, 64, 8)
    logits[..., 0] = (1.0 + 0.01 * checkerboard).reshape(-1)
    layout = layout_from_latent(torch.empty(1, 16, 16, 16), 2, 4)
    metrics = text_spectral_metrics(
        logits, layout, AFMConfig(diagnostic_top_k=4, cutoff=0.25), radial_bins=32
    )
    assert metrics["rho_text_raw"] == pytest.approx(3.83e-6, rel=0.03)
    assert metrics["rho_text_centered"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["metric_defined"]["centered"] is True


def test_constant_and_k_at_least_text_keep_raw_but_mark_centered_undefined():
    layout = layout_from_latent(torch.empty(1, 16, 8, 8), 2, 2)
    logits = torch.zeros(1, 1, 16, 3)
    metrics = text_spectral_metrics(
        logits, layout, AFMConfig(diagnostic_top_k=4), radial_bins=32
    )
    assert metrics["rho_text_raw"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["rho_text_radial_binned"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["rho_text_centered"] is None
    assert metrics["constant_topk"] is True
    assert metrics["metric_defined"] == {"raw": True, "centered": False, "radial_binned": True}
    assert metrics["undefined_reasons"]["centered"] == "constant_topk"


@pytest.mark.parametrize(
    "concentration,expected_centered",
    [
        ([[0.55, 0.55, 0.75, 0.75]] * 4, 0.0),
        ([[0.55, 0.75, 0.55, 0.75], [0.75, 0.55, 0.75, 0.55]] * 2, 1.0),
    ],
)
def test_single_low_and_high_frequency_maps_have_expected_centered_ratio(
    concentration, expected_centered
):
    layout = layout_from_latent(torch.empty(1, 16, 8, 8), 2, 2)
    metrics = text_spectral_metrics(
        _logits_for_concentration(concentration),
        layout,
        AFMConfig(diagnostic_top_k=1, cutoff=0.4),
        radial_bins=32,
    )
    assert metrics["rho_text_centered"] == pytest.approx(expected_centered, abs=1e-10)


def test_explicit_physical_heads_are_not_clipped_by_kernel_chunk_size():
    qkv, layout, config, progress, _ = fixture()
    config = replace(
        config,
        diagnostic_head_indices="2,0",
        diagnostic_head_limit=1,
        head_chunk_size=1,
    )
    chunks = choose_chunks(
        qkv[0].shape, qkv[1].shape, layout, config, "manual", qkv[0].dtype
    )
    snapshots = [value.clone() for value in qkv]
    records = collect_attention_diagnostics(
        *qkv, layout, config, progress, chunks, .2, [0], edit_applied=False
    )
    assert [record["head_index"] for record in records] == [2, 0]
    assert [record["diagnostic_heads"] for record in records] == [[2], [0]]
    assert chunks.head_chunk_size == 1
    for value, snapshot in zip(qkv, snapshots):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


def test_preserve_diagnostic_mass_uses_corrected_after_logits_and_max_error():
    qkv, layout, config, progress, chunks = fixture()
    config = replace(config, normalization_mode="preserve_text_mass", strength=0.2)
    records = collect_attention_diagnostics(
        *qkv, layout, config, progress, chunks, .2, [0], edit_applied=True
    )
    record = records[0]
    assert record["normalization_mode"] == "preserve_text_mass"
    assert record["text_attention_mass_after"] == pytest.approx(
        record["text_attention_mass_before"], abs=1e-6
    )
    assert record["text_mass_max_abs_delta"] <= 1e-5
    assert record["image_probability_max_abs_delta"] <= 1e-5
    assert record["direct_delta_rho"] == pytest.approx(
        record["rho_text_raw_after"] - record["rho_text_raw_before"], abs=1e-12
    )


@pytest.mark.parametrize("shape", [(2, 1, 16, 3), (1, 2, 16, 3), (1, 1, 15, 3)])
def test_spectral_metrics_requires_one_branch_and_one_head(shape):
    layout = layout_from_latent(torch.empty(1, 16, 8, 8), 2, 2)
    with pytest.raises(ValueError, match=r"\[1,1"):
        text_spectral_metrics(torch.zeros(shape), layout, AFMConfig())


def test_preserve_zero_augmented_rejected_before_kernel(monkeypatch):
    qkv, layout, config, progress, chunks = fixture()
    config = replace(
        config,
        strength=0.0,
        normalization_mode="preserve_text_mass",
        zero_strength_mode="augmented",
    )
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("kernel must not run")

    monkeypatch.setattr("afm.observation.joint_image_rows", forbidden)
    monkeypatch.setattr("afm.observation.augmented_image_rows", forbidden)
    with pytest.raises(AFMUnsupportedError, match="unsupported_normalization_implementation"):
        collect_attention_diagnostics(
            *qkv, layout, config, progress, chunks, .2, [0], edit_applied=True
        )
    assert calls == 0


def test_preserve_compare_kernels_rejected_before_kernel(monkeypatch):
    qkv, layout, config, progress, chunks = fixture()
    config = replace(
        config,
        normalization_mode="preserve_text_mass",
        compare_kernels=True,
    )
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("kernel must not run")

    monkeypatch.setattr("afm.observation.joint_image_rows", forbidden)
    monkeypatch.setattr("afm.observation.augmented_image_rows", forbidden)
    with pytest.raises(AFMUnsupportedError, match="unsupported_normalization_comparison"):
        collect_attention_diagnostics(
            *qkv, layout, config, progress, chunks, .2, [0], edit_applied=True
        )
    assert calls == 0
