from dataclasses import replace

import pytest
import torch

from afm.attention_math import preserve_text_partition
from afm.config import AFMConfig
from afm.joint_attention import joint_image_rows
from afm.memory import ChunkPlan
from afm.runtime import ProgressInfo, TokenLayout
from reference import (
    dense_joint_image_rows,
    dense_joint_probabilities,
    independent_edit_logits,
)


def _fixture(*, text_tokens=4, grid=(3, 5), dtype=torch.float32):
    torch.manual_seed(79)
    image_tokens = grid[0] * grid[1]
    total = text_tokens + image_tokens
    layout = TokenLayout((0, text_tokens), (text_tokens, total), grid, total)
    q, k, v = (torch.randn(1, 2, total, 8, dtype=dtype) for _ in range(3))
    config = AFMConfig(
        mode="edit",
        implementation="manual",
        normalization_mode="preserve_text_mass",
        strength=0.2,
        cutoff=0.25,
        soft_mask=True,
        mask_width=0.05,
        preserve_dc=True,
    )
    progress = ProgressInfo(3, 5, 0.75, 0.25)
    return q, k, v, layout, config, progress


def _baseline_probabilities(q, k, layout, scale):
    image = slice(*layout.image_slice)
    scores = q.double() @ k.double().transpose(-2, -1) * scale
    return scores[..., image, :].softmax(-1)


def test_partition_correction_is_fp32_nonmutating_and_preserves_logsumexp():
    torch.manual_seed(1)
    edited = torch.randn(2, 3, 7, 5, dtype=torch.float32)
    original = torch.randn(2, 3, 7, 5, dtype=torch.float32)
    partition = original.logsumexp(-1, keepdim=True)
    snapshot = edited.clone()

    corrected = preserve_text_partition(edited, partition)

    assert corrected.dtype == torch.float32
    assert corrected.shape == edited.shape
    assert corrected.data_ptr() != edited.data_ptr()
    torch.testing.assert_close(edited, snapshot, atol=0, rtol=0)
    torch.testing.assert_close(
        corrected.logsumexp(-1, keepdim=True), partition, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("partition_shape", [(1, 1, 3), (1, 1, 3, 2)])
def test_partition_correction_rejects_wrong_partition_shape(partition_shape):
    edited = torch.zeros(1, 1, 3, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="partition|shape"):
        preserve_text_partition(edited, torch.zeros(partition_shape, dtype=torch.float32))


@pytest.mark.parametrize("bad", [torch.float16, torch.float64])
def test_partition_correction_requires_fp32_inputs(bad):
    edited = torch.zeros(1, 1, 3, 4, dtype=bad)
    partition = torch.zeros(1, 1, 3, 1, dtype=bad)
    with pytest.raises((TypeError, ValueError), match="float32"):
        preserve_text_partition(edited, partition)


def test_fp64_oracle_preserves_mass_image_probabilities_and_text_conditionals():
    q, k, _, layout, config, progress = _fixture()
    scale = 8 ** -0.5
    baseline = _baseline_probabilities(q, k, layout, scale)
    edited = dense_joint_probabilities(q, k, layout, config, progress, scale)
    text = slice(*layout.text_slice)
    image = slice(*layout.image_slice)
    baseline_mass = baseline[..., text].sum(-1, keepdim=True)
    edited_mass = edited[..., text].sum(-1, keepdim=True)

    torch.testing.assert_close(edited_mass, baseline_mass, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(edited[..., image], baseline[..., image], atol=1e-10, rtol=1e-10)

    original_text_logits = (
        q.double()[..., image, :] @ k.double()[..., text, :].transpose(-2, -1) * scale
    )
    filtered = independent_edit_logits(
        original_text_logits, layout.grid, config, progress.progress
    )
    torch.testing.assert_close(
        edited[..., text] / edited_mass,
        filtered.softmax(-1),
        atol=1e-10,
        rtol=1e-10,
    )


def test_manual_output_matches_independent_fp64_oracle_for_remainder_chunks():
    q, k, v, layout, config, progress = _fixture()
    scale = 0.2
    expected = dense_joint_image_rows(q, k, v, layout, config, progress, scale)
    small = joint_image_rows(
        q, k, v, layout, config, progress, ChunkPlan(1, 4, 1.0, "manual"), scale
    )
    large = joint_image_rows(
        q, k, v, layout, config, progress, ChunkPlan(2, 15, 1.0, "manual"), scale
    )
    torch.testing.assert_close(small, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(large, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(small, large, atol=1e-5, rtol=1e-5)


def test_preserved_output_satisfies_baseline_plus_text_conditional_delta_identity():
    q, k, v, layout, config, progress = _fixture()
    scale = 0.2
    baseline_prob = _baseline_probabilities(q, k, layout, scale)
    edited_prob = dense_joint_probabilities(q, k, layout, config, progress, scale)
    text = slice(*layout.text_slice)
    text_mass = baseline_prob[..., text].sum(-1, keepdim=True)
    baseline_output = baseline_prob @ v.double()
    original_conditional = baseline_prob[..., text] / text_mass
    edited_conditional = edited_prob[..., text] / text_mass
    decomposed = baseline_output + text_mass * (
        edited_conditional @ v.double()[..., text, :]
        - original_conditional @ v.double()[..., text, :]
    )
    direct = edited_prob @ v.double()
    torch.testing.assert_close(direct, decomposed, atol=1e-10, rtol=1e-10)

    actual = joint_image_rows(
        q, k, v, layout, config, progress, ChunkPlan(1, 7, 1.0, "manual"), scale
    )
    torch.testing.assert_close(actual, decomposed.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("strength", [-1.0, -0.2, 0.0, 0.05, 0.1, 0.2, 2.0])
@pytest.mark.parametrize("progress_value", [0.0, 0.5, 1.0])
def test_preservation_holds_across_curve_strength_and_progress(strength, progress_value):
    q, k, _, layout, config, progress = _fixture()
    config = replace(config, strength=strength)
    progress = replace(progress, progress=progress_value)
    baseline = _baseline_probabilities(q, k, layout, 0.2)
    edited = dense_joint_probabilities(q, k, layout, config, progress, 0.2)
    text = slice(*layout.text_slice)
    image = slice(*layout.image_slice)
    torch.testing.assert_close(
        edited[..., text].sum(-1), baseline[..., text].sum(-1), atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(edited[..., image], baseline[..., image], atol=1e-10, rtol=1e-10)


def test_negative_dc_gain_is_corrected_only_after_the_fft_stage():
    _, _, _, layout, config, _ = _fixture(text_tokens=2, grid=(2, 2))
    config = replace(config, strength=-1.0, preserve_dc=False, soft_mask=False)
    raw = torch.tensor(
        [[[[3.0, -1.0], [1.0, 2.0], [4.0, 0.0], [2.0, 1.0]]]], dtype=torch.float64
    )
    filtered = independent_edit_logits(raw, layout.grid, config, progress=0.0)
    assert filtered.mean(dim=-2).abs().max() < 1e-12
    corrected = filtered + raw.logsumexp(-1, keepdim=True) - filtered.logsumexp(-1, keepdim=True)
    torch.testing.assert_close(
        corrected.logsumexp(-1), raw.logsumexp(-1), atol=1e-10, rtol=1e-10
    )
    assert not torch.allclose(corrected.mean(dim=-2), raw.mean(dim=-2), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("text_tokens", [1, 2])
@pytest.mark.parametrize("offset", [-80.0, 80.0])
def test_extreme_text_partitions_and_single_token_remain_finite(text_tokens, offset):
    edited = torch.linspace(-2, 2, 12 * text_tokens, dtype=torch.float32).reshape(
        1, 2, 6, text_tokens
    )
    original = torch.full_like(edited, offset)
    partition = original.logsumexp(-1, keepdim=True)
    corrected = preserve_text_partition(edited, partition)
    assert torch.isfinite(corrected).all()
    torch.testing.assert_close(
        corrected.logsumexp(-1, keepdim=True), partition, atol=1e-5, rtol=1e-5
    )
    if text_tokens == 1:
        torch.testing.assert_close(corrected, original, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 5e-3, 1e-2),
        (torch.bfloat16, 2e-2, 5e-2),
    ],
)
def test_low_precision_outputs_cast_only_after_fp32_preservation(dtype, atol, rtol):
    q, k, v, layout, config, progress = _fixture(dtype=dtype)
    expected = dense_joint_image_rows(q.float(), k.float(), v.float(), layout, config, progress, 0.2)
    actual = joint_image_rows(
        q, k, v, layout, config, progress, ChunkPlan(1, 7, 1.0, "manual"), 0.2
    )
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
