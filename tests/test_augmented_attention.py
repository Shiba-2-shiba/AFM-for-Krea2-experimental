from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from afm.augmented_attention import (
    augmented_image_rows,
    build_augmented_kv,
    strip_augmented_heads,
)
from afm.config import AFMConfig
from afm.memory import ChunkPlan
from afm.runtime import AFMUnsupportedError, ProgressInfo, TokenLayout
from afm.spectral import apply_spatial_filter, edit_image_text_logits
from reference import dense_joint_image_rows


def _fixture(alpha_strength=0.2):
    torch.manual_seed(20)
    layout = TokenLayout((0, 4), (4, 19), (3, 5), 19)
    q, k, v = (torch.randn(2, 3, 19, 8) for _ in range(3))
    config = AFMConfig(strength=alpha_strength, cutoff=0.3)
    progress = ProgressInfo(2, 4, 0.3, 1.0)
    chunks = ChunkPlan(2, 6, 1.0, "augmented")
    return q, k, v, layout, config, progress, chunks


def test_augmented_logits_match_direct_fft_and_preserve_image_keys():
    q, k, v, layout, config, _, _ = _fixture()
    alpha_lf, alpha_hf = 1.08, 1.14
    q_img = q[..., 4:, :]
    filtered = apply_spatial_filter(q_img, layout.grid, alpha_lf, alpha_hf, config)
    delta = filtered - q_img.float()
    q_aug = torch.cat((q_img, delta), dim=-1)
    augmented = build_augmented_kv(k, v, layout)
    actual = torch.matmul(q_aug, augmented.k.transpose(-2, -1))
    direct = torch.matmul(q_img, k.transpose(-2, -1))
    direct[..., :4] = edit_image_text_logits(direct[..., :4], layout.grid,
                                                  alpha_lf, alpha_hf, config)
    torch.testing.assert_close(actual, direct, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[..., 4:], torch.matmul(q_img, k[..., 4:, :].transpose(-2, -1)),
                               atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(augmented.v[..., 8:]) == 0


@pytest.mark.parametrize("scale", [0.2, 8 ** -0.5])
def test_augmented_output_matches_dense_and_manual_with_original_scale(scale):
    q, k, v, layout, config, progress, chunks = _fixture()
    expected = dense_joint_image_rows(q, k, v, layout, config, progress, scale)
    actual = augmented_image_rows(q, k, v, layout, config, progress, chunks, scale)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_strip_per_head_not_flat_prefix():
    head = torch.arange(2 * 3 * 5 * 16, dtype=torch.float32).reshape(2, 3, 5, 16)
    expected = head[..., :8]
    torch.testing.assert_close(strip_augmented_heads(head, 8, 3, False), expected)
    merged = head.permute(0, 2, 1, 3).reshape(2, 5, 48)
    expected_merged = expected.permute(0, 2, 1, 3).reshape(2, 5, 24)
    torch.testing.assert_close(strip_augmented_heads(merged, 8, 3, True), expected_merged)


def test_zero_strength_still_uses_augmented_numerical_control(monkeypatch):
    q, k, v, layout, config, progress, chunks = _fixture(alpha_strength=0.0)
    calls = []
    original = F.scaled_dot_product_attention

    def record(*args, **kwargs):
        calls.append((args[0].shape, kwargs["scale"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    actual = augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    assert calls
    assert all(shape[-1] == 16 for shape, _ in calls)
    assert actual.shape == (2, 3, 15, 8)


def test_fft_precedes_query_chunking_and_sdpa_calls_are_bounded(monkeypatch):
    q, k, v, layout, config, progress, _ = _fixture()
    calls = []
    original = F.scaled_dot_product_attention

    def record(query, key, value, **kwargs):
        calls.append((query.shape, key.shape))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    a = augmented_image_rows(q, k, v, layout, config, progress,
                             ChunkPlan(2, 4, 1.0, "augmented"), 0.2)
    b = augmented_image_rows(q, k, v, layout, config, progress,
                             ChunkPlan(2, 7, 1.0, "augmented"), 0.2)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert all(qs[1] <= 2 and qs[2] <= 7 and ks[2] == 19 for qs, ks in calls)


@pytest.mark.parametrize("kwargs, reason", [({"mask": torch.ones(1)}, "unsupported_mask"),
                                             ({"is_causal": True}, "unsupported_causal")])
def test_mask_and_causal_are_rejected(kwargs, reason):
    q, k, v, layout, config, progress, chunks = _fixture()
    with pytest.raises(AFMUnsupportedError, match=reason):
        augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2, **kwargs)


def test_nonfinite_never_returns_partial_output():
    q, k, v, layout, config, progress, chunks = _fixture()
    v[1, -1, -1, -1] = torch.inf
    with pytest.raises(FloatingPointError, match="nonfinite"):
        augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)


def test_nonfinite_later_sdpa_chunk_aborts_without_retry(monkeypatch):
    q, k, v, layout, config, progress, chunks = _fixture()
    original = F.scaled_dot_product_attention
    calls = 0

    def inject(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        return torch.full_like(result, torch.nan) if calls == 2 else result

    monkeypatch.setattr(F, "scaled_dot_product_attention", inject)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    assert calls == 2


def test_sdpa_without_explicit_scale_support_is_rejected(monkeypatch):
    q, k, v, layout, config, progress, chunks = _fixture()

    def legacy_sdpa(query, key, value, dropout_p=0.0, is_causal=False):
        return query

    monkeypatch.setattr(F, "scaled_dot_product_attention", legacy_sdpa)
    with pytest.raises(AFMUnsupportedError, match="unsupported_scale"):
        augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)


def test_float32_diagnostic_precision_casts_only_sdpa_head_groups(monkeypatch):
    q, k, v, layout, config, progress, chunks = _fixture()
    q, k, v = q.half(), k.half(), v.half()
    object.__setattr__(config, "augmented_precision", "float32")
    seen = []
    original = F.scaled_dot_product_attention

    def record(query, key, value, **kwargs):
        seen.append((query.dtype, key.dtype, value.dtype, query.shape[1]))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    actual = augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    assert actual.dtype == torch.float16
    assert seen and all(dtypes[:3] == (torch.float32,) * 3 for dtypes in seen)
    assert all(heads <= chunks.head_chunk_size for *_, heads in seen)


def test_float32_augmented_math_is_not_changed_by_ambient_cpu_autocast():
    q, k, v, layout, config, progress, chunks = _fixture()
    config = replace(config, augmented_precision="float32")
    expected = dense_joint_image_rows(q, k, v, layout, config, progress, 0.2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_preserve_text_mass_is_rejected_before_sdpa(monkeypatch):
    q, k, v, layout, config, progress, chunks = _fixture()
    config = replace(config, normalization_mode="preserve_text_mass")
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("SDPA must not run for unsupported normalization")

    monkeypatch.setattr(F, "scaled_dot_product_attention", forbidden)
    with pytest.raises(AFMUnsupportedError, match="unsupported_normalization_implementation"):
        augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    assert calls == 0


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda q, k, v: (q[:, :, :-1], k, v), "layout"),
        (lambda q, k, v: (q, k[:, :1], v), "shape|head"),
        (lambda q, k, v: (q, k, v.double()), "dtype"),
    ],
)
def test_augmented_shared_validation_rejects_incompatible_qkv(mutation, reason):
    q, k, v, layout, config, progress, chunks = _fixture()
    q, k, v = mutation(q, k, v)
    with pytest.raises((AFMUnsupportedError, ValueError, TypeError), match=reason):
        augmented_image_rows(q, k, v, layout, config, progress, chunks, 0.2)

@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_augmented_kv_rejects_mixed_or_nonfloating_dtype(dtype):
    _, k, v, layout, *_ = _fixture()
    with pytest.raises(AFMUnsupportedError, match="dtype|floating"):
        build_augmented_kv(k, v.to(dtype), layout)

@pytest.mark.parametrize("scale", [True, "0.2", torch.tensor(0.2), 0.0, -0.1, float("nan"), float("inf")])
def test_augmented_rejects_nonpositive_or_nonreal_scale(scale):
    q, k, v, layout, config, progress, chunks = _fixture()
    with pytest.raises(AFMUnsupportedError, match="scale"):
        augmented_image_rows(q, k, v, layout, config, progress, chunks, scale)
