import pytest
import torch

from afm.config import AFMConfig
from afm.joint_attention import joint_image_rows
from afm.memory import ChunkPlan
from afm.runtime import AFMUnsupportedError, ProgressInfo, TokenLayout
from reference import dense_joint_image_rows


def _fixture():
    torch.manual_seed(10)
    layout = TokenLayout((0, 5), (5, 20), (3, 5), 20)
    q, k, v = (torch.randn(2, 6, 20, 8) for _ in range(3))
    config = AFMConfig(strength=0.1, cutoff=0.3)
    progress = ProgressInfo(2, 5, 0.5, 1.0)
    chunks = ChunkPlan(4, 7, 1.0, "manual")
    return q, k, v, layout, config, progress, chunks


def test_chunked_matches_dense_joint_with_remainder_chunks():
    q, k, v, layout, config, progress, chunks = _fixture()
    scale = 0.2
    expected = dense_joint_image_rows(q, k, v, layout, config, progress, scale)
    actual = joint_image_rows(q, k, v, layout, config, progress, chunks, scale)
    assert actual.shape == (2, 6, 15, 8)
    assert actual.dtype == v.dtype
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_joint_normalization_is_not_text_only():
    q, k, v, layout, config, progress, chunks = _fixture()
    actual = joint_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    expected = dense_joint_image_rows(q, k, v, layout, config, progress, 0.2)
    text_logits = torch.matmul(q[..., 5:, :], k[..., :5, :].transpose(-2, -1)) * 0.2
    text_only = torch.matmul(torch.softmax(text_logits, -1), v[..., :5, :])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(actual, text_only, atol=1e-3, rtol=1e-3)


def test_expanded_gqa_heads_are_not_expanded_again():
    q, k, v, layout, config, progress, _ = _fixture()
    chunks = ChunkPlan(5, 6, 1.0, "manual")
    actual = joint_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    assert actual.shape[1] == q.shape[1]


def test_nonfinite_input_raises_without_returning_partial_output():
    q, k, v, layout, config, progress, chunks = _fixture()
    q[1, 5, -1, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        joint_image_rows(q, k, v, layout, config, progress, chunks, 0.2)


def test_nonfinite_later_chunk_aborts_without_retry(monkeypatch):
    q, k, v, layout, config, progress, chunks = _fixture()
    original = torch.softmax
    calls = 0

    def inject(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(value, *args, **kwargs)
        return torch.full_like(result, torch.nan) if calls == 2 else result

    monkeypatch.setattr(torch, "softmax", inject)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        joint_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    assert calls == 2


@pytest.mark.parametrize("kwargs, reason", [({"mask": torch.ones(1)}, "unsupported_mask"),
                                             ({"is_causal": True}, "unsupported_causal")])
def test_mask_and_causal_are_explicitly_rejected(kwargs, reason):
    q, k, v, layout, config, progress, chunks = _fixture()
    with pytest.raises(AFMUnsupportedError, match=reason):
        joint_image_rows(q, k, v, layout, config, progress, chunks, 0.2, **kwargs)


def test_manual_fp32_math_is_not_changed_by_ambient_cpu_autocast():
    q, k, v, layout, config, progress, chunks = _fixture()
    expected = dense_joint_image_rows(q, k, v, layout, config, progress, 0.2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = joint_image_rows(q, k, v, layout, config, progress, chunks, 0.2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda q, k, v: (q[:, :, :-1], k, v), "layout"),
        (lambda q, k, v: (q, k[:, :1], v), "shape|head"),
        (lambda q, k, v: (q, k, v.double()), "dtype"),
    ],
)
def test_manual_shared_validation_rejects_incompatible_qkv(mutation, reason):
    q, k, v, layout, config, progress, chunks = _fixture()
    q, k, v = mutation(q, k, v)
    with pytest.raises((AFMUnsupportedError, ValueError, TypeError), match=reason):
        joint_image_rows(q, k, v, layout, config, progress, chunks, 0.2)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.0, -0.1])
def test_manual_shared_validation_rejects_invalid_scale(scale):
    q, k, v, layout, config, progress, chunks = _fixture()
    with pytest.raises((AFMUnsupportedError, ValueError), match="scale"):
        joint_image_rows(q, k, v, layout, config, progress, chunks, scale)
