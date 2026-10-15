"""Independent closed-form oracle for the opt-in uniform non-DC gain control."""

import pytest
import torch

from afm.augmented_attention import augmented_image_rows
from afm.config import AFMConfig
from afm.joint_attention import joint_image_rows
from afm.memory import ChunkPlan
from afm.runtime import ProgressInfo, TokenLayout


@pytest.mark.parametrize("preserve_dc", [True, False])
@pytest.mark.parametrize("strength", [0.0, 0.2])
@pytest.mark.parametrize("implementation,normalization_mode", [
    ("manual", "joint"),
    ("manual", "preserve_text_mass"),
    ("augmented", "joint"),
])
def test_frequency_neutral_attention_matches_closed_form_with_same_normalization(
    preserve_dc, strength, implementation, normalization_mode
):
    torch.manual_seed(183)
    text_tokens, total_tokens = 4, 19
    layout = TokenLayout((0, text_tokens), (text_tokens, total_tokens), (3, 5), total_tokens)
    q, k = (torch.randn(1, 2, total_tokens, total_tokens) for _ in range(2))
    # Identity values expose actual joint probabilities through the production kernel.
    v = torch.eye(total_tokens).reshape(1, 1, total_tokens, total_tokens).expand(1, 2, -1, -1)
    config = AFMConfig(
        mode="edit", implementation=implementation, schedule="frequency_neutral",
        strength=strength, preserve_dc=preserve_dc, normalization_mode=normalization_mode,
    )
    scale = 0.2
    raw = q.double()[..., text_tokens:, :] @ k.double().transpose(-2, -1) * scale
    text_logits = raw[..., :text_tokens]
    mean = text_logits.mean(-2, keepdim=True) if preserve_dc else 0.0
    edited = mean + (1.0 + strength / 2.0) * (text_logits - mean)
    if normalization_mode == "preserve_text_mass":
        edited = edited + text_logits.logsumexp(-1, keepdim=True) - edited.logsumexp(
            -1, keepdim=True
        )
    scores = raw.clone()
    scores[..., :text_tokens] = edited
    expected = scores.softmax(-1)

    kernel = joint_image_rows if implementation == "manual" else augmented_image_rows
    outputs = [
        kernel(q, k, v, layout, config, ProgressInfo(1, 5, progress, 0.5),
               ChunkPlan(1, 4, 1.0, implementation), scale)
        for progress in (0.0, 1.0)
    ]
    for actual in outputs:
        torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    if normalization_mode == "preserve_text_mass":
        baseline = raw.softmax(-1)
        torch.testing.assert_close(outputs[0][..., :text_tokens].double().sum(-1),
                                   baseline[..., :text_tokens].sum(-1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs[0][..., text_tokens:].double(),
                                   baseline[..., text_tokens:], atol=1e-5, rtol=1e-5)
