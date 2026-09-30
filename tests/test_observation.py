from dataclasses import replace

import pytest
import torch

from afm.config import AFMConfig
from afm.memory import choose_chunks
from afm.runtime import layout_from_latent, ProgressInfo
from afm.observation import collect_attention_diagnostics, diagnostic_requested


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
    assert 0 <= record["rho_text_conditional_before"] <= 1


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
