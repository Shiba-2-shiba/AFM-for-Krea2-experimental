import json
from dataclasses import replace

import pytest
import torch

from conftest import Container
import afm.backend as backend_module
from afm.backend import Krea2AFMAttentionOverride
from afm.config import AFMConfig
from afm.runtime import (RUN_KEY, LAYOUT_KEY, BACKENDS_KEY, EVAL_KEY, RunState,
                         layout_from_latent, progress_from_sigmas, branch_selection)


def edit_case(branches, *, merged=True):
    torch.manual_seed(31)
    batch = 2 * len(branches) if branches else 2
    q, k, v = [torch.randn(batch, 3, 18, 8) for _ in range(3)]
    baseline = torch.randn(batch, 18, 24) if merged else torch.randn(batch, 3, 18, 8)
    calls = []
    def original(q, k, v, heads, **kwargs):
        calls.append(1)
        return baseline
    layout = layout_from_latent(torch.empty(batch, 16, 5, 9), 2, 3)
    opts = {RUN_KEY: RunState(), LAYOUT_KEY: layout, BACKENDS_KEY: (original,),
            EVAL_KEY: 0, "block_type": "single", "block_index": 0, "total_blocks": 1,
            "img_slice": [3, 18], "cond_or_uncond": branches,
            "sample_sigmas": [2., 1., 0.], "sigmas": [1.]}
    return (q, k, v), baseline, original, opts, calls


@pytest.mark.parametrize("implementation", ["manual", "augmented"])
@pytest.mark.parametrize("branches", [[0], [1], [1, 0], [0, 1]])
@pytest.mark.parametrize("merged", [True, False])
def test_unselected_branch_and_text_rows_bitwise_equal(host, implementation, branches, merged):
    qkv, baseline, original, opts, calls = edit_case(branches, merged=merged)
    snapshot = baseline.clone()
    cfg = AFMConfig(mode="edit", implementation=implementation, block_scope="0",
                    strength=.2, head_chunk_size=2, query_chunk_size=4)
    result = Krea2AFMAttentionOverride(cfg)(original, *qkv, 3, transformer_options=opts,
                                          skip_reshape=True, skip_output_reshape=not merged,
                                          scale=.123)
    assert calls == [1]
    assert torch.equal(baseline, snapshot)
    selected = branch_selection(branches, baseline.shape[0], "positive_only")[0]
    for row in range(baseline.shape[0]):
        if row not in selected:
            assert torch.equal(result[row], baseline[row])
        elif merged:
            assert torch.equal(result[row, :3], baseline[row, :3])
            assert not torch.equal(result[row, 3:], baseline[row, 3:])
        else:
            assert torch.equal(result[row, :, :3], baseline[row, :, :3])
            assert not torch.equal(result[row, :, 3:], baseline[row, :, 3:])
    if selected:
        from reference import dense_joint_image_rows
        expected = dense_joint_image_rows(*qkv, opts[LAYOUT_KEY], cfg, progress_from_sigmas(opts), .123)
        for row in selected:
            actual = result[row, 3:].reshape(15, 3, 8).transpose(0, 1) if merged else result[row, :, 3:]
            torch.testing.assert_close(actual, expected[row], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("implementation", ["manual", "augmented"])
@pytest.mark.parametrize("failure", ["oom", "nan"])
@pytest.mark.parametrize("fail_mode", ["raise", "fallback_original"])
def test_fallback_reuses_baseline_once(host, monkeypatch, tmp_path, implementation, failure, fail_mode):
    qkv, baseline, original, opts, calls = edit_case([0])
    containers = [Container(value) for value in qkv]
    kernel_calls = []
    error = torch.OutOfMemoryError("injected OOM") if failure == "oom" else FloatingPointError("NaN chunk")
    def failing_kernel(q, k, v, *args, **kwargs):
        kernel_calls.append(1)
        if len(kernel_calls) == 2:
            raise error
        return torch.ones_like(q[:, :, 3:])
    kernel_name = "joint_image_rows" if implementation == "manual" else "augmented_image_rows"
    monkeypatch.setattr(backend_module, kernel_name, failing_kernel, raising=False)
    alternate = "augmented_image_rows" if implementation == "manual" else "joint_image_rows"
    monkeypatch.setattr(backend_module, alternate, lambda *a, **k: pytest.fail("no alternate retry"), raising=False)
    cfg = AFMConfig(mode="edit", implementation=implementation, fail_mode=fail_mode,
                    block_scope="0", jsonl_path=str(tmp_path / "failure.jsonl"))
    def execute():
        return Krea2AFMAttentionOverride(cfg).container_function(
            *containers, 3, transformer_options=opts, skip_reshape=True)
    if fail_mode == "raise":
        with pytest.raises(type(error), match="injected|NaN"):
            execute()
    else:
        assert execute() is baseline
        records = [json.loads(line) for line in (tmp_path / "failure.jsonl").read_text().splitlines()]
        assert records[-1]["edit_applied"] is False
        assert opts[RUN_KEY].reasons["fallback_original"] == 1
    assert kernel_calls == [1, 1]
    assert calls == [1]
    assert [value.takes for value in containers] == [1, 1, 1]
    assert opts[RUN_KEY].edited_calls == 0


@pytest.mark.parametrize("override", [{"mask": torch.ones(18, 18)}, {"is_causal": True},
                                     {"dropout_p": .1}, {"enable_gqa": True}, {"logit_bias": 1.}])
def test_unsupported_edit_conditions_return_baseline_once(host, override):
    qkv, baseline, original, opts, calls = edit_case([0])
    cfg = AFMConfig(mode="edit", block_scope="0", fail_mode="fallback_original")
    result = Krea2AFMAttentionOverride(cfg)(original, *qkv, 3, transformer_options=opts,
                                          skip_reshape=True, **override)
    assert result is baseline
    assert calls == [1]
    assert opts[RUN_KEY].edited_calls == 0
    assert opts[RUN_KEY].reasons["fallback_original"] == 1


def test_missing_branch_metadata_never_assumed_positive(host):
    qkv, baseline, original, opts, calls = edit_case(None)
    result = Krea2AFMAttentionOverride(AFMConfig(mode="edit", block_scope="0", fail_mode="fallback_original"))(
        original, *qkv, 3, transformer_options=opts, skip_reshape=True)
    assert result is baseline
    assert calls == [1]
    assert opts[RUN_KEY].edited_calls == 0


def test_auto_path_unqualified_is_logged_manual(host, tmp_path):
    qkv, baseline, original, opts, calls = edit_case([0])
    path = tmp_path / "edit.jsonl"
    cfg = AFMConfig(mode="edit", block_scope="0", jsonl_path=str(path))
    result = Krea2AFMAttentionOverride(cfg)(original, *qkv, 3, transformer_options=opts, skip_reshape=True)
    assert result is not baseline
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["edit_applied"] is True
    assert row["selected_implementation"] == "manual"
    assert row["selection_reason"] == "profile_not_qualified"
    assert row["max_logits_delta"] is None


def test_normal_augmented_path_has_no_dense_diagnostics(host, monkeypatch):
    import afm.observation as observation
    monkeypatch.setattr(observation, "collect_attention_diagnostics",
                        lambda *a, **k: pytest.fail("diagnostics must stay off"))
    qkv, baseline, original, opts, calls = edit_case([0])
    config = AFMConfig(mode="edit", implementation="augmented", block_scope="0")
    Krea2AFMAttentionOverride(config)(original, *qkv, 3, transformer_options=opts, skip_reshape=True)
    assert opts[RUN_KEY].edited_calls == 1


def test_sampled_diagnostics_and_same_qkv_records(host, tmp_path):
    qkv, baseline, original, opts, calls = edit_case([1, 0])
    path = tmp_path / "diagnostics.jsonl"
    config = AFMConfig(mode="edit", implementation="augmented", block_scope="0",
                       spectral_diag="sampled", compare_kernels=True, diagnostic_head_limit=1,
                       diagnostic_query_chunk_size=4, jsonl_path=str(path))
    result = Krea2AFMAttentionOverride(config)(original, *qkv, 3, transformer_options=opts, skip_reshape=True)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    comparisons = [row for row in records if row["event"] == "kernel_comparison"]
    spectral = [row for row in records if row["event"] == "spectral_diagnostic"]
    assert len(comparisons) == len(spectral) == 2  # positive batch rows 2,3 only
    assert {row["branch"] for row in comparisons} == {2, 3}
    assert all(row["unselected_rows_preserved"] for row in comparisons)
    assert all(row["diagnostic_heads"] == [0] for row in spectral)
    assert all(row["diagnostic_query_chunk_size"] <= 4 for row in spectral)
    assert torch.equal(result[:2], baseline[:2])


def test_diagnostic_scope_can_observe_unedited_block(host, tmp_path):
    qkv, baseline, original, opts, calls = edit_case([0])
    opts.update(block_index=1, total_blocks=2)
    opts[BACKENDS_KEY] = (original, original)
    path = tmp_path / "outside_edit_scope.jsonl"
    config = AFMConfig(mode="edit", block_scope="0", diagnostic_block_scope="1",
                       spectral_diag="sampled", jsonl_path=str(path))
    assert Krea2AFMAttentionOverride(config)(original, *qkv, 3, transformer_options=opts,
                                            skip_reshape=True) is baseline
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(row["event"] == "spectral_diagnostic" for row in records)
    assert opts[RUN_KEY].edited_calls == 0
