import pytest
import torch

from conftest import Container
from afm.backend import dispatch_baseline, Krea2AFMAttentionOverride
from afm.config import AFMConfig
from afm.runtime import (
    AFMUnsupportedError, RUN_KEY, LAYOUT_KEY, BACKENDS_KEY, RunState,
    layout_from_latent,
)


def attention_options(backend, *, block=0):
    layout = layout_from_latent(torch.empty(1, 16, 4, 6), 2, 3)
    return {RUN_KEY: RunState(), LAYOUT_KEY: layout,
            BACKENDS_KEY: (backend,) * 4,
            "block_type": "single", "block_index": block, "total_blocks": 4,
            "img_slice": list(layout.image_slice), "cond_or_uncond": [0],
            "sigmas": [1.], "sample_sigmas": [1., 0.]}


@pytest.mark.parametrize("containers", [False, True])
@pytest.mark.parametrize("container_backend", [False, True])
def test_preferred_backend_preserved_and_container_consumed_once(host, containers, container_backend):
    calls = []
    baseline = torch.randn(1, 9, 32)
    qkv = [torch.randn(1, 4, 9, 8) for _ in range(3)]

    def preferred(q, k, v, heads, **kwargs):
        calls.append("tensor")
        assert torch.equal(q, qkv[0])
        assert kwargs["scale"] == .123
        assert kwargs["skip_reshape"]
        assert "optimized_attention_override" not in kwargs["transformer_options"]
        return baseline

    def container_entry(q, k, v, heads, **kwargs):
        calls.append("container")
        assert torch.equal(q.take(), qkv[0])
        k.take()
        v.take()
        return baseline

    if container_backend:
        preferred.container_function = container_entry
    options = attention_options(preferred)
    override = Krea2AFMAttentionOverride(AFMConfig())
    options["optimized_attention_override"] = override

    def global_backend(*args, **kwargs):
        pytest.fail("preferred module backend must win")

    inputs = [Container(tensor) for tensor in qkv] if containers else qkv
    if containers:
        output = override.container_function(*inputs, 4, transformer_options=options,
                                             skip_reshape=True, scale=.123)
        assert [value.takes for value in inputs] == [1, 1, 1]
    else:
        output = override(global_backend, *inputs, 4, transformer_options=options,
                          skip_reshape=True, scale=.123)
    assert output is baseline
    assert calls == ["container" if containers and container_backend else "tensor"]
    assert options[RUN_KEY].observed_calls == 1
    assert options["optimized_attention_override"] is override


def test_dispatch_preserves_mask_and_output_reshape():
    q = torch.ones(1, 1, 2, 4)
    mask = torch.ones(2, 2, dtype=torch.bool)
    def backend(q, k, v, heads, **kwargs):
        assert kwargs["mask"] is mask
        assert kwargs["skip_output_reshape"]
        return q
    assert dispatch_baseline(backend, q, q, q, 1,
                             {"mask": mask, "skip_output_reshape": True}) is q


def test_attention_rejects_img_slice_mismatch(host):
    options = attention_options(lambda *a, **k: None)
    options["img_slice"] = [2, 9]
    q = torch.zeros(1, 4, 9, 8)
    with pytest.raises(AFMUnsupportedError, match="img_slice"):
        Krea2AFMAttentionOverride(AFMConfig())(lambda *a, **k: None, q, q, q, 4,
                                             transformer_options=options, skip_reshape=True)


def test_fallback_dispatches_exactly_once(host):
    calls = []
    baseline = torch.randn(1, 9, 32)
    def backend(*args, **kwargs):
        calls.append(1)
        return baseline
    options = attention_options(backend)
    options["img_slice"] = [0, 9]
    qkv = [Container(torch.zeros(1, 4, 9, 8)) for _ in range(3)]
    result = Krea2AFMAttentionOverride(AFMConfig(fail_mode="fallback_original")).container_function(
        *qkv, 4, transformer_options=options, skip_reshape=True)
    assert result is baseline
    assert calls == [1]
    assert [c.takes for c in qkv] == [1, 1, 1]
    assert options[RUN_KEY].reasons["fallback_original"] == 1
    assert options[RUN_KEY].edited_calls == 0


def test_backend_exception_not_retried(host):
    calls = []
    def backend(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("backend failure")
    options = attention_options(backend)
    q = torch.zeros(1, 4, 9, 8)
    with pytest.raises(RuntimeError, match="backend failure"):
        Krea2AFMAttentionOverride(AFMConfig(fail_mode="fallback_original"))(
            backend, q, q, q, 4, transformer_options=options, skip_reshape=True)
    assert calls == [1]


@pytest.mark.parametrize("mode,expected", [("discover", 1), ("observe", 0)])
def test_discover_ignores_block_scope(host, mode, expected):
    q = torch.zeros(1, 4, 9, 8)
    baseline = q.transpose(1, 2).flatten(2)
    options = attention_options(lambda *a, **k: baseline, block=3)
    assert Krea2AFMAttentionOverride(AFMConfig(mode=mode, block_scope="0"))(
        None, q, q, q, 4, transformer_options=options, skip_reshape=True) is baseline
    assert options[RUN_KEY].observed_calls == expected


@pytest.mark.parametrize("branches", [torch.tensor([0]), [object()], object()])
def test_malformed_branch_metadata_stays_json_safe(host, tmp_path, branches):
    import json
    path = tmp_path / "branch.jsonl"
    baseline = torch.randn(1, 9, 32)
    calls = []
    def backend(*args, **kwargs):
        calls.append(1)
        return baseline
    options = attention_options(backend)
    options["cond_or_uncond"] = branches
    qkv = [Container(torch.zeros(1, 4, 9, 8)) for _ in range(3)]
    override = Krea2AFMAttentionOverride(AFMConfig(jsonl_path=str(path)))
    result = override.container_function(*qkv, 4, transformer_options=options, skip_reshape=True)
    assert result is baseline
    assert calls == [1]
    assert all(value.takes == 1 for value in qkv)
    record = json.loads(path.read_text())
    assert record["cond_or_uncond"] is None
    assert record["selected_batch_rows"] == []
    assert record["branch_reason"] == "unsupported_branch_layout"


@pytest.mark.parametrize("case,reason", [
    ("branch", "unsupported_branch_layout"), ("mask", "unsupported_mask"),
])
def test_unsupported_reason_counted_once_per_failed_call(host, case, reason):
    q = torch.randn(1, 4, 9, 8)
    baseline = q.transpose(1, 2).flatten(2).clone()
    options = attention_options(lambda *args, **kwargs: baseline)
    extra = {}
    if case == "branch":
        options["cond_or_uncond"] = None
    else:
        extra["mask"] = torch.ones(9, 9)
    config = AFMConfig(mode="edit", block_scope="0", fail_mode="fallback_original")
    result = Krea2AFMAttentionOverride(config)(
        None, q, q, q, 4, transformer_options=options, skip_reshape=True, **extra)
    assert result is baseline
    assert options[RUN_KEY].reasons[reason] == 1
    assert options[RUN_KEY].fallback_calls == 1


def test_preserve_diagnostics_include_schema_and_physical_head_mapping(host, tmp_path):
    import json
    qkv = [torch.randn(2, 4, 9, 8) for _ in range(3)]
    baseline = torch.randn(2, 9, 32)
    options = attention_options(lambda *args, **kwargs: baseline)
    options["cond_or_uncond"] = [1, 0]
    options["krea2_afm_model_eval_index"] = 1
    log = tmp_path / "preserve.jsonl"
    config = AFMConfig(
        mode="edit", normalization_mode="preserve_text_mass", block_scope="0",
        spectral_diag="sampled", diagnostic_head_indices="1,3", head_chunk_size=1,
        jsonl_path=str(log))
    output = Krea2AFMAttentionOverride(config)(
        None, *qkv, 4, transformer_options=options, skip_reshape=True)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    spectral = [event for event in events if event["event"] == "spectral_diagnostic"]
    assert {event["head_index"] for event in spectral} == {1, 3}
    assert all(event["schema_version"] == 5 for event in events)
    assert all(event["branch"] == 1 and event["branch_kind"] == 0
               and event["sample_index"] == 0 for event in spectral)
    assert all(event["normalization_mode"] == "preserve_text_mass"
               and event["selected_implementation"] == "manual"
               and event["edit_applied"] for event in spectral)
    assert all(event["text_mass_max_abs_delta"] <= 1e-5
               and event["image_probability_max_abs_delta"] <= 1e-5 for event in spectral)
    assert all("rho_text_conditional_after" not in event for event in spectral)
    assert torch.equal(output[0], baseline[0])
    assert torch.equal(output[1, :3], baseline[1, :3])
    assert options[RUN_KEY].reasons["normalization_requires_manual"] == 1


def test_unqualified_auto_reason_is_in_run_summary_counters(host):
    q = torch.randn(1, 4, 9, 8)
    options = attention_options(lambda *args, **kwargs: q.transpose(1, 2).flatten(2))
    Krea2AFMAttentionOverride(AFMConfig(mode="edit", block_scope="0"))(
        None, q, q, q, 4, transformer_options=options, skip_reshape=True)
    assert options[RUN_KEY].reasons["profile_not_qualified"] == 1


def test_edited_attention_call_is_counted_as_observed(host):
    q = torch.randn(1, 4, 9, 8)
    options = attention_options(lambda *args, **kwargs: q.transpose(1, 2).flatten(2))
    Krea2AFMAttentionOverride(AFMConfig(
        mode="edit", block_scope="0", normalization_mode="preserve_text_mass"))(
            None, q, q, q, 4, transformer_options=options, skip_reshape=True)
    assert options[RUN_KEY].observed_calls == 1
    assert options[RUN_KEY].edited_calls == 1
