import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from afm.config import AFMConfig
from afm.runtime import (
    AFMUnsupportedError, RUN_KEY, LAYOUT_KEY, RunState, branch_selection,
    layout_from_latent, validate_post_input, progress_from_sigmas,
    make_outer_sample_wrapper, make_diffusion_wrapper, make_post_input_patch,
    install_krea2_afm,
)


def payload_for(layout, batch=1):
    h, w = layout.grid
    ids = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1)
    ids = torch.cat((torch.zeros(h, w, 1), ids), -1).reshape(1, h * w, 3)
    return {"img": torch.zeros(batch, h * w, 8),
            "txt": torch.zeros(batch, layout.text_slice[1], 8),
            "img_ids": ids.repeat(batch, 1, 1),
            "txt_ids": torch.zeros(batch, layout.text_slice[1], 3),
            "transformer_options": {}}


def test_grid_uses_patch_padding():
    layout = layout_from_latent(torch.empty(1, 16, 65, 97), 2, 5)
    assert layout.grid == (33, 49)
    assert layout.image_slice == (5, 1622)
    assert layout.total_tokens == 1622
    assert validate_post_input(payload_for(layout), layout) == layout


def test_single_frame_5d_matches_4d():
    expected = layout_from_latent(torch.empty(1, 16, 5, 9), 2, 5)
    assert layout_from_latent(torch.empty(1, 16, 1, 5, 9), 2, 5) == expected
    with pytest.raises(AFMUnsupportedError, match="multiple_frames"):
        layout_from_latent(torch.empty(1, 16, 2, 5, 9), 2, 5)


@pytest.mark.parametrize("change", ["reverse", "extra", "source", "batch", "nonfinite"])
def test_post_input_rejects_reordered_or_extra_tokens(change):
    layout = layout_from_latent(torch.empty(1, 16, 4, 6), 2, 3)
    payload = payload_for(layout)
    if change == "reverse":
        payload["img_ids"] = payload["img_ids"].flip(1)
    elif change == "extra":
        payload["img"] = torch.zeros(1, 7, 8)
    elif change == "source":
        payload["img_ids"][0, -1, 0] = 1
    elif change == "batch":
        payload["img_ids"] = payload["img_ids"].repeat(2, 1, 1)
    else:
        payload["img_ids"][0, 0, 0] = float("nan")
    with pytest.raises(AFMUnsupportedError):
        validate_post_input(payload, layout)


@pytest.mark.parametrize("sigma,expected", [(10., (0, 3, 0.)), (5., (1, 3, .5)), (1., (2, 3, 1.))])
def test_progress_uses_actual_schedule(sigma, expected):
    info = progress_from_sigmas({"sample_sigmas": torch.tensor([10., 5., 1., 0.]),
                                 "sigmas": torch.tensor([sigma, sigma])})
    assert (info.step_index, info.num_steps, info.progress) == expected


def test_progress_one_step_and_img2img_subset():
    assert progress_from_sigmas({"sample_sigmas": [5., 0.], "sigmas": [5.]}).progress == 0.
    assert progress_from_sigmas({"sample_sigmas": [5., 1., 0.], "sigmas": [1.]}).progress == 1.


@pytest.mark.parametrize("options", [
    {}, {"sample_sigmas": [5., 1., 0.], "sigmas": [3.]},
    {"sample_sigmas": [5., 1., 0.], "sigmas": [5., 1.]},
    {"sample_sigmas": [5., 5., 0.], "sigmas": [5.]},
    {"sample_sigmas": [5., 1., 0.], "sigmas": [float("nan")]},
    {"sample_sigmas": [1., 5., 0.], "sigmas": [1.]},
])
def test_unsupported_progress_not_nearest_sigma(options):
    assert progress_from_sigmas(options) is None


@pytest.mark.parametrize("branches,batch,indices", [
    ([0], 2, (0, 1)), ([1], 2, ()), ([1, 0], 4, (2, 3)),
    ([0, 1], 4, (0, 1)), (None, 1, ()), ([0, 1], 3, ()),
])
def test_branch_layouts(branches, batch, indices):
    assert branch_selection(branches, batch, "positive_only")[0] == indices


class Executor:
    def __init__(self, obj, fn):
        self.class_obj = obj
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def test_txtfusion_has_no_override_and_options_not_mutated(host):
    config = AFMConfig()
    state = RunState()
    original = {RUN_KEY: state, "sample_sigmas": [1., 0.], "sigmas": [1.]}
    patch = make_post_input_patch(config)
    seen = []

    def forward(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs):
        assert "optimized_attention_override" not in transformer_options  # txtfusion entry
        local = transformer_options.copy()  # native Krea2 _forward contract
        payload = payload_for(layout_from_latent(x, 2, context.shape[1]))
        payload["transformer_options"] = local
        result = patch(payload)
        assert result is payload
        assert "optimized_attention_override" in local
        seen.append(local[LAYOUT_KEY])
        return x

    model = host.model.diffusion_model
    wrapper = make_diffusion_wrapper(config)
    latent = torch.zeros(1, 16, 5, 9)
    for _ in range(2):
        assert wrapper(Executor(model, forward), latent, torch.ones(1), torch.zeros(1, 3, 8),
                       None, None, original) is latent
        assert "optimized_attention_override" not in original
        assert LAYOUT_KEY not in original
    assert len(seen) == 2
    assert state.model_evaluations == 2


def test_reference_rejected_before_forward(host):
    def forward(*args, **kwargs):
        pytest.fail("reference should be rejected before native forward")
    with pytest.raises(AFMUnsupportedError, match="reference_not_supported"):
        make_diffusion_wrapper(AFMConfig())(
            Executor(host.model.diffusion_model, forward), torch.zeros(1, 16, 4, 4),
            torch.ones(1), torch.zeros(1, 3, 8), ref_latents=[torch.zeros(1)],
            transformer_options={RUN_KEY: RunState()})


def test_run_state_isolated_after_exception(host, tmp_path):
    log = tmp_path / "runs.jsonl"
    config = replace(AFMConfig(), jsonl_path=str(log))
    wrapper = make_outer_sample_wrapper(config)
    seen = []
    for index, model in enumerate((host.clone(), host.clone(), host.clone())):
        original = model.model_options
        guider = SimpleNamespace(model_options=original, model_patcher=model)

        def sample(*args, **kwargs):
            assert guider.model_options is not original
            state = guider.model_options["transformer_options"][RUN_KEY]
            seen.append(state)
            if index == 1:
                raise RuntimeError("sample failed")
            return "ok"

        executor = Executor(guider, sample)
        if index == 1:
            with pytest.raises(RuntimeError, match="sample failed"):
                wrapper(executor, None, None, None, [1., 0.])
        else:
            assert wrapper(executor, None, None, None, [1., 0.]) == "ok"
        assert guider.model_options is original
        assert RUN_KEY not in original["transformer_options"]
    assert len({state.run_id for state in seen}) == 3
    assert all(state.finished for state in seen)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    summaries = [row for row in records if row["event"] == "run_summary"]
    assert len(summaries) == 3
    assert [row["status"] for row in summaries] == ["ok", "error", "ok"]
    assert all(row["edited_calls"] == 0 for row in summaries)


def test_off_no_hooks_and_input_model_unchanged(host):
    clone = install_krea2_afm(host, AFMConfig(mode="off"))
    assert clone is not host
    assert clone.model_options == host.model_options == {"transformer_options": {}}


def test_install_clone_and_conflict(host):
    clone = install_krea2_afm(host, AFMConfig())
    assert host.model_options == {"transformer_options": {}}
    assert clone.model.diffusion_model is host.model.diffusion_model
    assert "optimized_attention_override" not in clone.model_options["transformer_options"]
    assert set(clone.model_options["transformer_options"]["wrappers"]) == {
        "outer_sample", "diffusion_model"}
    with pytest.raises(ValueError, match="conflict"):
        install_krea2_afm(clone, AFMConfig())


@pytest.mark.parametrize("strength", [0., .05])
def test_edit_is_explicitly_unimplemented(host, strength):
    with pytest.raises(NotImplementedError, match="diagnostic"):
        install_krea2_afm(host, AFMConfig(mode="edit", strength=strength))


@pytest.mark.parametrize("patch_name", ["post_input", "attn1_patch", "attn1_output_patch"])
def test_patch_added_after_afm_is_rejected_before_forward(host, patch_name):
    clone = install_krea2_afm(host, AFMConfig())
    options = clone.model_options["transformer_options"]
    options.setdefault("patches", {}).setdefault(patch_name, []).append(lambda *a, **k: None)
    options[RUN_KEY] = RunState()
    called = []
    wrapper = options["wrappers"]["diffusion_model"]["krea2_afm"][0]
    def forward(*args, **kwargs):
        called.append(True)
        return args[0]
    with pytest.raises(ValueError, match="conflict"):
        wrapper(Executor(clone.model.diffusion_model, forward), torch.zeros(1, 16, 4, 6),
                torch.ones(1), torch.zeros(1, 3, 8), transformer_options=options)
    assert not called


def test_summary_io_failure_does_not_mask_primary_error(host, monkeypatch):
    import afm.runtime as runtime
    attempted = []
    def emitter(settings, run_id, event, **kwargs):
        attempted.append(event)
        if event == "run_summary":
            raise OSError("log disk unavailable")
    monkeypatch.setattr(runtime, "emit_record", emitter)
    original = host.model_options
    guider = SimpleNamespace(model_options=original)
    states = []
    def sample(*args, **kwargs):
        states.append(guider.model_options["transformer_options"][RUN_KEY])
        raise RuntimeError("primary backend error")
    with pytest.raises(RuntimeError, match="primary backend error"):
        make_outer_sample_wrapper(AFMConfig())(Executor(guider, sample))
    assert attempted == ["run_start", "run_summary"]
    assert guider.model_options is original
    assert states[0].finished


def test_summary_failure_is_reported_on_successful_sample(host, monkeypatch):
    import afm.runtime as runtime
    def emitter(settings, run_id, event, **kwargs):
        if event == "run_summary":
            raise OSError("summary unavailable")
    monkeypatch.setattr(runtime, "emit_record", emitter)
    guider = SimpleNamespace(model_options=host.model_options)
    with pytest.raises(OSError, match="summary unavailable"):
        make_outer_sample_wrapper(AFMConfig())(Executor(guider, lambda: 42))
    assert guider.model_options is host.model_options


@pytest.mark.parametrize("signal", ["clone_options", "registered_models"])
def test_multigpu_rejected_before_outer_sample_creates_pool(host, signal):
    options = host.model_options
    if signal == "clone_options":
        options["multigpu_clones"] = {"cuda:1": object()}
    else:
        host.get_additional_models_with_key = lambda key: [object()] if key == "multigpu" else []
    guider = SimpleNamespace(model_options=options, model_patcher=host)
    entered = []
    def sample():
        entered.append(True)
        guider.model_options["multigpu_thread_pool"] = object()
    with pytest.raises(AFMUnsupportedError, match="multi_gpu_not_supported"):
        make_outer_sample_wrapper(AFMConfig())(Executor(guider, sample))
    assert not entered
    assert guider.model_options is options
