from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from afm.capabilities import detect_krea2, detect_patch_conflicts
from afm.config import AFMConfig, validate_config


def test_native_krea2_attributes(host):
    caps = detect_krea2(host)
    assert (caps.patch_size, caps.block_count, caps.query_heads, caps.kv_heads) == (2, 4, 4, 2)
    assert caps.uses_containers
    assert caps.sdpa_explicit_scale
    with pytest.raises(FrozenInstanceError):
        AFMConfig().mode = "edit"


@pytest.mark.parametrize("name", ["Anima", "Flux", "SingleStreamDiT"])
def test_reject_non_krea2(host, name):
    host.model.diffusion_model = type(name, (), {"blocks": [object()]})()
    with pytest.raises(ValueError, match="native Krea2"):
        detect_krea2(host)


@pytest.mark.parametrize("changes", [
    {"start_percent": .8, "end_percent": .2}, {"block_scope": "0,4"},
    {"block_scope": "3-1"}, {"block_scope": "encoder_equivalent"},
    {"head_chunk_size": 0}, {"query_chunk_size": 0}, {"strength": float("nan")},
    {"strength": float("inf")}, {"max_workspace_mib": 0}, {"cutoff": 1.1},
    {"implementation": "guess"}, {"zero_strength_mode": "observe"},
    {"fail_mode": "ignore"}, {"spectral_diag": "sampled"},
])
def test_scope_and_window_validation(host, changes):
    with pytest.raises(ValueError):
        validate_config(replace(AFMConfig(), **changes), detect_krea2(host))


def test_required_host_api_detection(host):
    host.set_model_post_input_patch = None
    with pytest.raises(ValueError, match="set_model_post_input_patch"):
        detect_krea2(host)


def test_head_structure_must_be_consistent(host):
    host.model.diffusion_model.blocks[2].attn.heads = 3
    with pytest.raises(ValueError, match="head"):
        detect_krea2(host)


def test_sdpa_scale_feature_detection(host, monkeypatch):
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", lambda q, k, v: q)
    assert not detect_krea2(host).sdpa_explicit_scale


@pytest.mark.parametrize("patch", ["post_input", "attn1_patch", "attn1_output_patch"])
def test_attention_patch_conflicts(host, patch):
    host.model_options["transformer_options"]["patches"] = {patch: [object()]}
    assert any(patch in conflict for conflict in detect_patch_conflicts(host))


def test_override_and_duplicate_conflicts(host):
    opts = host.model_options["transformer_options"]
    opts["optimized_attention_override"] = object()
    opts["krea2_afm_config"] = AFMConfig()
    assert len(detect_patch_conflicts(host)) == 2


def test_lora_weight_patches_not_an_attention_conflict(host):
    host.patches = {"some.weight": [(1.0, object())]}
    assert detect_patch_conflicts(host) == []
