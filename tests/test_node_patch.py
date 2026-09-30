from dataclasses import replace

import pytest

from afm.config import AFMConfig
from afm.runtime import install_krea2_afm


@pytest.mark.parametrize("mode,strength", [("off", .05), ("edit", 0.)])
def test_zero_original_installs_no_attention_override(host, mode, strength):
    original = host.model_options
    patched = install_krea2_afm(host, AFMConfig(mode=mode, strength=strength))
    assert patched is not host
    assert host.model_options is original
    assert patched.model_options == original == {"transformer_options": {}}


@pytest.mark.parametrize("zero", ["manual", "augmented"])
def test_zero_controls_install_edit_path(host, zero):
    patched = install_krea2_afm(host, AFMConfig(mode="edit", strength=0, zero_strength_mode=zero))
    opts = patched.model_options["transformer_options"]
    assert opts["patches"]["post_input"]
    assert "optimized_attention_override" not in opts  # still delayed until after txtfusion
    assert host.model_options == {"transformer_options": {}}


def test_input_model_not_modified_and_duplicate_afm_rejected(host):
    first = install_krea2_afm(host, AFMConfig(mode="edit", block_scope="0"))
    second = install_krea2_afm(host, AFMConfig(mode="edit", block_scope="1"))
    assert first.model.diffusion_model is second.model.diffusion_model is host.model.diffusion_model
    assert first.model_options is not second.model_options
    assert host.model_options == {"transformer_options": {}}
    with pytest.raises(ValueError, match="conflict"):
        install_krea2_afm(first, AFMConfig(mode="edit"))


@pytest.mark.parametrize("patch", ["post_input", "attn1_patch", "attn1_output_patch"])
def test_existing_attention_patch_rejected(host, patch):
    host.model_options["transformer_options"]["patches"] = {patch: [lambda: None]}
    with pytest.raises(ValueError, match="conflict"):
        install_krea2_afm(host, AFMConfig(mode="edit"))
