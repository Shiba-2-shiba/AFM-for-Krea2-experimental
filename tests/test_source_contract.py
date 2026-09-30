"""Optional source-only checks: no ComfyUI server, model, weights or GPU.

Set KREA2_COMFY_SOURCE_ROOT to a source checkout. Extract only upstream
container/decorator/wrapper definitions with AST so their real dispatch logic
is tested without importing ComfyUI's GPU-dependent modules.
"""
import ast
import functools
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from afm.backend import Krea2AFMAttentionOverride
from afm.config import AFMConfig
from afm.runtime import (BACKENDS_KEY, LAYOUT_KEY, RUN_KEY, RunState,
                         layout_from_latent, make_outer_sample_wrapper)


@pytest.fixture
def upstream():
    source = os.environ.get("KREA2_COMFY_SOURCE_ROOT")
    if not source:
        pytest.skip("set KREA2_COMFY_SOURCE_ROOT for source-only upstream contract checks")
    root = Path(source)
    namespace = {"torch": torch, "functools": functools, "Callable": Callable}
    selections = {
        "comfy/ldm/modules/attention.py": {"AttentionTensorContainer", "wrap_attn"},
        "comfy/patcher_extension.py": {"WrapperExecutor", "copy_nested_dicts", "merge_nested_dicts"},
    }
    for filename, names in selections.items():
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        chosen = [node for node in tree.body if getattr(node, "name", None) in names]
        assert {node.name for node in chosen} == names
        module = ast.Module(body=chosen, type_ignores=[])
        exec(compile(module, str(root / filename), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("preferred", [False, True])
def test_upstream_wrap_attn_dispatch(host, monkeypatch, upstream, preferred):
    import comfy.ldm.modules.attention as attention
    monkeypatch.setattr(attention, "AttentionTensorContainer", upstream.AttentionTensorContainer)
    calls = []
    expected = torch.randn(1, 9, 32)
    @upstream.wrap_attn
    def standard(q, k, v, heads, **kwargs):
        calls.append("standard")
        return expected
    @upstream.wrap_attn
    def module_backend(q, k, v, heads, **kwargs):
        pytest.fail("container-capable preferred backend should keep its container entry")
    def container_entry(q, k, v, heads, **kwargs):
        calls.append("preferred")
        q.take()
        k.take()
        v.take()
        return expected
    module_backend.container_function = container_entry
    monkeypatch.setattr(attention, "optimized_attention_masked", standard)
    selected = module_backend if preferred else None
    layout = layout_from_latent(torch.zeros(1, 16, 4, 6), 2, 3)
    options = {RUN_KEY: RunState(), LAYOUT_KEY: layout, BACKENDS_KEY: (selected,),
               "block_type": "single", "block_index": 0, "total_blocks": 1,
               "img_slice": [3, 9], "cond_or_uncond": [0],
               "sample_sigmas": [1., 0.], "sigmas": [1.],
               "optimized_attention_override": Krea2AFMAttentionOverride(AFMConfig(block_scope="all"))}
    qkv = [upstream.AttentionTensorContainer(torch.zeros(1, 4, 9, 8)) for _ in range(3)]
    result = standard(*qkv, 4, skip_reshape=True, transformer_options=options,
                      preferred_attention=SimpleNamespace(function=selected))
    assert result is expected
    assert calls == ["preferred" if preferred else "standard"]
    assert all(container.tensor is None for container in qkv)


def test_upstream_wrapper_and_nested_options_keep_run_identity(host, upstream):
    original = host.model_options
    guider = SimpleNamespace(model_options=original)
    recorded = []
    def sample(*args, **kwargs):
        state = guider.model_options["transformer_options"][RUN_KEY]
        cloned = upstream.copy_nested_dicts(guider.model_options)
        merged = upstream.merge_nested_dicts({}, cloned["transformer_options"])
        assert merged[RUN_KEY] is state
        recorded.append(state)
        return 42
    executor = upstream.WrapperExecutor.new_class_executor(
        sample, guider, [make_outer_sample_wrapper(AFMConfig(debug_level="off"))])
    assert executor.execute(None, None, None, [1., 0.]) == 42
    assert guider.model_options is original
    assert recorded[0].finished
