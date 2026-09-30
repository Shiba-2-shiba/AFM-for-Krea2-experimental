"""Minimal host contract doubles; these do not establish real-model support."""
import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest


class SingleStreamDiT:
    def __init__(self):
        self.patch = 2
        self.txtfusion = object()
        self.txtmlp = object()
        self.blocks = [SimpleNamespace(attn=SimpleNamespace(
            heads=4, kvheads=2, headdim=8,
            comfy_attention=SimpleNamespace(function=None, config=None),
        )) for _ in range(4)]

    def process_img(self, *args):
        raise NotImplementedError

    def _unpack_context(self, *args):
        raise NotImplementedError


class Container:
    def __init__(self, tensor):
        self.tensor = tensor
        self.takes = 0

    def peek(self):
        if self.tensor is None:
            raise RuntimeError("container consumed twice")
        return self.tensor

    def take(self):
        tensor = self.peek()
        self.tensor = None
        self.takes += 1
        return tensor


class WrappersMP:
    OUTER_SAMPLE = "outer_sample"
    DIFFUSION_MODEL = "diffusion_model"


class ModelPatcher:
    def __init__(self, diffusion_model=None):
        self.model = SimpleNamespace(diffusion_model=diffusion_model or SingleStreamDiT())
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        return getattr(self.model, name)

    def clone(self):
        result = ModelPatcher(self.model.diffusion_model)
        result.model_options = copy.deepcopy(self.model_options)
        return result

    def set_model_post_input_patch(self, patch):
        opts = self.model_options["transformer_options"]
        opts.setdefault("patches", {}).setdefault("post_input", []).append(patch)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        opts = self.model_options["transformer_options"]
        opts.setdefault("wrappers", {}).setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


@pytest.fixture
def host(monkeypatch):
    modules = {}
    for name in ("comfy", "comfy.ldm", "comfy.ldm.krea2", "comfy.ldm.krea2.model",
                 "comfy.ldm.modules", "comfy.ldm.modules.attention", "comfy.patcher_extension"):
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    modules["comfy.ldm.krea2.model"].SingleStreamDiT = SingleStreamDiT
    modules["comfy.ldm.modules.attention"].AttentionTensorContainer = Container
    modules["comfy.ldm.modules.attention"].optimized_attention_masked = lambda *a, **k: None
    modules["comfy.patcher_extension"].WrappersMP = WrappersMP
    return ModelPatcher()
