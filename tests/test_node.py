import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def node_module(monkeypatch, host, tmp_path):
    """Minimal V3 schema double; not a substitute for a real ComfyUI load."""
    class Input:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)
    class Output:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class NodeOutput:
        def __init__(self, *values):
            self.result = values
    latest = ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})
    latest.io = SimpleNamespace(ComfyNode=object, Schema=Schema, NodeOutput=NodeOutput,
                               **{name: SimpleNamespace(Input=Input, Output=Output)
                                  for name in ("Model", "Combo", "String", "Float", "Int", "Boolean")})
    monkeypatch.setitem(sys.modules, "comfy_api", ModuleType("comfy_api"))
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    paths = ModuleType("folder_paths")
    paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("krea2_afm_test_package", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    extension = asyncio.run(package.comfy_entrypoint())
    return asyncio.run(extension.get_node_list())[0]


def test_v3_node_schema(node_module):
    schema = node_module.define_schema()
    assert schema.node_id == "Krea2AFMModelPatch"
    fields = {field.name: field for field in schema.inputs}
    assert fields["mode"].default == "observe"
    assert fields["mode"].options == ["off", "discover", "observe", "edit"]
    assert fields["fail_mode"].default == "raise"
    assert schema.is_experimental
    assert len(schema.outputs) == 1


def test_node_clones_model(node_module, host):
    result = node_module.execute(host)
    assert result.result[0] is not host
    assert host.model_options == {"transformer_options": {}}


def test_node_exposes_edit(node_module, host):
    patched = node_module.execute(host, mode="edit").result[0]
    assert patched.model_options["transformer_options"]["krea2_afm_config"].mode == "edit"


def test_edit_controls_use_shared_config_defaults(node_module):
    from afm.config import AFMConfig
    fields = {field.name: field for field in node_module.define_schema().inputs}
    for name in ("strength", "implementation", "cutoff", "schedule", "preserve_dc", "soft_mask",
                 "mask_width", "zero_strength_mode", "head_chunk_size", "query_chunk_size",
                 "max_workspace_mib", "augmented_precision", "spectral_diag", "compare_kernels"):
        assert fields[name].default == getattr(AFMConfig(), name)


@pytest.mark.parametrize("path", ["../escape.jsonl", "krea2_afm/../../escape.jsonl",
                                  "other/log.jsonl", "krea2_afm/file.txt"])
def test_log_path_is_confined_to_diagnostic_directory(node_module, host, path):
    with pytest.raises(ValueError, match="jsonl_path"):
        node_module.execute(host, jsonl_path=path)


def test_absolute_log_path_rejected(node_module, host, tmp_path):
    with pytest.raises(ValueError, match="jsonl_path"):
        node_module.execute(host, jsonl_path=str(tmp_path / "outside.jsonl"))


def test_log_path_follows_output_directory(node_module, host, tmp_path):
    patched = node_module.execute(host, jsonl_path="krea2_afm/run.jsonl").result[0]
    config = patched.model_options["transformer_options"]["krea2_afm_config"]
    assert Path(config.jsonl_path) == tmp_path / "krea2_afm" / "run.jsonl"


def test_log_path_rejects_resolved_escape(node_module, host, tmp_path, monkeypatch):
    original_resolve = Path.resolve
    redirected = tmp_path / "krea2_afm" / "redirect" / "run.jsonl"
    def resolve(path, *args, **kwargs):
        if path == redirected:
            return tmp_path / "outside" / "run.jsonl"
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="jsonl_path"):
        node_module.execute(host, jsonl_path="krea2_afm/redirect/run.jsonl")
