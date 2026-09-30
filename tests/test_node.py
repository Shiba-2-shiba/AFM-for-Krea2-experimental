import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def node_module(monkeypatch, host, tmp_path):
    """V3 schema double; source-contract test separately checks actual io syntax."""
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
                                  for name in ("Model", "Combo", "String", "Float")})
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
    assert fields["mode"].options == ["off", "discover", "observe"]
    assert fields["fail_mode"].default == "raise"
    assert schema.is_experimental
    assert len(schema.outputs) == 1


def test_node_clones_model(node_module, host):
    result = node_module.execute(host)
    assert result.result[0] is not host
    assert host.model_options == {"transformer_options": {}}


def test_node_refuses_edit_even_via_api(node_module, host):
    with pytest.raises(NotImplementedError, match="diagnostic"):
        node_module.execute(host, mode="edit")
