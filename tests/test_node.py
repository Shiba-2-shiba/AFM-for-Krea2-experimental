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


def test_frequency_neutral_schedule_is_explicit_opt_in(node_module):
    fields = {field.name: field for field in node_module.define_schema().inputs}
    assert fields["schedule"].options == ["curve", "frequency_neutral"]
    assert fields["schedule"].default == "curve"
    assert fields["schedule"].advanced is True


def test_node_accepts_and_forwards_frequency_neutral_schedule(node_module, host):
    patched = node_module.execute(
        host, mode="edit", schedule="frequency_neutral", jsonl_path=""
    ).result[0]
    config = patched.model_options["transformer_options"]["krea2_afm_config"]
    assert config.schedule == "frequency_neutral"
    assert config.preserve_dc is True
    assert config.normalization_mode == "joint"


def test_edit_controls_use_shared_config_defaults(node_module):
    from afm.config import AFMConfig
    fields = {field.name: field for field in node_module.define_schema().inputs}
    for name in ("strength", "implementation", "cutoff", "schedule", "preserve_dc", "soft_mask",
                 "mask_width", "zero_strength_mode", "head_chunk_size", "query_chunk_size",
                 "max_workspace_mib", "augmented_precision", "spectral_diag", "compare_kernels",
                 "normalization_mode", "diagnostic_head_indices", "diagnostic_radial_bins"):
        assert fields[name].default == getattr(AFMConfig(), name)


def test_new_controls_are_optional_advanced_and_appended(node_module):
    fields = node_module.define_schema().inputs
    names = [field.name for field in fields]
    assert names[-3:] == [
        "normalization_mode",
        "diagnostic_head_indices",
        "diagnostic_radial_bins",
    ]
    for field in fields[-3:]:
        assert field.optional is True
        assert field.advanced is True


def test_execute_omitting_new_controls_preserves_legacy_defaults(node_module, host):
    patched = node_module.execute(host, mode="edit", jsonl_path="").result[0]
    config = patched.model_options["transformer_options"]["krea2_afm_config"]
    assert config.normalization_mode == "joint"
    assert config.diagnostic_head_indices == ""
    assert config.diagnostic_radial_bins == 32


def test_execute_forwards_new_controls(node_module, host):
    patched = node_module.execute(
        host,
        mode="edit",
        jsonl_path="",
        normalization_mode="preserve_text_mass",
        diagnostic_head_indices="1,3",
        diagnostic_radial_bins=64,
    ).result[0]
    config = patched.model_options["transformer_options"]["krea2_afm_config"]
    assert config.normalization_mode == "preserve_text_mass"
    assert config.diagnostic_head_indices == "1,3"
    assert config.diagnostic_radial_bins == 64


@pytest.mark.parametrize(
    "value,expected",
    [("", (0, 1, 2)), ("2", (2,)), (" 3, 1 ", (3, 1))],
)
def test_parse_diagnostic_heads_uses_physical_indices(value, expected):
    from afm.config import parse_diagnostic_heads

    assert parse_diagnostic_heads(value, head_count=4, limit=3) == expected


@pytest.mark.parametrize("value", ["-1", "1,1", "4", "1,,2", "one"])
def test_parse_diagnostic_heads_rejects_invalid_or_duplicate_indices(value):
    from afm.config import parse_diagnostic_heads

    with pytest.raises(ValueError, match="diagnostic_head_indices"):
        parse_diagnostic_heads(value, head_count=4, limit=2)


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
