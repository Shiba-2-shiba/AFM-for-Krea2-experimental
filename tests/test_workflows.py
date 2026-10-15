import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_node import node_module  # noqa: F401 - shared V3 host fixture

from scripts.validate_workflow import validate_workflow

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "tests" / "fixtures" / "legacy_944e507"
LEGACY_SHA256 = {
    "krea2_afm_baseline_api.json": "06207cbf5e81a292e39f7ca32f7d71e47321a75bedbcd307613aa833a33f24be",
    "krea2_afm_discover_api.json": "edcdec624db92a6d8d9b78af241025d18aac29330f1608ad8f2e4c33f7796af4",
    "krea2_afm_edit_api.json": "5d26d62ab958095fd60b071fb1239e7e7f10fb23657772536965a5cfcc2e9d9f",
    "krea2_afm_edit_ui.json": "0be3fdd6a31ca4fbb45d3ee796337de6c0473c63a140b09bb56e28c4ea7263d0",
    "krea2_afm_observe_api.json": "e1699aadfb952083a2bcb34a99e6df0fc0cd5fdc27f6e8744908413a9ea6a279",
}


@pytest.mark.parametrize("mode", ["baseline", "discover", "observe", "edit"])
def test_example_graphs_are_connected(mode):
    graph = json.loads((ROOT / "examples" / f"krea2_afm_{mode}_api.json").read_text())
    assert validate_workflow(graph) == []
    assert graph["3"]["inputs"]["steps"] == 8
    assert graph["3"]["inputs"]["sampler_name"] == "euler"
    assert graph["3"]["inputs"]["scheduler"] == "simple"
    assert graph["13"]["inputs"]["conditioning"] == ["6", 0]
    if mode != "baseline":
        assert graph["20"]["inputs"]["mode"] == mode
        assert graph["3"]["inputs"]["model"] == ["20", 0]


def test_examples_match_v3_node_schema(node_module):
    schema_fields = node_module.define_schema().inputs
    fields = {field.name for field in schema_fields}
    required = {field.name for field in schema_fields if not getattr(field, "optional", False)}
    for mode in ("discover", "observe", "edit"):
        graph = json.loads((ROOT / "examples" / f"krea2_afm_{mode}_api.json").read_text())
        inputs = set(graph["20"]["inputs"])
        assert required <= inputs <= fields


@pytest.mark.parametrize("checkout_newline", [b"\n", b"\r\n"])
def test_legacy_fixture_lf_normalized_bytes_are_pinned_to_944e507(checkout_newline):
    assert {path.name for path in LEGACY.iterdir()} == set(LEGACY_SHA256)
    for name, expected in LEGACY_SHA256.items():
        # Git checkout may convert LF to CRLF. Normalize only that spelling;
        # JSON whitespace, key order and all other content remain byte-pinned.
        contents = (LEGACY / name).read_bytes().replace(b"\r\n", b"\n")
        checkout = contents.replace(b"\n", checkout_newline)
        assert hashlib.sha256(checkout.replace(b"\r\n", b"\n")).hexdigest() == expected


def _patch_object_info(node_module):
    required = {}
    optional = {}
    for field in node_module.define_schema().inputs:
        target = optional if getattr(field, "optional", False) else required
        target[field.name] = ["*"]
    return {
        "Loader": {"input": {"required": {}}, "output": ["MODEL"]},
        "Krea2AFMModelPatch": {
            "input": {"required": required, "optional": optional},
            "output": ["MODEL"],
        },
    }


def test_legacy_api_omits_only_new_optional_inputs(node_module):
    legacy = json.loads((LEGACY / "krea2_afm_edit_api.json").read_text())
    legacy_inputs = dict(legacy["20"]["inputs"])
    legacy_inputs["model"] = ["1", 0]
    graph = {
        "1": {"class_type": "Loader", "inputs": {}},
        "20": {"class_type": "Krea2AFMModelPatch", "inputs": legacy_inputs},
    }
    assert validate_workflow(graph, _patch_object_info(node_module)) == []
    assert "normalization_mode" not in legacy_inputs
    assert "diagnostic_head_indices" not in legacy_inputs
    assert "diagnostic_radial_bins" not in legacy_inputs


def test_schema_validation_still_rejects_missing_required_and_unknown_inputs(node_module):
    legacy = json.loads((LEGACY / "krea2_afm_edit_api.json").read_text())
    legacy_inputs = dict(legacy["20"]["inputs"])
    legacy_inputs["model"] = ["1", 0]
    base = {
        "1": {"class_type": "Loader", "inputs": {}},
        "20": {"class_type": "Krea2AFMModelPatch", "inputs": legacy_inputs},
    }
    missing = copy.deepcopy(base)
    missing["20"]["inputs"].pop("mode")
    assert any("missing required input mode" in error
               for error in validate_workflow(missing, _patch_object_info(node_module)))
    unknown = copy.deepcopy(base)
    unknown["20"]["inputs"]["future_typo"] = True
    assert any("unknown input future_typo" in error
               for error in validate_workflow(unknown, _patch_object_info(node_module)))


def test_legacy_ui_widget_values_remain_in_the_original_order(node_module):
    ui = json.loads((LEGACY / "krea2_afm_edit_ui.json").read_text())
    api = json.loads((LEGACY / "krea2_afm_edit_api.json").read_text())
    patch = next(node for node in ui["nodes"] if node["id"] == 20)
    legacy_widget_names = [name for name in api["20"]["inputs"] if name != "model"]
    current_widget_names = [
        field.name for field in node_module.define_schema().inputs if field.name != "model"
    ]
    assert current_widget_names[:len(legacy_widget_names)] == legacy_widget_names
    assert patch["widgets_values"] == [api["20"]["inputs"][name] for name in legacy_widget_names]


def test_missing_links_and_cycles_rejected():
    assert validate_workflow({"1": {"class_type": "Node", "inputs": {"x": ["2", 0]}}})
    assert validate_workflow({"1": {"class_type": "Node", "inputs": {"x": ["1", 0]}}})


@pytest.fixture
def info():
    return {"Loader": {"input": {"required": {"name": [["known.safetensors"]]}}, "output": ["MODEL"]},
            "Patch": {"input": {"required": {"model": ["MODEL"], "mode": [["observe"]]}}, "output": ["MODEL"]}}


def test_object_info_checks_combo_socket_and_output_index(info):
    graph = {"1": {"class_type": "Loader", "inputs": {"name": "known.safetensors"}},
             "2": {"class_type": "Patch", "inputs": {"model": ["1", 0], "mode": "observe"}}}
    assert validate_workflow(graph, info) == []
    bad = copy.deepcopy(graph)
    bad["1"]["inputs"]["name"] = "not-installed.safetensors"
    assert any("name" in error for error in validate_workflow(bad, info))
    bad = copy.deepcopy(graph)
    bad["2"]["inputs"]["model"] = ["1", 1]
    assert any("output" in error for error in validate_workflow(bad, info))
    bad_info = copy.deepcopy(info)
    bad_info["Loader"]["output"] = ["CLIP"]
    assert any("type" in error for error in validate_workflow(graph, bad_info))


def test_invalid_linked_source_is_reported_instead_of_crashing(info):
    graph = {"1": None, "2": {"class_type": "Patch", "inputs": {"model": ["1", 0], "mode": "observe"}}}
    assert any("invalid API node" in error for error in validate_workflow(graph, info))


def test_edit_ui_links_and_widgets_match_api(node_module):
    ui = json.loads((ROOT / "examples/krea2_afm_edit_ui.json").read_text())
    api = json.loads((ROOT / "examples/krea2_afm_edit_api.json").read_text())
    by_id = {node["id"]: node for node in ui["nodes"]}
    assert {str(key) for key in by_id} == set(api)
    assert ui["version"] == .4
    for link_id, source, slot, target, target_slot, socket_type in ui["links"]:
        input_name = by_id[target]["inputs"][target_slot]["name"]
        assert api[str(target)]["inputs"][input_name] == [str(source), slot]
        assert by_id[target]["inputs"][target_slot]["link"] == link_id
        assert link_id in by_id[source]["outputs"][slot]["links"]
    widget_names = [
        field.name for field in node_module.define_schema().inputs
        if field.name != "model" and field.name in api["20"]["inputs"]
    ]
    assert by_id[20]["widgets_values"] == [api["20"]["inputs"][name] for name in widget_names]
