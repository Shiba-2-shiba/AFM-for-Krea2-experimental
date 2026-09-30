import copy
import json
from pathlib import Path

import pytest
from test_node import node_module  # noqa: F401 - shared V3 host fixture

from scripts.validate_workflow import validate_workflow

ROOT = Path(__file__).resolve().parents[1]


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
    fields = {field.name for field in node_module.define_schema().inputs}
    for mode in ("discover", "observe", "edit"):
        graph = json.loads((ROOT / "examples" / f"krea2_afm_{mode}_api.json").read_text())
        assert set(graph["20"]["inputs"]) == fields


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
    widget_names = [field.name for field in node_module.define_schema().inputs if field.name != "model"]
    assert by_id[20]["widgets_values"] == [api["20"]["inputs"][name] for name in widget_names]
