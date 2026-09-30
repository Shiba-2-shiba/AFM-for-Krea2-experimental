"""Validate an API graph, optionally against a saved /object_info response.

This command never contacts a server, downloads models or submits inference.
"""
import argparse
import json
from pathlib import Path


def _is_link(value):
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and type(value[1]) is int)


def validate_workflow(graph: dict, object_info: dict | None = None) -> list[str]:
    errors = []
    if not isinstance(graph, dict) or not graph:
        return ["Expected a nonempty API prompt object"]
    dependencies = {}
    for node_id, node in graph.items():
        if (not isinstance(node, dict) or not isinstance(node.get("class_type"), str)
                or not isinstance(node.get("inputs"), dict)):
            errors.append(f"{node_id}: invalid API node")
            continue
        dependencies[node_id] = []
        schema = object_info.get(node["class_type"]) if object_info is not None else None
        if object_info is not None and schema is None:
            errors.append(f"{node_id}: class_type {node['class_type']} absent from object_info")
        specifications = {}
        if schema is not None:
            inputs = schema.get("input", {})
            specifications = {**inputs.get("required", {}), **inputs.get("optional", {})}
            for key in inputs.get("required", {}):
                if key not in node["inputs"]:
                    errors.append(f"{node_id}: missing required input {key}")
        for name, value in node["inputs"].items():
            if schema is not None and name not in specifications:
                errors.append(f"{node_id}: unknown input {name}")
            kind = specifications[name][0] if name in specifications else None
            if _is_link(value):
                source, slot = value
                dependencies[node_id].append(source)
                if source not in graph or slot < 0:
                    errors.append(f"{node_id}.{name}: invalid link {value}")
                    continue
                if object_info is not None:
                    source_node = graph[source]
                    if not isinstance(source_node, dict):
                        continue  # The source's invalid API node is already reported.
                    source_schema = object_info.get(source_node.get("class_type"), {})
                    outputs = source_schema.get("output", [])
                    if slot >= len(outputs):
                        errors.append(f"{node_id}.{name}: source output index {slot} is unavailable")
                    elif isinstance(kind, str) and kind != "*" and outputs[slot] not in (kind, "*"):
                        errors.append(f"{node_id}.{name}: socket type {outputs[slot]} does not match {kind}")
            elif isinstance(kind, list) and value not in kind:
                errors.append(f"{node_id}.{name}: value {value!r} is not an installed choice")
    visiting, done = set(), set()
    def visit(node_id):
        if node_id in visiting:
            errors.append(f"{node_id}: graph cycle")
            return
        if node_id in done:
            return
        visiting.add(node_id)
        for parent in dependencies.get(node_id, []):
            visit(parent)
        visiting.remove(node_id)
        done.add(node_id)
    for node_id in graph:
        visit(node_id)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--object-info", type=Path, help="Previously saved ComfyUI /object_info JSON")
    args = parser.parse_args()
    graph = json.loads(args.workflow.read_text(encoding="utf-8"))
    info = json.loads(args.object_info.read_text(encoding="utf-8")) if args.object_info else None
    errors = validate_workflow(graph, info)
    print(json.dumps({"valid": not errors, "scope": "object_info" if info is not None else "graph_only",
                      "inference_tested": False, "errors": errors}, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
