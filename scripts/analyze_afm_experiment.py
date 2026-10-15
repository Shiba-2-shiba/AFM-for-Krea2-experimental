"""Offline paired AFM analysis with explicit coverage and prompt-cluster CIs.

This tool never submits prompts or runs a model. It analyzes already-collected
manifest/schema-5 records. Missing evidence remains unverified.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_krea2_afm import load_reports

PROTOCOL = "euler_simple_single_eval_per_branch_v1"
METRIC_VERSION = "text_spectrum_v2"
METRICS = ("rho_text_raw", "rho_text_centered", "rho_text_radial_binned")
DEFINED_KEYS = dict(zip(METRICS, ("raw", "centered", "radial_binned")))
SEMANTIC_FIELDS = ("step_index", "block_index", "branch_kind", "head_index", "sample_index")
PHYSICAL_FIELDS = ("step_index", "model_eval_index", "block_index", "branch", "head_index")
SEMANTIC_CONFIG_FIELDS = (
    "mode", "normalization_mode", "implementation", "strength", "schedule", "cutoff",
    "soft_mask", "mask_width", "preserve_dc", "block_scope", "branch_mode",
    "start_percent", "end_percent", "zero_strength_mode", "head_chunk_size",
    "query_chunk_size", "max_workspace_mib", "augmented_precision",
)
SHARED_EDIT_FIELDS = (
    "cutoff", "soft_mask", "mask_width", "preserve_dc", "block_scope", "branch_mode",
    "start_percent", "end_percent", "head_chunk_size", "query_chunk_size",
    "max_workspace_mib", "augmented_precision",
)
CONTRASTS = {
    "preserve_minus_baseline": ("preserve", "baseline"),
    "joint_minus_baseline": ("joint", "baseline"),
    "preserve_minus_joint": ("preserve", "joint"),
    "preserve_minus_manual_zero": ("preserve", "manual_zero"),
    "manual_zero_minus_baseline": ("manual_zero", "baseline"),
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _index_key(event, fields):
    values = tuple(event.get(field) for field in fields)
    return values if all(type(value) is int and value >= 0 for value in values) else None


def _events(report, event_type):
    field = "spectral_events" if event_type == "spectral_diagnostic" else "attention_events"
    values = report.get(field)
    if isinstance(values, list):
        return values
    return [event for event in report.get("events", [])
            if isinstance(event, dict) and event.get("event") == event_type]


def _role(run):
    label = run.get("condition_role", run.get("role", run.get("control")))
    if label in ("baseline", "observe", "true_baseline"):
        return "baseline"
    if label == "manual_zero":
        return "manual_zero"
    mode = run.get("normalization_mode")
    if mode == "joint":
        return "joint"
    if mode == "preserve_text_mass":
        return "preserve"
    return None


def _config(run, report):
    return {**report.get("config", {}), **run.get("afm_config", {})}


def _expected_implementation(run, report, expected):
    explicit = expected.get("expected_implementation", expected.get("selected_implementation"))
    if explicit is not None and explicit != "auto":
        return explicit
    if not expected["expected_edit"]:
        return "original"
    if run.get("normalization_mode") == "preserve_text_mass":
        return "manual"
    config = _config(run, report)
    requested = config.get("implementation", run.get("requested_implementation"))
    if config.get("strength") == 0:
        requested = config.get("zero_strength_mode", requested)
    if requested in ("manual", "augmented"):
        return requested
    actual = run.get("actual_implementation")
    if actual in ("manual", "augmented"):
        return actual
    paths = set(report.get("actual_implementations", [])) - {"original"}
    return next(iter(paths)) if len(paths) == 1 and paths <= {"manual", "augmented"} else None


def _coverage(run, report):
    failures, unknowns, undefined = [], [], []
    expected_spec = run.get("expected_coverage", {})
    expected_list = expected_spec.get("semantic_keys", [])
    if expected_spec.get("protocol") != PROTOCOL or not expected_list:
        return {"status": "unverified", "reasons": [
            "unsupported or missing expected coverage protocol"]}, {}
    if run.get("status") not in ("completed",):
        unknowns.append("run has no completed execution evidence")
    if report is None:
        return {"status": "unverified", "reasons": unknowns + ["missing runtime report"]}, {}
    if report.get("schema_version") != 5:
        unknowns.append("schema 5 is required for paired spectral analysis")
    config = _config(run, report)
    planned, actual = run.get("afm_config", {}), report.get("config", {})
    for field in SEMANTIC_CONFIG_FIELDS:
        if field not in planned or field not in actual:
            unknowns.append("missing configuration provenance: " + field)
        elif planned[field] != actual[field]:
            failures.append("runtime configuration mismatch: " + field)
    if _role(run) == "manual_zero" and not (
            config.get("mode") == "edit" and config.get("strength") == 0
            and config.get("zero_strength_mode") == "manual"
            and config.get("normalization_mode") in ("joint", "preserve_text_mass")):
        failures.append("invalid manual_zero control semantics")
    mode = run.get("normalization_mode")
    if mode not in ("joint", "preserve_text_mass"):
        unknowns.append("missing normalization mode")
    if config.get("normalization_mode") != mode or config.get("normalization_version") != 1:
        unknowns.append("missing or inconsistent normalization configuration")
    definition = run.get("metric_definition", {})
    if (definition.get("metric_version") != METRIC_VERSION
            or type(definition.get("diagnostic_top_k")) is not int
            or definition.get("diagnostic_top_k", 0) <= 0
            or not _finite(definition.get("cutoff"))
            or not 0 < definition.get("cutoff", 0) < 1
            or type(definition.get("radial_bins")) is not int
            or not 1 <= definition.get("radial_bins", 0) <= 256):
        unknowns.append("incomplete metric definition")
    summary = report.get("summary", {})
    if summary.get("status") != "ok":
        failures.append("runtime terminal status is not ok")
    for field in ("fallback_calls", "nan_events", "oom_events"):
        if type(summary.get(field)) is not int:
            unknowns.append("missing summary " + field)
        elif summary[field] != 0:
            failures.append(field + " is not zero")

    observed_calls, edited_calls = summary.get("observed_calls"), summary.get("edited_calls")
    if (type(observed_calls) is not int or type(edited_calls) is not int
            or min(observed_calls, edited_calls) < 0):
        unknowns.append("missing or invalid observed/edited counters")
    elif edited_calls > observed_calls:
        failures.append("edited calls exceed observed calls")
    elif any(isinstance(key, dict) and key.get("expected_edit") for key in expected_list) and edited_calls == 0:
        failures.append("no edited calls despite expected edits")
    elif _role(run) == "baseline" and edited_calls != 0:
        failures.append("baseline unexpectedly edited")

    expected_map = {}
    for expected in expected_list:
        if not isinstance(expected, dict):
            failures.append("invalid expected key")
            continue
        key = _index_key(expected, SEMANTIC_FIELDS)
        if key is None or type(expected.get("expected_edit")) is not bool:
            failures.append("invalid expected semantic key")
        elif key in expected_map:
            failures.append("duplicate expected semantic key")
        else:
            expected_map[key] = expected

    attention_by_call = defaultdict(list)
    for event in _events(report, "attention"):
        if not isinstance(event, dict):
            failures.append("invalid attention event")
            continue
        key = _index_key(event, ("step_index", "model_eval_index", "block_index"))
        if key is not None:
            attention_by_call[key].append(event)
    physical_seen = set()
    observed = {}
    for event in _events(report, "spectral_diagnostic"):
        if not isinstance(event, dict):
            failures.append("invalid spectral event")
            continue
        physical = _index_key(event, PHYSICAL_FIELDS)
        if physical is None:
            unknowns.append("missing physical evaluation/row/head identity")
            continue
        if physical in physical_seen:
            failures.append("duplicate physical spectral event")
            continue
        physical_seen.add(physical)
        call_key = physical[:3]
        calls = attention_by_call.get(call_key, [])
        if len(calls) != 1:
            (failures if len(calls) > 1 else unknowns).append(
                "duplicate attention call" if len(calls) > 1 else "missing corresponding attention call")
            continue
        attention = calls[0]
        branches, batch = attention.get("cond_or_uncond"), attention.get("batch_size")
        row = event["branch"]
        if (not isinstance(branches, list) or not branches
                or any(type(item) is not int or item not in (0, 1) for item in branches)
                or type(batch) is not int or batch <= 0 or batch % len(branches)
                or row >= batch):
            unknowns.append("cannot map CFG row to branch/sample")
            continue
        per_branch = batch // len(branches)
        branch_kind, sample_index = branches[row // per_branch], row % per_branch
        if event.get("branch_kind") != branch_kind or event.get("sample_index") != sample_index:
            failures.append("CFG semantic/physical identity mismatch")
            continue
        key = (event["step_index"], event["block_index"], branch_kind,
               event["head_index"], sample_index)
        if key not in expected_map:
            failures.append("unexpected spectral semantic key")
            continue
        if key in observed:
            failures.append("duplicate semantic spectral event")
            continue
        observed[key] = event
        expected = expected_map[key]
        implementation = _expected_implementation(run, report, expected)
        if implementation is None:
            unknowns.append("unknown expected implementation")
        for source in (event, attention):
            if source.get("run_id") != run.get("runtime_run_id"):
                failures.append("runtime identity mismatch")
            if source.get("normalization_mode") != mode:
                failures.append("normalization mode mismatch")
            if source.get("normalization_version") != 1:
                unknowns.append("missing normalization version")
            if source.get("edit_applied") is not expected["expected_edit"]:
                failures.append("edit_applied mismatch")
            if implementation is not None and source.get("selected_implementation") != implementation:
                failures.append("selected implementation mismatch")
            if source.get("fallback_original") or source.get("failure_reason"):
                failures.append("attention fallback or failure")
        if row not in attention.get("selected_batch_rows", []):
            failures.append("spectral row is not a selected CFG row")
        if mode == "preserve_text_mass" and expected["expected_edit"]:
            for field in ("text_mass_max_abs_delta", "image_probability_max_abs_delta"):
                delta = event.get(field)
                if not _finite(delta):
                    unknowns.append("missing preservation evidence: " + field)
                elif not 0 <= delta <= 1e-5:
                    failures.append("preservation tolerance exceeded: " + field)
        if event.get("metric_version") != definition.get("metric_version"):
            failures.append("metric version mismatch")
        for field in ("diagnostic_top_k", "radial_bins", "cutoff"):
            if event.get(field) != definition.get(field):
                failures.append("metric definition mismatch: " + field)
        if event.get("finite") is not True:
            unknowns.append("spectral finite evidence missing")
        progress = event.get("progress")
        if not _finite(progress) or not 0 <= progress <= 1:
            unknowns.append("invalid denoising progress")
        value = event.get("rho_text_raw_after")
        if value is not None and (not _finite(value) or not 0 <= value <= 1):
            failures.append("invalid raw HF ratio")
        if (event.get("metric_defined", {}).get("raw") is not True
                or value is None or event.get("constant_topk")):
            undefined.append("primary spectrum undefined or constant_topk")
    if set(expected_map) - set(observed):
        (unknowns if unknowns else failures).append("missing expected semantic keys")
    status = ("fail" if failures else "unverified" if unknowns
              else "unverified_metric" if undefined else "pass")
    return {
        "status": status, "reasons": sorted(set(failures + unknowns + undefined)),
        "expected_keys": len(expected_map), "observed_keys": len(observed),
    }, observed


def _metric_value(observed, metric):
    key = DEFINED_KEYS[metric]
    late = defaultdict(lambda: defaultdict(list))
    for semantic, event in observed.items():
        value = event.get(metric + "_after")
        if (event.get("metric_defined", {}).get(key) is not True
                or event.get("metric_defined_before", {}).get(key) is not True
                or event.get("metric_defined_after", {}).get(key) is not True
                or value is None or not _finite(value) or not 0 <= value <= 1
                or event.get("constant_topk")):
            return None
        if event["progress"] >= 0.8:
            step, block = semantic[:2]
            late[step][block].append(float(value))
    if not late:
        return None
    # Average physical heads, then blocks, then late steps with equal weight.
    step_means = []
    for blocks in late.values():
        block_means = [math.fsum(heads) / len(heads) for heads in blocks.values()]
        step_means.append(math.fsum(block_means) / len(block_means))
    return math.fsum(step_means) / len(step_means)


def _stratum(run):
    case = run.get("case_identity", {})
    return {
        "model_components": case.get("model_components", run.get("model_identity")),
        "sampler": case.get("sampler", run.get("resolved_sampling")),
        "resolution": case.get("resolution", run.get("resolution")),
        "input_mode": "img2img" if case.get("input_images") else "txt2img",
        "metric_definition": run.get("metric_definition"),
        "coverage_protocol": run.get("expected_coverage", {}).get("protocol"),
        "measurement_coordinates": sorted(
            [list(key) for item in run.get("expected_coverage", {}).get("semantic_keys", [])
             if isinstance(item, dict) and (key := _index_key(item, SEMANTIC_FIELDS)) is not None]),
        "phase": run.get("phase", "unspecified"),
    }


def _comparison_group(run, report):
    config = _config(run, report)
    fields = ("strength", "schedule", *SHARED_EDIT_FIELDS)
    group = {field: config.get(field) for field in fields}
    # A caller's label may split groups, but must never erase real differences.
    if run.get("comparison_group") is not None:
        group["label"] = run["comparison_group"]
    return group


def _case_identity_verified(run):
    case = run.get("case_identity")
    if not isinstance(case, dict):
        return False
    if (type(case.get("seed")) is not int or case["seed"] < 0
            or not isinstance(case.get("positive_text"), list) or not case["positive_text"]
            or not all(isinstance(text, str) for text in case["positive_text"])
            or not isinstance(case.get("negative_text"), list)
            or not all(isinstance(text, str) for text in case["negative_text"])):
        return False
    models = case.get("model_components")
    if (not isinstance(models, list) or not models
            or any(not isinstance(model, dict) or not model.get("class_type")
                   or not isinstance(model.get("identity"), dict) or not model["identity"]
                   for model in models)):
        return False
    sampler, resolution = case.get("sampler"), case.get("resolution")
    if (not isinstance(sampler, dict) or not sampler.get("sampler_name")
            or not sampler.get("scheduler") or type(sampler.get("steps")) is not int
            or sampler["steps"] <= 0 or not _finite(sampler.get("cfg"))
            or not _finite(sampler.get("denoise")) or not 0 < sampler["denoise"] <= 1
            or not isinstance(resolution, dict)
            or any(type(resolution.get(field)) is not int or resolution[field] <= 0
                   for field in ("width", "height", "batch_size"))):
        return False
    images = case.get("input_images")
    return (isinstance(images, list)
            and all(isinstance(image, dict) and isinstance(image.get("sha256"), str)
                    and len(image["sha256"]) == 64 for image in images))


def _prompt_identity(run):
    case = run["case_identity"]
    return _digest({field: case[field] for field in ("positive_text", "negative_text")})


def _runtime_contract(item, *, include_implementation):
    """Compare actual call provenance on semantic coordinates, not physical CFG order."""
    calls = {_index_key(event, ("step_index", "model_eval_index", "block_index")): event
             for event in _events(item["report"], "attention") if isinstance(event, dict)}
    contract = []
    for semantic, event in sorted(item["observed"].items()):
        call = calls.get(_index_key(event, ("step_index", "model_eval_index", "block_index")), {})
        backend, dtype, scale = (call.get(field) for field in
                                 ("baseline_backend", "activation_dtype", "original_scale"))
        if (not isinstance(backend, str) or backend in ("", "unknown")
                or not isinstance(dtype, str) or dtype in ("", "unknown")
                or not _finite(scale) or scale <= 0):
            return None
        values = [list(semantic), backend, dtype, scale]
        if include_implementation:
            values.append(event.get("selected_implementation"))
        contract.append(values)
    return contract


def _percentile(values, probability):
    position = probability * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _contrast(pairs, left, right, bootstrap_seed, bootstrap_samples):
    by_prompt = defaultdict(list)
    for pair in pairs:
        by_prompt[pair["prompt"]].append(pair[left] - pair[right])
    means = [math.fsum(by_prompt[prompt]) / len(by_prompt[prompt])
             for prompt in sorted(by_prompt)]
    if not means:
        return {"estimate": None, "bootstrap_ci95": None, "prompt_count": 0,
                "negative_pair_fraction": None}
    estimate = math.fsum(means) / len(means)
    rng = random.Random(bootstrap_seed)
    samples = sorted(math.fsum(rng.choices(means, k=len(means))) / len(means)
                     for _ in range(bootstrap_samples))
    return {
        "estimate": estimate,
        "bootstrap_ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "prompt_count": len(means),
        "negative_pair_fraction": sum(pair[left] < pair[right] for pair in pairs) / len(pairs),
    }



def _baseline_noise(items, metric):
    repeats = defaultdict(dict)
    for item in items:
        run = item["run"]
        if item["role"] != "baseline":
            continue
        repeat = run.get("repeat_index", 0)
        pair = run.get("pair_id")
        if type(repeat) is int and repeat >= 0 and isinstance(pair, str):
            repeats[pair][repeat] = None if repeat in repeats[pair] else item
    differences = []
    for values in repeats.values():
        reference = values.get(0)
        for index, item in values.items():
            if index == 0 or reference is None or item is None:
                continue
            selected = (reference, item)
            if any(entry["coverage"]["status"] != "pass" or entry["values"][metric] is None
                   or not _case_identity_verified(entry["run"]) for entry in selected):
                continue
            if any(entry["run"].get("prompt_text_sha256") != _prompt_identity(entry["run"])
                   for entry in selected):
                continue
            cases = {_canonical(entry["run"]["case_identity"]) for entry in selected}
            configs = {_canonical({field: entry["run"]["afm_config"][field]
                                   for field in SEMANTIC_CONFIG_FIELDS}) for entry in selected}
            contracts = [_runtime_contract(entry, include_implementation=True) for entry in selected]
            if (len(cases) != 1 or len(configs) != 1 or any(value is None for value in contracts)
                    or len({_canonical(value) for value in contracts}) != 1):
                continue
            differences.append(item["values"][metric] - reference["values"][metric])
    mean = math.fsum(differences) / len(differences) if differences else None
    variance = (math.fsum((value - mean) ** 2 for value in differences)
                / (len(differences) - 1)) if len(differences) > 1 else None
    return {
        "paired_repeats": len(differences), "mean_delta": mean,
        "sample_std_delta": math.sqrt(variance) if variance is not None else None,
        "max_abs_delta": max(map(abs, differences)) if differences else None,
        "interpretation": "descriptive baseline variation; not an adoption margin",
    }


def analyze_experiment(manifest: dict, reports: list[dict], *,
                       bootstrap_seed: int = 20261015, bootstrap_samples: int = 10000) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("analysis requires manifest schema_version 2")
    if not isinstance(manifest.get("runs"), list) or not isinstance(reports, list):
        raise ValueError("manifest runs and reports must be lists")
    if type(bootstrap_seed) is not int or type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap seed must be integer and samples must be positive")
    report_index = defaultdict(list)
    for report in reports:
        if isinstance(report, dict) and isinstance(report.get("run_id"), str):
            report_index[report["run_id"]].append(report)
    result = {"schema_version": 1, "coverage": {}, "strata": {}, "metrics": {},
              "bootstrap_seed": bootstrap_seed, "bootstrap_samples": bootstrap_samples,
              "ci_method": "prompt_cluster_percentile_linear",
              "primary_contrast": ("preserve_minus_manual_zero"
                                   if any(isinstance(run, dict) and _role(run) == "manual_zero"
                                          for run in manifest["runs"])
                                   else "preserve_minus_baseline"),
              "adoption_decision": "not_evaluated"}
    grouped = defaultdict(list)
    runtime_owners = Counter(run.get("runtime_run_id") for run in manifest["runs"]
                             if isinstance(run, dict) and run.get("runtime_run_id"))
    run_ids = set()
    for run in manifest["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("id"), str):
            raise ValueError("every manifest run needs an id")
        identifier = run["id"]
        if identifier in run_ids:
            raise ValueError("duplicate manifest run id")
        run_ids.add(identifier)
        matches = report_index.get(run.get("runtime_run_id"), [])
        report = matches[0] if len(matches) == 1 else None
        coverage, observed = _coverage(run, report)
        if len(matches) > 1:
            coverage = {"status": "fail", "reasons": ["duplicate runtime reports"]}
        if runtime_owners.get(run.get("runtime_run_id"), 0) > 1:
            coverage = {"status": "fail", "reasons": ["runtime report reused by multiple runs"]}
        result["coverage"][identifier] = coverage
        role = _role(run)
        if role is None:
            continue
        values = {metric: _metric_value(observed, metric)
                  if coverage["status"] == "pass" else None for metric in METRICS}
        item = {"run": run, "report": report or {}, "role": role,
                "values": values, "coverage": coverage, "observed": observed}
        grouped[_canonical(_stratum(run))].append(item)

    for stratum_json, items in sorted(grouped.items()):
        targets = [item for item in items if item["role"] not in ("baseline", "manual_zero")
                   and item["run"].get("repeat_index", 0) == 0]
        target_contrasts = {name: roles for name, roles in CONTRASTS.items()
                            if name != "manual_zero_minus_baseline"}
        groups = {_canonical(_comparison_group(item["run"], item["report"])): target_contrasts
                  for item in targets}
        control_group = _canonical({"control_contrast": "manual_zero_minus_baseline"})
        if any(item["role"] == "manual_zero" for item in items):
            groups[control_group] = {"manual_zero_minus_baseline": ("manual_zero", "baseline")}
        if not groups:
            groups = {_canonical(None): target_contrasts}
        for group_json in sorted(groups):
            candidates = [item for item in items
                          if item["run"].get("repeat_index", 0) == 0
                          and (item["role"] in ("baseline", "manual_zero")
                               or _canonical(_comparison_group(item["run"], item["report"])) == group_json)]
            by_pair = defaultdict(lambda: defaultdict(list))
            for item in candidates:
                pair_id = item["run"].get("pair_id")
                if isinstance(pair_id, str) and pair_id:
                    by_pair[pair_id][item["role"]].append(item)
            metrics = {}
            for metric in METRICS:
                contrasts = {}
                for name, (left, right) in groups[group_json].items():
                    complete = []
                    excluded = Counter()
                    # Controls cannot invent target cases that the experiment never planned.
                    target_roles = (left, right) if right == "joint" else (left,)
                    expected_pairs = [roles for roles in by_pair.values()
                                      if any(role in roles for role in target_roles)]
                    for roles in expected_pairs:
                        if any(len(roles.get(role, [])) != 1 for role in (left, right)):
                            excluded["missing_or_duplicate_condition"] += 1
                            continue
                        selected = {role: roles[role][0] for role in (left, right)}
                        statuses = {item["coverage"]["status"] for item in selected.values()}
                        if statuses != {"pass"}:
                            reason = "coverage_fail" if "fail" in statuses else "coverage_unverified"
                            excluded[reason] += 1
                            continue
                        if any(item["values"][metric] is None for item in selected.values()):
                            excluded["metric_undefined"] += 1
                            continue
                        if not all(_case_identity_verified(item["run"]) for item in selected.values()):
                            excluded["case_identity_unverified"] += 1
                            continue
                        prompts = {_prompt_identity(item["run"]) for item in selected.values()}
                        if (len(prompts) != 1
                                or any(item["run"].get("prompt_text_sha256") != _prompt_identity(item["run"])
                                       for item in selected.values())):
                            excluded["prompt_identity_mismatch"] += 1
                            continue
                        cases = {_canonical(item["run"].get("case_identity"))
                                 for item in selected.values()}
                        if len(cases) != 1:
                            excluded["case_identity_mismatch"] += 1
                            continue
                        if "manual_zero" in selected and "baseline" not in selected:
                            control_configs = {_canonical({field: item["run"]["afm_config"][field]
                                                           for field in (*SHARED_EDIT_FIELDS, "normalization_mode")})
                                               for item in selected.values()}
                            if len(control_configs) != 1:
                                excluded["control_configuration_mismatch"] += 1
                                continue
                        contracts = [_runtime_contract(item, include_implementation="baseline" not in selected)
                                     for item in selected.values()]
                        if any(contract is None for contract in contracts):
                            excluded["runtime_contract_unverified"] += 1
                            continue
                        if len({_canonical(contract) for contract in contracts}) != 1:
                            excluded["runtime_contract_mismatch"] += 1
                            continue
                        complete.append({"prompt": next(iter(prompts)),
                                         **{role: item["values"][metric]
                                            for role, item in selected.items()}})
                    contrasts[name] = {
                        **_contrast(complete, left, right, bootstrap_seed, bootstrap_samples),
                        "comparison_scope": ("practical_original_vs_edit_implementation" if "baseline" in (left, right)
                                             else "matched_edit_implementation"),
                        "expected_pairs": len(expected_pairs), "complete_pairs": len(complete),
                        "excluded_pairs": dict(excluded),
                        "ci_status": "complete_cases" if not excluded else "exploratory_missing_pairs",
                    }
                metrics[metric] = {"contrasts": contrasts}
            stratum = {"case": json.loads(stratum_json),
                       "comparison_group": json.loads(group_json), "metrics": metrics,
                       "baseline_repeat_noise": {metric: _baseline_noise(items, metric)
                                                 for metric in METRICS}}
            result["strata"][_digest([stratum["case"], stratum["comparison_group"]])] = stratum
    if len(result["strata"]) == 1:
        result["metrics"] = next(iter(result["strata"].values()))["metrics"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reports", nargs="*", type=Path, default=[])
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20261015)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reports = []
    for report_path in args.reports:
        reports.extend(load_reports(report_path))
    if not args.reports:
        for run in manifest.get("runs", []):
            if isinstance(run.get("server_metrics"), list):
                reports.extend(run["server_metrics"])
    analysis = analyze_experiment(manifest, reports, bootstrap_seed=args.bootstrap_seed,
                                  bootstrap_samples=args.bootstrap_samples)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False),
                             encoding="utf-8")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "stratum_id", "metric", "contrast", "comparison_scope", "estimate", "ci_low", "ci_high",
                "expected_pairs", "complete_pairs", "prompt_count", "ci_status"])
            writer.writeheader()
            for stratum_id, stratum in analysis["strata"].items():
                for metric, record in stratum["metrics"].items():
                    for contrast, values in record["contrasts"].items():
                        bounds = values["bootstrap_ci95"] or [None, None]
                        writer.writerow({
                            "stratum_id": stratum_id, "metric": metric, "contrast": contrast,
                            "comparison_scope": values["comparison_scope"],
                            "estimate": values["estimate"], "ci_low": bounds[0], "ci_high": bounds[1],
                            **{key: values[key] for key in
                               ("expected_pairs", "complete_pairs", "prompt_count", "ci_status")},
                        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
