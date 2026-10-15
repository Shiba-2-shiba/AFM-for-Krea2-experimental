import copy

import pytest

from scripts.analyze_afm_experiment import analyze_experiment


def _key(step, mode, block=0, head=0):
    return {
        "step_index": step, "block_index": block, "branch_kind": 0,
        "head_index": head, "sample_index": 0,
        "expected_edit": mode != "baseline" and step >= 8,
        "normalization_mode": "joint" if mode == "baseline" else mode,
        "expected_implementation": "original" if mode == "baseline" else "manual",
    }


def _run(run_id, pair_id, prompt_hash, mode, condition, values):
    keys = [_key(8, mode), _key(9, mode)]
    return {
        "id": run_id, "runtime_run_id": run_id, "pair_id": pair_id,
        "status": "completed",
        "prompt_text_sha256": prompt_hash, "condition_id": condition,
        "normalization_mode": "joint" if mode == "baseline" else mode,
        "condition_role": mode, "control": mode, "actual_implementation": (
            "original" if mode == "baseline" else "manual"
        ),
        "expected_coverage": {
            "protocol": "euler_simple_single_eval_per_branch_v1", "semantic_keys": keys,
        },
        "metric_definition": {"metric_version": "text_spectrum_v2", "primary": "rho_text_raw",
                              "diagnostic_top_k": 8, "radial_bins": 32, "cutoff": 0.25},
        "phase": "confirmation", "repeat_index": 0,
        "_values": values,
    }


def _report(run):
    spectral = []
    attentions = []
    for expected, value in zip(run["expected_coverage"]["semantic_keys"], run["_values"]):
        common = {
            "schema_version": 5, "run_id": run["runtime_run_id"],
            "step_index": expected["step_index"], "model_eval_index": 3,
            "block_index": expected["block_index"], "branch_kind": 0,
            "branch": 0, "sample_index": 0, "head_index": expected["head_index"],
            "normalization_mode": run["normalization_mode"], "normalization_version": 1,
            "diagnostic_top_k": 8, "radial_bins": 32, "cutoff": 0.25,
            "selected_implementation": run["actual_implementation"],
            "edit_applied": expected["expected_edit"], "progress": expected["step_index"] / 10,
        }
        attentions.append({**common, "event": "attention", "cond_or_uncond": [0],
                           "selected_batch_rows": [0], "batch_size": 1})
        spectral.append({**common, "event": "spectral_diagnostic", "metric_version": "text_spectrum_v2",
                         "metric_defined": {"raw": True, "centered": True,
                                            "radial_binned": True},
                         "metric_defined_before": {"raw": True, "centered": True,
                                                   "radial_binned": True},
                         "metric_defined_after": {"raw": True, "centered": True,
                                                  "radial_binned": True},
                         "constant_topk": False,
                         "rho_text_raw_before": value + 0.1,
                         "rho_text_raw_after": value,
                         "rho_text_centered_before": 0.5,
                         "rho_text_centered_after": 0.4,
                         "rho_text_radial_binned_before": 0.3,
                         "rho_text_radial_binned_after": 0.2,
                         "direct_delta_rho": -0.1, "finite": True,
                         "text_attention_mass_before": 0.5,
                         "text_attention_mass_after": 0.5,
                         "text_mass_max_abs_delta": 0.0,
                         "image_probability_max_abs_delta": 0.0})
    return {
        "schema_version": 5, "run_id": run["runtime_run_id"],
        "config": {"mode": "observe" if run["condition_role"] == "baseline" else "edit",
                   "normalization_mode": run["normalization_mode"],
                   "normalization_version": 1,
                   "diagnostic_metric_version": "text_spectrum_v2",
                   "diagnostic_top_k": 8, "diagnostic_radial_bins": 32,
                   "cutoff": 0.25},
        "events": attentions + spectral,
        "attention_events": attentions, "spectral_events": spectral,
        "summary": {"status": "ok", "observed_calls": 2,
                    "edited_calls": 0 if run["condition_role"] == "baseline" else 2,
                    "fallback_calls": 0, "nan_events": 0, "oom_events": 0},
        "actual_implementations": [run["actual_implementation"]], "comparisons": [],
    }


def test_analysis_uses_common_complete_pairs_and_prompt_cluster_bootstrap():
    runs = []
    reports = []
    for prompt, seed, baseline, joint, preserve in (
        ("p1", 1, 0.10, 0.09, 0.07), ("p1", 2, 0.20, 0.18, 0.14),
        ("p2", 1, 0.30, 0.27, 0.24),
    ):
        pair = f"{prompt}-{seed}"
        for mode, condition, value in (
            ("baseline", "baseline", baseline), ("joint", "joint", joint),
            ("preserve_text_mass", "preserve", preserve),
        ):
            run = _run(f"{pair}-{condition}", pair, prompt, mode, condition, [value, value])
            runs.append({key: value for key, value in run.items() if key != "_values"})
            reports.append(_report(run))
    manifest = {"schema_version": 2, "runs": runs}

    first = analyze_experiment(manifest, reports, bootstrap_seed=7, bootstrap_samples=100)
    second = analyze_experiment(manifest, reports, bootstrap_seed=7, bootstrap_samples=100)

    primary = first["metrics"]["rho_text_raw"]["contrasts"]["preserve_minus_baseline"]
    assert primary["expected_pairs"] == 3
    assert primary["complete_pairs"] == 3
    assert primary["estimate"] == pytest.approx(-0.0525)
    assert primary["bootstrap_ci95"] == second["metrics"]["rho_text_raw"]["contrasts"]["preserve_minus_baseline"]["bootstrap_ci95"]


def test_analysis_rejects_duplicate_physical_event_and_reports_missing_common_pair():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [1.0, 1.0])
    joint = _run("joint", "pair", "prompt", "joint", "joint", [0.9, 0.9])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.8, 0.8])
    reports = [_report(base), _report(joint), _report(preserve)]
    reports[1]["spectral_events"].append(copy.deepcopy(reports[1]["spectral_events"][0]))
    reports[1]["events"] = reports[1]["attention_events"] + reports[1]["spectral_events"]
    manifest = {"schema_version": 2, "runs": [
        {key: value for key, value in run.items() if key != "_values"}
        for run in (base, joint, preserve)
    ]}

    result = analyze_experiment(manifest, reports, bootstrap_samples=10)

    assert result["coverage"]["joint"]["status"] == "fail"
    assert "duplicate" in " ".join(result["coverage"]["joint"]["reasons"])
    primary = result["metrics"]["rho_text_raw"]["contrasts"]["preserve_minus_baseline"]
    assert primary["complete_pairs"] == 0
    assert primary["excluded_pairs"]["coverage_fail"] == 1
