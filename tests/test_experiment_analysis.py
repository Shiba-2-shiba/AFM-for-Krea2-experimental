import copy
import csv
import hashlib
import json
from pathlib import Path

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
        "case_identity": {
            "positive_text": [prompt_hash], "negative_text": [], "seed": 1,
            "model_components": [{"class_type": "UNETLoader", "identity": {"unet_name": "fixture"}}],
            "sampler": {"sampler_name": "euler", "scheduler": "simple", "steps": 10,
                        "cfg": 1, "denoise": 1},
            "resolution": {"width": 512, "height": 512, "batch_size": 1}, "input_images": [],
        },
        "afm_config": {
            "mode": "observe" if mode == "baseline" else "edit",
            "normalization_mode": "joint" if mode == "baseline" else mode,
            "implementation": "manual", "strength": 1.0, "schedule": "curve",
            "cutoff": 0.25, "soft_mask": True, "mask_width": 0.05, "preserve_dc": True,
            "block_scope": "0", "branch_mode": "positive_only", "start_percent": 0.0,
            "end_percent": 1.0, "zero_strength_mode": "original", "head_chunk_size": 2,
            "query_chunk_size": 128, "max_workspace_mib": 512, "augmented_precision": "activation",
        },
        "prompt_text_sha256": hashlib.sha256(json.dumps(
            {"positive_text": [prompt_hash], "negative_text": []}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "condition_id": condition,
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
                           "selected_batch_rows": [0], "batch_size": 1,
                           "baseline_backend": "fixture.native", "activation_dtype": "torch.float32",
                           "original_scale": 0.125})
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
        "config": {**run["afm_config"],
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


def test_analysis_excludes_bad_run_only_from_contrasts_using_that_run():
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
    assert primary["complete_pairs"] == 1
    assert primary["excluded_pairs"] == {}
    joint_contrast = result["metrics"]["rho_text_raw"]["contrasts"]["preserve_minus_joint"]
    assert joint_contrast["complete_pairs"] == 0
    assert joint_contrast["excluded_pairs"]["coverage_fail"] == 1


def _analyze(runs, reports=None):
    manifest = {"schema_version": 2, "runs": [
        {key: value for key, value in run.items() if key != "_values"} for run in runs
    ]}
    return analyze_experiment(manifest, reports or [_report(run) for run in runs],
                              bootstrap_samples=20)


def test_preserve_baseline_contrast_does_not_require_a_joint_run():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])

    contrasts = _analyze([base, preserve])["metrics"]["rho_text_raw"]["contrasts"]

    assert contrasts["preserve_minus_baseline"]["complete_pairs"] == 1
    assert contrasts["preserve_minus_baseline"]["estimate"] == pytest.approx(-0.2)
    assert contrasts["preserve_minus_joint"]["complete_pairs"] == 0
    assert contrasts["preserve_minus_joint"]["excluded_pairs"] == {
        "missing_or_duplicate_condition": 1}


def test_manual_zero_is_a_distinct_control_reusable_for_each_preserve_strength():
    zero = _run("zero", "pair", "prompt", "preserve_text_mass", "manual_zero", [0.4, 0.4])
    zero["condition_role"] = "manual_zero"
    zero["afm_config"] = {**zero["afm_config"], "mode": "edit", "strength": 0.0, "zero_strength_mode": "manual",
                          "implementation": "manual", "schedule": "curve"}
    runs = [zero]
    for strength in (1.0, 1.5, 2.0):
        run = _run(f"preserve-{strength}", "pair", "prompt", "preserve_text_mass",
                   f"preserve-{strength}", [0.2, 0.2])
        run["afm_config"].update(strength=strength)
        runs.append(run)

    result = _analyze(runs)

    target_strata = [stratum for stratum in result["strata"].values()
                     if "preserve_minus_manual_zero" in stratum["metrics"]["rho_text_raw"]["contrasts"]]
    assert len(target_strata) == 3
    for stratum in target_strata:
        contrasts = stratum["metrics"]["rho_text_raw"]["contrasts"]
        assert contrasts["preserve_minus_manual_zero"]["complete_pairs"] == 1
        assert contrasts["preserve_minus_manual_zero"]["estimate"] == pytest.approx(-0.2)
        assert contrasts["preserve_minus_joint"]["complete_pairs"] == 0


def test_shared_group_label_does_not_override_different_filter_settings():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    joint = _run("joint", "pair", "prompt", "joint", "joint", [0.3, 0.3])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    for run, strength in ((joint, 1.0), (preserve, 2.0)):
        run["comparison_group"] = "same-label"
        run["afm_config"].update(strength=strength, schedule="curve")

    result = _analyze([base, joint, preserve])

    assert len(result["strata"]) == 2
    assert all(stratum["metrics"]["rho_text_raw"]["contrasts"][
        "preserve_minus_joint"]["complete_pairs"] == 0 for stratum in result["strata"].values())


@pytest.mark.parametrize("difference", ["resolution", "metric_definition", "measurement_coordinates"])
def test_contrasts_do_not_mix_incomparable_measurement_settings(difference):
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    if difference == "resolution":
        preserve["case_identity"]["resolution"] = {"width": 768, "height": 512}
    elif difference == "metric_definition":
        preserve["metric_definition"]["diagnostic_top_k"] = 4
    else:
        preserve["expected_coverage"]["semantic_keys"][0]["head_index"] = 1

    result = _analyze([base, preserve])

    assert len(result["strata"]) == 2
    assert all(stratum["metrics"]["rho_text_raw"]["contrasts"][
        "preserve_minus_baseline"]["complete_pairs"] == 0 for stratum in result["strata"].values())


def test_matching_pair_id_does_not_override_different_case_identity():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    base["case_identity"]["seed"] = 1
    preserve["case_identity"]["seed"] = 2

    result = _analyze([base, preserve])

    contrast = result["metrics"]["rho_text_raw"]["contrasts"]["preserve_minus_baseline"]
    assert contrast["complete_pairs"] == 0
    assert contrast["excluded_pairs"] == {"case_identity_mismatch": 1}


@pytest.mark.parametrize("field,value", [
    ("strength", 2.0), ("schedule", "frequency_neutral"), ("preserve_dc", False),
    ("soft_mask", False), ("block_scope", "all"), ("start_percent", 0.5),
    ("implementation", "augmented"), ("head_chunk_size", 4),
])
def test_runtime_config_cannot_silently_override_planned_semantics(field, value):
    run = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    report = _report(run)
    report["config"][field] = value

    coverage = _analyze([run], [report])["coverage"]["preserve"]

    assert coverage["status"] == "fail"
    assert "runtime configuration mismatch: " + field in coverage["reasons"]


@pytest.mark.parametrize("source", ["manifest", "report"])
def test_missing_config_provenance_stays_unverified(source):
    run = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    report = _report(run)
    del (run["afm_config"] if source == "manifest" else report["config"])["strength"]

    coverage = _analyze([run], [report])["coverage"]["preserve"]

    assert coverage["status"] == "unverified"
    assert "missing configuration provenance: strength" in coverage["reasons"]


def test_manual_zero_label_cannot_hide_a_nonzero_edit():
    zero = _run("zero", "pair", "prompt", "joint", "zero", [0.2, 0.2])
    zero["condition_role"] = "manual_zero"

    coverage = _analyze([zero])["coverage"]["zero"]

    assert coverage["status"] == "fail"
    assert "invalid manual_zero control semantics" in coverage["reasons"]


@pytest.mark.parametrize("field,value", [("baseline_backend", "different.native"),
                                        ("activation_dtype", "torch.float16"),
                                        ("original_scale", 0.5)])
def test_contrasts_reject_different_actual_backend_contract(field, value):
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    reports = [_report(base), _report(preserve)]
    for event in reports[1]["attention_events"]:
        event[field] = value

    contrast = _analyze([base, preserve], reports)["metrics"]["rho_text_raw"][
        "contrasts"]["preserve_minus_baseline"]

    assert contrast["complete_pairs"] == 0
    assert contrast["excluded_pairs"] == {"runtime_contract_mismatch": 1}


def test_missing_backend_provenance_is_not_a_valid_pair():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    reports = [_report(base), _report(preserve)]
    del reports[1]["attention_events"][0]["baseline_backend"]

    contrast = _analyze([base, preserve], reports)["metrics"]["rho_text_raw"][
        "contrasts"]["preserve_minus_baseline"]

    assert contrast["complete_pairs"] == 0
    assert contrast["excluded_pairs"] == {"runtime_contract_unverified": 1}


@pytest.mark.parametrize("field", [None, "seed", "model_components", "input_images", "sampler", "resolution"])
def test_missing_case_provenance_cannot_pass_even_when_both_runs_agree(field):
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    for run in (base, preserve):
        if field is None:
            run.pop("case_identity")
        else:
            run["case_identity"].pop(field)

    contrast = _analyze([base, preserve])["metrics"]["rho_text_raw"][
        "contrasts"]["preserve_minus_baseline"]

    assert contrast["complete_pairs"] == 0
    assert contrast["excluded_pairs"] == {"case_identity_unverified": 1}


def test_unrelated_controls_do_not_add_missing_target_pairs():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    extra = _run("unrelated", "other-pair", "other-prompt", "baseline", "baseline", [0.5, 0.5])

    contrasts = _analyze([base, preserve, extra])["metrics"]["rho_text_raw"]["contrasts"]

    assert contrasts["preserve_minus_baseline"]["expected_pairs"] == 1
    assert contrasts["preserve_minus_baseline"]["excluded_pairs"] == {}
    assert contrasts["joint_minus_baseline"]["expected_pairs"] == 0


def test_manual_zero_baseline_contrast_is_reported_once_per_case_stratum():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    zero = _run("zero", "pair", "prompt", "preserve_text_mass", "zero", [0.4, 0.4])
    zero["condition_role"] = "manual_zero"
    zero["afm_config"].update(strength=0, zero_strength_mode="manual")
    runs = [base, zero]
    for strength in (1.0, 1.5, 2.0):
        run = _run(f"preserve-{strength}", "pair", "prompt", "preserve_text_mass",
                   f"preserve-{strength}", [0.2, 0.2])
        run["afm_config"]["strength"] = strength
        runs.append(run)

    result = _analyze(runs)
    controls = [stratum["metrics"]["rho_text_raw"]["contrasts"]["manual_zero_minus_baseline"]
                for stratum in result["strata"].values()
                if "manual_zero_minus_baseline" in stratum["metrics"]["rho_text_raw"]["contrasts"]]

    assert len(controls) == 1
    assert controls[0]["complete_pairs"] == 1


def test_pilot_matrix_prepares_eighty_images_without_submitting(tmp_path):
    from collections import Counter
    from scripts.run_afm_matrix import main

    root = Path(__file__).resolve().parents[1]
    matrix_path = root / "examples" / "validation_matrix_afm_pilot.json"
    assert matrix_path.is_file()
    output = tmp_path / "pilot"

    def forbidden_network(*args, **kwargs):
        pytest.fail("pilot dry-run must not contact ComfyUI")

    assert main(["--comfy-url", "http://127.0.0.1:9", "--matrix", str(matrix_path),
                 "--out-dir", str(output)], opener=forbidden_network) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mode"] == "dry_run"
    assert len(manifest["runs"]) == 80
    assert set(Counter(run["pair_id"] for run in manifest["runs"]).values()) == {5}
    assert len({run["prompt_text_sha256"] for run in manifest["runs"]}) == 4
    assert len({run["case_identity"]["seed"] for run in manifest["runs"]}) == 4
    assert {run["status"] for run in manifest["runs"]} == {"not_submitted"}
    assert {run["actual_implementation"] for run in manifest["runs"]} == {"unverified"}
    for run in manifest["runs"]:
        config = run["afm_config"]
        assert config["block_scope"] == "all"
        assert config["implementation"] == "manual"
        assert config["head_chunk_size"] == 2 and config["query_chunk_size"] == 128
        assert config["diagnostic_head_indices"] == "0,16,32,47"
        assert config["diagnostic_block_scope"] == "0,13,27"
        assert run["case_identity"]["resolution"] == {"width": 512, "height": 512, "batch_size": 1}
        if run["condition_role"] == "baseline":
            assert config["mode"] == "off"
        elif run["condition_role"] == "manual_zero":
            assert config["normalization_mode"] == "preserve_text_mass"
            assert config["strength"] == 0 and config["zero_strength_mode"] == "manual"
        else:
            assert run["condition_role"] == "preserve_text_mass"
            assert config["strength"] in (1.0, 1.5, 2.0)
    analysis = analyze_experiment(manifest, [], bootstrap_samples=10)
    assert all(record["status"] == "unverified" for record in analysis["coverage"].values())
    assert all(contrast["estimate"] is None for stratum in analysis["strata"].values()
               for metric in stratum["metrics"].values() for contrast in metric["contrasts"].values())


def test_optional_neutral_matrix_adds_only_sixteen_matched_strength_two_cases():
    root = Path(__file__).resolve().parents[1] / "examples"
    path = root / "validation_matrix_afm_pilot_neutral.json"
    assert path.is_file()
    neutral = json.loads(path.read_text())["runs"]
    base = json.loads((root / "validation_matrix_afm_pilot.json").read_text())["runs"]
    preserve_two = [run for run in base if run["condition_role"] == "preserve_text_mass"
                    and run["node_inputs"]["20"]["strength"] == 2.0]
    assert len(neutral) == len(preserve_two) == 16
    for treatment, control in zip(preserve_two, neutral):
        assert treatment["seed"] == control["seed"]
        assert treatment["node_inputs"]["6"] == control["node_inputs"]["6"]
        expected = {**treatment["node_inputs"]["20"], "schedule": "frequency_neutral"}
        assert control["node_inputs"]["20"] == expected


def test_pilot_rating_templates_are_blank_blinded_and_have_separate_key():
    root = Path(__file__).resolve().parents[1] / "examples"
    ratings_path = root / "afm_pilot_ratings.csv"
    key_path = root / "afm_pilot_rating_key.csv"
    assert ratings_path.is_file() and key_path.is_file()
    ratings = list(csv.DictReader(ratings_path.open()))
    key = list(csv.DictReader(key_path.open()))
    assert len(ratings) == len(key) == 80
    assert {row["blind_id"] for row in ratings} == {row["blind_id"] for row in key}
    assert len({row["blind_id"] for row in ratings}) == 80
    assert "planned_run_id" not in ratings[0]
    assert all(not value for row in ratings for field, value in row.items() if field != "blind_id")
    matrix = json.loads((root / "validation_matrix_afm_pilot.json").read_text())
    assert {row["planned_run_id"] for row in key} == {run["id"] for run in matrix["runs"]}
    assert all(not row["runtime_run_id"] and not row["image_sha256"] for row in key)


@pytest.mark.parametrize("field,value", [("head_chunk_size", 4), ("block_scope", "all"),
                                        ("preserve_dc", False), ("start_percent", 0.5)])
def test_manual_zero_comparison_rejects_unmatched_scope_or_arithmetic(field, value):
    zero = _run("zero", "pair", "prompt", "preserve_text_mass", "zero", [0.4, 0.4])
    zero["condition_role"] = "manual_zero"
    zero["afm_config"].update(strength=0, zero_strength_mode="manual")
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    preserve["afm_config"][field] = value

    result = _analyze([zero, preserve])
    contrast = next(stratum["metrics"]["rho_text_raw"]["contrasts"]["preserve_minus_manual_zero"]
                    for stratum in result["strata"].values()
                    if "preserve_minus_manual_zero" in stratum["metrics"]["rho_text_raw"]["contrasts"])

    assert contrast["complete_pairs"] == 0
    assert contrast["excluded_pairs"] == {"control_configuration_mismatch": 1}



def test_forged_prompt_hash_cannot_split_one_prompt_into_independent_clusters():
    runs = []
    for pair_id, fake_hash in (("seed-1", "fake-one"), ("seed-2", "fake-two")):
        for mode in ("baseline", "preserve_text_mass"):
            run = _run(f"{pair_id}-{mode}", pair_id, "same real prompt", mode, mode, [0.2, 0.2])
            run["prompt_text_sha256"] = fake_hash
            runs.append(run)

    contrast = _analyze(runs)["metrics"]["rho_text_raw"]["contrasts"]["preserve_minus_baseline"]

    assert contrast["complete_pairs"] == 0
    assert contrast["excluded_pairs"] == {"prompt_identity_mismatch": 2}


@pytest.mark.parametrize("difference", ["seed", "backend", "missing_case", "config"])
def test_baseline_noise_rejects_incomparable_repeats(difference):
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    repeat = _run("repeat", "pair", "prompt", "baseline", "baseline", [0.3, 0.3])
    repeat["repeat_index"] = 1
    if difference == "seed":
        repeat["case_identity"]["seed"] = 2
    elif difference == "missing_case":
        repeat.pop("case_identity")
    elif difference == "config":
        repeat["afm_config"]["head_chunk_size"] = 4
    reports = [_report(base), _report(repeat)]
    if difference == "backend":
        for event in reports[1]["attention_events"]:
            event["baseline_backend"] = "different.native"

    result = _analyze([base, repeat], reports)

    assert all(stratum["baseline_repeat_noise"]["rho_text_raw"]["paired_repeats"] == 0
               for stratum in result["strata"].values())


def test_baseline_noise_reports_verified_like_for_like_repeat():
    base = _run("base", "pair", "prompt", "baseline", "baseline", [0.4, 0.4])
    repeat = _run("repeat", "pair", "prompt", "baseline", "baseline", [0.3, 0.3])
    repeat["repeat_index"] = 1

    result = _analyze([base, repeat])
    noise = next(iter(result["strata"].values()))["baseline_repeat_noise"]["rho_text_raw"]

    assert noise["paired_repeats"] == 1
    assert noise["mean_delta"] == pytest.approx(-0.1)


def test_pilot_declares_manual_zero_primary_and_csv_retains_comparison_scope(tmp_path):
    from scripts.analyze_afm_experiment import main

    zero = _run("zero", "pair", "prompt", "preserve_text_mass", "zero", [0.4, 0.4])
    zero["condition_role"] = "manual_zero"
    zero["afm_config"].update(strength=0, zero_strength_mode="manual")
    preserve = _run("preserve", "pair", "prompt", "preserve_text_mass", "preserve", [0.2, 0.2])
    runs = [{**run, "server_metrics": [_report(run)]} for run in (zero, preserve)]
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({"schema_version": 2, "runs": runs}))
    out_json, out_csv = tmp_path / "analysis.json", tmp_path / "analysis.csv"

    assert main(["--manifest", str(source), "--out-json", str(out_json),
                 "--out-csv", str(out_csv), "--bootstrap-samples", "10"]) == 0

    assert json.loads(out_json.read_text())["primary_contrast"] == "preserve_minus_manual_zero"
    rows = list(csv.DictReader(out_csv.open()))
    assert "comparison_scope" in rows[0]
    assert {row["comparison_scope"] for row in rows if row["contrast"] == "manual_zero_minus_baseline"} == {
        "practical_original_vs_edit_implementation"}


@pytest.mark.parametrize("mode,spectral_diag,zero_mode,strength", [
    ("off", "sampled", "manual", 1.0),
    ("edit", "off", "manual", 1.0),
    ("edit", "sampled", "original", 0.0),
])
def test_runner_does_not_invent_coverage_for_no_diagnostic_runtime(mode, spectral_diag, zero_mode, strength):
    from scripts.run_afm_matrix import _expected_coverage

    graph = json.loads((Path(__file__).resolve().parents[1] / "examples" /
                        "krea2_afm_experiment_api.json").read_text())
    config = {**graph["20"]["inputs"], "mode": mode, "spectral_diag": spectral_diag,
              "zero_strength_mode": zero_mode, "strength": strength}

    coverage = _expected_coverage(graph, config)

    assert coverage["status"] == "unverified"
    assert coverage["protocol"] is None
    assert coverage["semantic_keys"] == []
