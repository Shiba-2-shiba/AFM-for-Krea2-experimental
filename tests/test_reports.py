import copy
import json

import pytest

from scripts.summarize_krea2_afm import (
    ReportValidationError,
    load_reports,
    profile_qualification_errors,
    qualify_profile,
    summarize_run,
)


def _valid_report():
    comparisons = []
    for index, strength in enumerate((0.05, 0.1, 0.2)):
        comparisons.append(
            {
                "step_index": index,
                "model_eval_index": 0,
                "block_index": 0,
                "branch": 0,
                "dtype": "float16",
                "strength": strength,
                "candidate_implementation": "augmented",
                "reference_implementation": "manual",
                "augmented_precision": "activation",
                "reference_precision": "float32",
                "same_qkv": True,
                "unselected_rows_preserved": True,
                "metrics": {
                    "rel_l2": 0.001,
                    "scaled_max_abs": 0.002,
                    "effect_rel_l2": 0.1,
                    "effect_resolved": True,
                    "finite": True,
                },
            }
        )
    return {
        "run_id": "synthetic",
        "summary": {
            "status": "ok",
            "observed_calls": 3,
            "edited_calls": 3,
            "fallback_calls": 0,
            "nan_events": 0,
            "oom_events": 0,
        },
        "artifacts": {"images": [{"filename": "synthetic.png"}]},
        "comparisons": comparisons,
        "benchmark": {
            "measurement_source": "server",
            "cuda_synchronized": True,
            "diagnostics_enabled": False,
            "warmup_runs_per_implementation": 1,
            "alternating": True,
            "manual": [
                {"generation_seconds": 10.0, "peak_allocated_bytes": 100} for _ in range(5)
            ],
            "augmented": [
                {"generation_seconds": 9.4, "peak_allocated_bytes": 100} for _ in range(5)
            ],
        },
    }


def test_summary_requires_edited_calls_even_when_image_was_saved():
    report = _valid_report()
    report["summary"]["edited_calls"] = 0

    summary = summarize_run(report)

    assert summary["observed_calls"] == 3
    assert summary["edited_calls"] == 0
    assert summary["edit_status"] == "not_edited"
    assert qualify_profile(report) is False


def test_terminal_run_error_prevents_edit_success_and_qualification():
    report = _valid_report()
    report["summary"]["status"] = "error"
    summary = summarize_run(report)
    assert summary["edit_status"] == "run_failed"
    assert qualify_profile(report) is False
    assert any("terminal status" in error for error in profile_qualification_errors(report))


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_compare_pairs_are_unambiguous(problem):
    report = _valid_report()
    if problem == "missing":
        del report["comparisons"][0]["branch"]
    else:
        report["comparisons"][1].update(
            {key: report["comparisons"][0][key]
             for key in ("step_index", "model_eval_index", "block_index", "branch")}
        )

    with pytest.raises(ReportValidationError):
        summarize_run(report)


def test_profile_requires_correctness_and_benchmark():
    assert qualify_profile(_valid_report()) is True

    mutations = []
    for mutate in (
        lambda report: report["summary"].update(fallback_calls=1),
        lambda report: report["comparisons"][0]["metrics"].update(effect_rel_l2=0.3),
        lambda report: report["comparisons"][0].update(same_qkv=False),
        lambda report: report.pop("benchmark"),
    ):
        report = copy.deepcopy(_valid_report())
        mutate(report)
        mutations.append(report)

    for report in mutations:
        assert qualify_profile(report) is False
        assert profile_qualification_errors(report)


def test_profile_rejects_variability_larger_than_measured_difference():
    report = _valid_report()
    report["benchmark"]["augmented"] = [
        {"generation_seconds": value, "peak_allocated_bytes": 100}
        for value in (8.0, 11.0, 8.0, 11.0, 9.4)
    ]
    assert qualify_profile(report) is False
    assert any("variability" in error for error in profile_qualification_errors(report))


def test_fp32_candidate_does_not_qualify_bfloat16_activation_profile():
    report = _valid_report()
    for comparison in report["comparisons"]:
        comparison.update(dtype="bfloat16", augmented_precision="float32")
    assert qualify_profile(report) is False
    assert any("precision" in error for error in profile_qualification_errors(report))


def test_profile_cannot_union_required_strengths_across_activation_dtypes():
    report = _valid_report()
    report["comparisons"][1].update(dtype="bfloat16", augmented_precision="activation")
    report["comparisons"][2].update(dtype="bf16", augmented_precision="activation")

    assert qualify_profile(report) is False
    assert any("mixed activation dtype" in error for error in profile_qualification_errors(report))


@pytest.mark.parametrize(
    "aliases",
    [
        ("float16", "fp16", "FLOAT16"),
        ("bfloat16", "bf16", "BF16"),
    ],
)
def test_profile_treats_activation_dtype_aliases_as_one_dtype(aliases):
    report = _valid_report()
    for comparison, dtype in zip(report["comparisons"], aliases):
        comparison["dtype"] = dtype

    assert qualify_profile(report) is True
    assert not any(
        "mixed activation dtype" in error
        for error in profile_qualification_errors(report)
    )


def test_runtime_jsonl_is_grouped_by_run_and_maps_server_benchmark(tmp_path):
    records = [
        {
            "run_id": "run-a",
            "event": "run_start",
            "config": {"implementation": "manual", "benchmark": True},
        },
        {
            "run_id": "run-a",
            "event": "kernel_comparison",
            "step_index": 1,
            "model_eval_index": 0,
            "block_index": 2,
            "branch": 0,
            "dtype": "float16",
            "strength": 0.05,
            "candidate_implementation": "augmented",
            "reference_implementation": "manual",
            "augmented_precision": "activation",
            "reference_precision": "float32",
            "same_qkv": True,
            "unselected_rows_preserved": True,
            "metrics": {"rel_l2": 0.001, "scaled_max_abs": 0.002,
                        "effect_rel_l2": 0.1, "effect_resolved": True, "finite": True},
        },
        {
            "run_id": "run-a",
            "event": "run_summary",
            "status": "ok",
            "observed_calls": 1,
            "edited_calls": 1,
            "fallback_calls": 0,
            "nan_events": 0,
            "oom_events": 0,
            "wall_seconds": 2.0,
            "peak_allocated_bytes": 100,
            "measurement_source": "server",
            "cuda_synchronized": True,
            "diagnostics_enabled": False,
        },
    ]
    path = tmp_path / "runtime.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    reports = load_reports(path)

    assert len(reports) == 1
    assert reports[0]["run_id"] == "run-a"
    assert reports[0]["summary"]["edited_calls"] == 1
    assert reports[0]["comparisons"][0]["run_id"] == "run-a"
    assert reports[0]["benchmark"]["manual"][0]["generation_seconds"] == 2.0


def test_comparison_keys_are_namespaced_by_run_id():
    report = _valid_report()
    duplicate = copy.deepcopy(report["comparisons"][0])
    report["comparisons"][0]["run_id"] = "first"
    duplicate["run_id"] = "second"
    report["comparisons"].append(duplicate)
    summarize_run(report)


def test_comparison_key_indices_and_branch_must_be_nonnull_integers():
    report = _valid_report()
    report["comparisons"][0]["branch"] = None
    with pytest.raises(ReportValidationError, match="integer"):
        summarize_run(report)


def test_report_schema_rejects_boolean_counts_and_negative_metrics():
    report = _valid_report()
    report["summary"]["observed_calls"] = True
    with pytest.raises(ReportValidationError, match="observed_calls"):
        summarize_run(report)

    report = _valid_report()
    report["comparisons"][0]["metrics"]["rel_l2"] = -0.1
    assert qualify_profile(report) is False
    assert any("non-negative" in error for error in profile_qualification_errors(report))


def test_edited_calls_cannot_exceed_observed_calls():
    report = _valid_report()
    report["summary"].update(observed_calls=1, edited_calls=2)
    with pytest.raises(ReportValidationError, match="exceed"):
        summarize_run(report)
