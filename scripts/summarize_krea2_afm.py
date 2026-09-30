"""Validate and summarize explicit Krea2 AFM reports.

Qualification report schema (version-independent fields used here):

* ``summary``: terminal status, observed_calls, edited_calls, fallback_calls, nan_events,
  oom_events.
* ``comparisons``: unique step_index/model_eval_index/block_index/branch,
  dtype, strength, augmented/manual implementation provenance, same_qkv,
  unselected_rows_preserved, and the five ValidationMetrics fields.
* ``benchmark``: server measurement provenance, CUDA synchronization,
  diagnostics disabled, at least one warm-up per implementation, alternating
  order, and at least five manual and five augmented warm measurements. Each
  measurement contains generation_seconds and peak_allocated_bytes.

The qualifier only returns a decision. It never writes a qualified profile.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


class ReportValidationError(ValueError):
    pass


_KEY_FIELDS = ("step_index", "model_eval_index", "block_index", "branch")
_THRESHOLDS = {
    "float32": (1e-4, 1e-4),
    "fp32": (1e-4, 1e-4),
    "float16": (5e-3, 1e-2),
    "fp16": (5e-3, 1e-2),
    "bfloat16": (2e-2, 5e-2),
    "bf16": (2e-2, 5e-2),
}
_CANONICAL_DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def _validated_comparisons(report: dict) -> list[dict]:
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise ReportValidationError("comparisons must be a list")
    seen = set()
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            raise ReportValidationError(f"comparison {index} must be an object")
        missing = [field for field in _KEY_FIELDS if field not in comparison]
        if missing:
            raise ReportValidationError(
                f"comparison {index} missing key fields: {', '.join(missing)}"
            )
        if any(
            type(comparison[field]) is not int or comparison[field] < 0
            for field in _KEY_FIELDS
        ):
            raise ReportValidationError(
                f"comparison {index} key fields must be nonnull non-negative integers"
            )
        comparison_run_id = comparison.get("run_id")
        if comparison_run_id is not None and (
            not isinstance(comparison_run_id, str) or not comparison_run_id
        ):
            raise ReportValidationError(f"comparison {index} run_id must be a non-empty string")
        key = (comparison_run_id,) + tuple(
            comparison[field] for field in _KEY_FIELDS
        )
        if key in seen:
            raise ReportValidationError(f"duplicate comparison key: {key!r}")
        seen.add(key)
    return comparisons


def summarize_run(report: dict) -> dict:
    _validated_comparisons(report)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ReportValidationError("summary must be an object")
    observed = summary.get("observed_calls")
    edited = summary.get("edited_calls")
    if type(observed) is not int or observed < 0:
        raise ReportValidationError("observed_calls must be a non-negative integer")
    if type(edited) is not int or edited < 0:
        raise ReportValidationError("edited_calls must be a non-negative integer")
    if edited > observed:
        raise ReportValidationError("edited_calls cannot exceed observed_calls")
    status = (
        "run_failed"
        if summary.get("status") != "ok"
        else ("edited" if edited > 0 else "not_edited")
    )
    return {
        "run_id": report.get("run_id", "unknown"),
        "observed_calls": observed,
        "edited_calls": edited,
        "edit_status": status,
        "image_count": len(report.get("artifacts", {}).get("images", [])),
        "qualified": qualify_profile(report),
    }


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _performance_errors(benchmark) -> list[str]:
    if not isinstance(benchmark, dict):
        return ["benchmark evidence is missing"]
    errors = []
    if benchmark.get("measurement_source") != "server":
        errors.append("benchmark metrics are not server measurements")
    if benchmark.get("cuda_synchronized") is not True:
        errors.append("benchmark did not record CUDA synchronization")
    if benchmark.get("diagnostics_enabled") is not False:
        errors.append("benchmark diagnostics were not disabled")
    if benchmark.get("alternating") is not True:
        errors.append("benchmark order was not alternating")
    warmups = benchmark.get("warmup_runs_per_implementation")
    if type(warmups) is not int or warmups < 1:
        errors.append("benchmark warm-up is missing")
    manual = benchmark.get("manual")
    augmented = benchmark.get("augmented")
    if not isinstance(manual, list) or not isinstance(augmented, list):
        return errors + ["manual and augmented warm measurements are required"]
    if len(manual) < 5 or len(augmented) < 5:
        return errors + ["five warm measurements per implementation are required"]
    if not all(isinstance(row, dict) for row in manual + augmented):
        return errors + ["warm measurements must be objects"]

    improvements = []
    for field in ("generation_seconds", "peak_allocated_bytes"):
        manual_values = [row.get(field) for row in manual]
        augmented_values = [row.get(field) for row in augmented]
        if not all(
            _finite_number(value) and value >= 0
            for value in manual_values + augmented_values
        ):
            errors.append(f"{field} measurements must be finite and non-negative")
            continue
        manual_median = statistics.median(manual_values)
        augmented_median = statistics.median(augmented_values)
        difference = manual_median - augmented_median
        if difference < 0:
            errors.append(f"augmented median {field} is worse than manual")
        improvement = difference / manual_median if manual_median > 0 else 0.0
        improvements.append(improvement)
        variability = max(statistics.pstdev(manual_values), statistics.pstdev(augmented_values))
        if variability > max(difference, 0.0):
            errors.append(f"{field} variability is larger than the median difference")
    if improvements and max(improvements) < 0.05:
        errors.append("neither warm median improves by at least 5%")
    return errors


def profile_qualification_errors(report: dict) -> list[str]:
    errors = []
    try:
        comparisons = _validated_comparisons(report)
    except ReportValidationError as exc:
        return [str(exc)]
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return ["summary evidence is missing"]
    if summary.get("status") != "ok":
        errors.append("terminal status is not ok")
    for field in ("observed_calls", "edited_calls", "fallback_calls", "nan_events", "oom_events"):
        value = summary.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"{field} must be a non-negative integer")
    if summary.get("edited_calls", 0) <= 0:
        errors.append("no edited calls were observed")
    for field in ("fallback_calls", "nan_events", "oom_events"):
        if summary.get(field, 0) != 0:
            errors.append(f"{field} must be zero")

    strengths_by_dtype = {}
    for index, comparison in enumerate(comparisons):
        prefix = f"comparison {index}"
        dtype = str(comparison.get("dtype", "")).lower()
        thresholds = _THRESHOLDS.get(dtype)
        if thresholds is None:
            errors.append(f"{prefix} has unsupported dtype")
            continue
        canonical_dtype = _CANONICAL_DTYPES[dtype]
        strengths = strengths_by_dtype.setdefault(canonical_dtype, set())
        if comparison.get("candidate_implementation") != "augmented" or comparison.get(
            "reference_implementation"
        ) != "manual":
            errors.append(f"{prefix} lacks augmented/manual provenance")
        augmented_precision = comparison.get("augmented_precision")
        if comparison.get("reference_precision") != "float32":
            errors.append(f"{prefix} reference precision is not float32")
        if canonical_dtype in ("float16", "bfloat16") and augmented_precision != "activation":
            errors.append(f"{prefix} candidate precision does not match activation dtype")
        if canonical_dtype == "float32" and augmented_precision not in ("activation", "float32"):
            errors.append(f"{prefix} has unsupported candidate precision")
        if comparison.get("same_qkv") is not True:
            errors.append(f"{prefix} does not prove same-QKV comparison")
        if comparison.get("unselected_rows_preserved") is not True:
            errors.append(f"{prefix} does not preserve unselected rows")
        metrics = comparison.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{prefix} metrics are missing")
            continue
        numeric = [metrics.get(name) for name in ("rel_l2", "scaled_max_abs", "effect_rel_l2")]
        if not all(_finite_number(value) for value in numeric) or metrics.get("finite") is not True:
            errors.append(f"{prefix} metrics are not finite")
            continue
        if any(value < 0 for value in numeric):
            errors.append(f"{prefix} metrics must be non-negative")
            continue
        if numeric[0] > thresholds[0] or numeric[1] > thresholds[1]:
            errors.append(f"{prefix} exceeds dtype numerical thresholds")
        if metrics.get("effect_resolved") is not True:
            errors.append(f"{prefix} effect is unresolved")
        if numeric[2] > 0.25:
            errors.append(f"{prefix} effect_rel_l2 exceeds 0.25")
        strength = comparison.get("strength")
        if _finite_number(strength):
            for required in (0.05, 0.1, 0.2):
                if math.isclose(strength, required, rel_tol=0.0, abs_tol=1e-9):
                    strengths.add(required)
    if len(strengths_by_dtype) > 1:
        errors.append(
            f"mixed activation dtype evidence: {sorted(strengths_by_dtype)}"
        )
    elif strengths_by_dtype:
        strengths = next(iter(strengths_by_dtype.values()))
        missing_strengths = sorted({0.05, 0.1, 0.2} - strengths)
        if missing_strengths:
            errors.append(f"missing strength evidence: {missing_strengths}")
    else:
        errors.append("missing strength evidence: [0.05, 0.1, 0.2]")
    errors.extend(_performance_errors(report.get("benchmark")))
    return errors


def qualify_profile(report: dict) -> bool:
    return not profile_qualification_errors(report)


def _reports_from_runtime_records(records: list[dict]) -> list[dict]:
    grouped = {}
    for record in records:
        if not isinstance(record, dict):
            raise ReportValidationError("JSONL records must be objects")
        run_id = record.get("run_id")
        event = record.get("event")
        if not isinstance(run_id, str) or not run_id or not isinstance(event, str):
            raise ReportValidationError("runtime records require run_id and event")
        report = grouped.setdefault(
            run_id, {"run_id": run_id, "comparisons": [], "events": []}
        )
        report["events"].append(record)
        if event == "run_start":
            report["config"] = record.get("config", {})
            report["environment"] = record.get("environment", {})
        elif event == "kernel_comparison":
            comparison = {
                key: value
                for key, value in record.items()
                if key not in ("schema_version", "model_family", "event")
            }
            report["comparisons"].append(comparison)
        elif event == "run_summary":
            report["runtime_status"] = record.get("status")
            report["summary"] = {
                field: record.get(field)
                for field in (
                    "status",
                    "observed_calls",
                    "edited_calls",
                    "fallback_calls",
                    "nan_events",
                    "oom_events",
                )
            }
            config = report.get("config", {})
            implementation = config.get("implementation")
            benchmark = {
                "measurement_source": record.get("measurement_source"),
                "cuda_synchronized": record.get("cuda_synchronized"),
                "diagnostics_enabled": record.get("diagnostics_enabled"),
                "warmup_runs_per_implementation": 0,
                "alternating": False,
                "manual": [],
                "augmented": [],
            }
            if (
                record.get("cuda_synchronized") is True
                and implementation in ("manual", "augmented")
                and _finite_number(record.get("wall_seconds"))
                and _finite_number(record.get("peak_allocated_bytes"))
            ):
                benchmark[implementation].append(
                    {
                        "generation_seconds": record["wall_seconds"],
                        "peak_allocated_bytes": record["peak_allocated_bytes"],
                        "peak_reserved_bytes": record.get("peak_reserved_bytes"),
                        "edit_seconds": record.get("edit_seconds"),
                        "timing_scope": record.get("timing_scope"),
                    }
                )
            report["benchmark"] = benchmark
    return list(grouped.values())


def load_reports(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _reports_from_runtime_records(records)
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else [value]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args(argv)
    summaries = [summarize_run(report) for path in args.reports for report in load_reports(path)]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]) if summaries else [])
            if summaries:
                writer.writeheader()
                writer.writerows(summaries)
    if not args.json_out and not args.csv_out:
        print(json.dumps(summaries, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
