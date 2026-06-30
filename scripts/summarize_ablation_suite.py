#!/usr/bin/env python
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


RATE_FIELDS = {
    "deterministic_head_in_success",
    "stochastic_head_in_success",
    "stage3_success",
    "stage4_success",
    "stage3_no_latch_success",
    "stage4_recovery_success",
    "collision_rate",
    "timeout_rate",
    "deadlock_rate",
    "forced_stop_rate",
    "mask_invalid_rate",
    "mask_zero_fraction",
    "rs_latched_rate",
    "rs_valid_rate",
    "dwa_trigger_rate",
    "dwa_used_rate",
    "dwa_assisted_success_rate",
}

SUMMARY_FIELDS = [
    "experiment",
    "seed",
    "final_episode",
    "checkpoint_selection_score",
    "deterministic_head_in_success",
    "stochastic_head_in_success",
    "stage3_success",
    "stage4_success",
    "stage3_no_latch_success",
    "stage4_recovery_success",
    "collision_rate",
    "timeout_rate",
    "deadlock_rate",
    "forced_stop_rate",
    "mask_invalid_rate",
    "mask_zero_fraction",
    "mask_safe_ratio_mean",
    "min_lidar_distance",
    "rs_latched_rate",
    "rs_valid_rate",
    "rs_reward_mean",
    "dwa_trigger_rate",
    "dwa_used_rate",
    "dwa_assisted_success_rate",
    "episode_count",
    "status",
]

NUMERIC_FIELDS = [field for field in SUMMARY_FIELDS if field not in ("experiment", "seed", "status")]


def _nan():
    return float("nan")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_float(value):
    if value is None:
        return _nan()
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return _nan()


def _last_jsonl(path):
    last = None
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _jsonl_count(path):
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _nested_get(mapping, path):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first(mapping, paths):
    for path in paths:
        value = _nested_get(mapping, path)
        if value is not None:
            return value
    return None


def _read_status(run_dir, final_eval):
    status_path = os.path.join(run_dir, "status.txt")
    if os.path.exists(status_path):
        with open(status_path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw.startswith("success"):
            return "success"
        if raw.startswith("failed"):
            return "failed"
        if raw.startswith("dry_run"):
            return "incomplete"
    log_path = os.path.join(run_dir, "train.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            tail = handle.read()[-20000:]
        if "Traceback (most recent call last)" in tail:
            return "failed"
    if final_eval is not None and os.path.exists(os.path.join(run_dir, "checkpoint_final.pt")):
        return "success"
    return "incomplete"


def _seed_from_dir(name):
    if name.startswith("seed_"):
        return name[len("seed_") :]
    return name


def _collect_run(run_dir, experiment, seed, warnings):
    final_eval = _last_jsonl(os.path.join(run_dir, "evaluation_metrics.jsonl")) or {}
    last_update = _last_jsonl(os.path.join(run_dir, "training_metrics.jsonl")) or {}
    last_episode = _last_jsonl(os.path.join(run_dir, "episode_metrics.jsonl")) or {}
    episode_count = _jsonl_count(os.path.join(run_dir, "episode_metrics.jsonl"))
    status = _read_status(run_dir, final_eval if final_eval else None)

    det_summary = final_eval.get("deterministic_summary", {})
    row = {
        "experiment": experiment,
        "seed": seed,
        "final_episode": _to_float(
            _first(final_eval, (("episode",),))
            if final_eval
            else last_update.get("episode", last_episode.get("episode"))
        ),
        "checkpoint_selection_score": _to_float(final_eval.get("checkpoint_selection_score")),
        "deterministic_head_in_success": _to_float(
            _first(
                final_eval,
                (
                    ("deterministic", "head_in"),
                    ("no_guide", "deterministic", "head_in"),
                ),
            )
        ),
        "stochastic_head_in_success": _to_float(
            _first(
                final_eval,
                (
                    ("stochastic", "head_in"),
                    ("no_guide", "stochastic", "head_in"),
                ),
            )
        ),
        "stage3_success": _to_float(final_eval.get("stage3_success")),
        "stage4_success": _to_float(final_eval.get("stage4_success")),
        "stage3_no_latch_success": _to_float(final_eval.get("stage3_no_latch_success")),
        "stage4_recovery_success": _to_float(final_eval.get("stage4_recovery_success")),
        "collision_rate": _to_float(final_eval.get("collision_rate", det_summary.get("collision_rate"))),
        "timeout_rate": _to_float(final_eval.get("timeout_rate", det_summary.get("timeout_rate"))),
        "deadlock_rate": _to_float(final_eval.get("deadlock_rate", det_summary.get("deadlock_rate"))),
        "forced_stop_rate": _to_float(last_update.get("forced_stop_rate")),
        "mask_invalid_rate": _to_float(last_update.get("mask_invalid_rate")),
        "mask_zero_fraction": _to_float(last_update.get("mask_zero_fraction")),
        "mask_safe_ratio_mean": _to_float(last_update.get("mask_safe_ratio_mean")),
        "min_lidar_distance": _to_float(last_update.get("min_lidar_distance")),
        "rs_latched_rate": _to_float(
            det_summary.get("rs_latched_rate", last_update.get("rs_latched_rate"))
        ),
        "rs_valid_rate": _to_float(last_update.get("rs_valid_rate")),
        "rs_reward_mean": _to_float(last_update.get("rs_reward_mean")),
        "dwa_trigger_rate": _to_float(
            det_summary.get("dwa_trigger_rate", last_update.get("dwa_trigger_rate"))
        ),
        "dwa_used_rate": _to_float(
            det_summary.get("dwa_used_rate", last_update.get("dwa_used_rate"))
        ),
        "dwa_assisted_success_rate": _to_float(final_eval.get("dwa_assisted_success_rate")),
        "episode_count": float(episode_count),
        "status": status,
    }

    missing = [
        field
        for field in SUMMARY_FIELDS
        if field not in ("experiment", "seed", "status")
        and (not _is_number(row[field]) or math.isnan(float(row[field])))
    ]
    if missing:
        warnings.append(
            "{} seed_{} missing: {}".format(experiment, seed, ", ".join(missing))
        )
    return row


def _write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_std(values):
    clean = [float(value) for value in values if _is_number(value) and not math.isnan(float(value))]
    if not clean:
        return _nan(), _nan(), 0
    mean = sum(clean) / len(clean)
    if len(clean) <= 1:
        return mean, 0.0, len(clean)
    variance = sum((item - mean) ** 2 for item in clean) / (len(clean) - 1)
    return mean, math.sqrt(variance), len(clean)


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    aggregate_rows = []
    for experiment in sorted(grouped):
        items = grouped[experiment]
        record = {"experiment": experiment, "count": len(items)}
        for field in NUMERIC_FIELDS:
            mean, std, count = _mean_std([item[field] for item in items])
            record["{}_mean".format(field)] = mean
            record["{}_std".format(field)] = std
            record["{}_count".format(field)] = count
        aggregate_rows.append(record)
    fields = ["experiment", "count"]
    for field in NUMERIC_FIELDS:
        fields.extend(
            [
                "{}_mean".format(field),
                "{}_std".format(field),
                "{}_count".format(field),
            ]
        )
    return aggregate_rows, fields


def _fmt_mean_std(row, field, percent=False):
    mean = row.get("{}_mean".format(field), _nan())
    std = row.get("{}_std".format(field), _nan())
    if not _is_number(mean) or math.isnan(float(mean)):
        return "NaN"
    scale = 100.0 if percent else 1.0
    return "{:.1f}±{:.1f}".format(float(mean) * scale, float(std) * scale)


def _write_markdown(path, aggregate_rows, warnings):
    header = (
        "| Experiment | Success | Stage3 no-latch | Stage4 recovery | "
        "Collision | Timeout | Forced stop | RS latch | N seeds |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Ablation Summary",
        "",
        header,
        divider,
    ]
    for row in aggregate_rows:
        lines.append(
            "| {experiment} | {success} | {stage3} | {stage4} | {collision} | "
            "{timeout} | {forced_stop} | {rs_latch} | {n} |".format(
                experiment=row["experiment"],
                success=_fmt_mean_std(row, "deterministic_head_in_success", True),
                stage3=_fmt_mean_std(row, "stage3_no_latch_success", True),
                stage4=_fmt_mean_std(row, "stage4_recovery_success", True),
                collision=_fmt_mean_std(row, "collision_rate", True),
                timeout=_fmt_mean_std(row, "timeout_rate", True),
                forced_stop=_fmt_mean_std(row, "forced_stop_rate", True),
                rs_latch=_fmt_mean_std(row, "rs_latched_rate", True),
                n=int(row.get("deterministic_head_in_success_count", 0)),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- full is the main method.",
            "- no_rs_potential removes near-goal RS potential shaping only; it is not a teacher ablation.",
            "- no_mask_observation and no_mask_cost keep execution-layer safe action masking enabled.",
            "- front_lidar_only zeroes only rear LiDAR observations; rear-body collision checking and mask truth remain active.",
            "- dwa_assisted is an execution-time fallback diagnostic.",
            "- unsafe_no_action_mask_execution is an unsafe diagnostic and should not be mixed into fair main-table comparisons.",
        ]
    )
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend("- {}".format(item) for item in warnings)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_best_checkpoints(path, suite_dir, rows):
    lines = [
        "# Best Checkpoints",
        "",
        "| Experiment | Seed | Status | Best checkpoint files |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        run_dir = os.path.join(suite_dir, row["experiment"], "seed_{}".format(row["seed"]))
        files = sorted(os.path.basename(item) for item in glob.glob(os.path.join(run_dir, "checkpoint_best_*.pt")))
        if not files:
            files_text = "none"
        else:
            files_text = "<br>".join(files)
        lines.append(
            "| {} | {} | {} | {} |".format(
                row["experiment"],
                row["seed"],
                row["status"],
                files_text,
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def summarize(suite_dir):
    warnings = []
    rows = []
    for experiment in sorted(os.listdir(suite_dir)):
        exp_dir = os.path.join(suite_dir, experiment)
        if not os.path.isdir(exp_dir):
            continue
        for seed_name in sorted(os.listdir(exp_dir)):
            run_dir = os.path.join(exp_dir, seed_name)
            if not os.path.isdir(run_dir) or not seed_name.startswith("seed_"):
                continue
            seed = _seed_from_dir(seed_name)
            rows.append(_collect_run(run_dir, experiment, seed, warnings))
    rows.sort(key=lambda item: (item["experiment"], str(item["seed"])))

    summary_csv = os.path.join(suite_dir, "summary.csv")
    by_experiment_csv = os.path.join(suite_dir, "summary_by_experiment.csv")
    summary_md = os.path.join(suite_dir, "summary.md")
    best_md = os.path.join(suite_dir, "best_checkpoints.md")
    warnings_path = os.path.join(suite_dir, "summary_warnings.txt")

    _write_csv(summary_csv, rows, SUMMARY_FIELDS)
    aggregate_rows, aggregate_fields = _aggregate(rows)
    _write_csv(by_experiment_csv, aggregate_rows, aggregate_fields)
    _write_markdown(summary_md, aggregate_rows, warnings)
    _write_best_checkpoints(best_md, suite_dir, rows)
    with open(warnings_path, "w", encoding="utf-8") as handle:
        for item in warnings:
            handle.write(item + "\n")
    return {
        "summary_csv": summary_csv,
        "summary_by_experiment_csv": by_experiment_csv,
        "summary_md": summary_md,
        "best_checkpoints_md": best_md,
        "warnings": warnings_path,
        "run_count": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize local parking ablation suite outputs.")
    parser.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    result = summarize(os.path.abspath(args.suite_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
