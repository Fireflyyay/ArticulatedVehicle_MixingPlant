#!/usr/bin/env python3
"""Standalone teacher evaluation script for articulated vehicle local parking.

Evaluates teachers across scene families, stages, and seeds. Produces
JSONL results, summary table, and mask compatibility report.

Usage:
    PYTHONPATH=src conda run -n HOPE python scripts/evaluate_teachers.py \
        --teachers lattice multi_anchor \
        --stages 1 \
        --families head_in \
        --seeds-per-family 20 \
        --output-dir runs/teacher_eval \
        --mask-test

Phase: this script is part of Phase 1 and does not interface with PPO/BC/DAgger.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_ENV_CONFIG
from env.local_parking_env import LocalParkingEnv
from env.vehicle import ArticulatedState, ArticulatedVehicleModel
from teachers import (
    ArticulatedLatticeTeacher,
    MultiAnchorTeacher,
    PlanResult,
)
from teachers.heuristics import count_gear_switches

TEACHER_REGISTRY = {
    "lattice": lambda: ArticulatedLatticeTeacher(),
    "multi_anchor": lambda: MultiAnchorTeacher(),
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate teacher planners for local parking")
    p.add_argument("--teachers", nargs="+", default=["lattice"],
                   choices=sorted(TEACHER_REGISTRY.keys()),
                   help="Teachers to evaluate")
    p.add_argument("--stages", nargs="+", type=int, default=[1],
                   help="Curriculum stages (1-4)")
    p.add_argument("--families", nargs="+", default=["head_in"], choices=["head_in"],
                   help="Task families to evaluate")
    p.add_argument("--seeds-per-family", type=int, default=100,
                   help="Number of seeds per (stage, family) pair")
    p.add_argument("--output-dir", default="runs/teacher_eval",
                   help="Directory for output files")
    p.add_argument("--mask-test", action="store_true",
                   help="Run masked replay comparison")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Offset added to per-scene seed")
    p.add_argument("--stage-seed", type=int, default=42,
                   help="Base seed for evaluation")
    p.add_argument("--render-count", type=int, default=0,
                   help="Number of trajectories to render per (teacher, family)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def run_unmasked_replay(
    initial_state: ArticulatedState,
    actions_physical: List[np.ndarray],
    scene,
    vehicle_model: ArticulatedVehicleModel,
    slot,
):
    state = ArticulatedState(
        x_front=initial_state.x_front,
        y_front=initial_state.y_front,
        theta_front=initial_state.theta_front,
        theta_rear=initial_state.theta_rear,
    )
    collision = False
    success = False
    final_overlap = 0.0
    final_heading_err = float("inf")
    final_phi = state.phi
    min_clearance = float("inf")
    step_count = 0

    for action in actions_physical:
        v_cmd = float(action[0])
        phi_dot_cmd = float(action[1])
        state = vehicle_model.step(state, (v_cmd, phi_dot_cmd))
        step_count += 1

        front_box, rear_box = vehicle_model.body_boxes(state)
        from env.geometry import overlap_ratio, wrap_to_pi
        target_front = slot.front_box()
        final_overlap = overlap_ratio(front_box, target_front)
        final_heading_err = abs(wrap_to_pi(state.theta_front - slot.theta_goal))
        final_phi = state.phi

        try:
            fc = front_box.distance(scene.obstacle_union)
            rc = rear_box.distance(scene.obstacle_union)
            min_clearance = min(min_clearance, float(fc), float(rc))
        except Exception:
            pass

        if scene.prepared_obstacles.intersects(front_box) or scene.prepared_obstacles.intersects(rear_box):
            collision = True
            break

        if final_overlap >= 0.80 and final_heading_err <= math.radians(15.0):
            success = True
            break

    return {
        "success": success and not collision,
        "collision": collision,
        "final_overlap": final_overlap,
        "final_heading_error": final_heading_err,
        "final_phi": final_phi,
        "min_clearance": min_clearance,
        "steps_executed": step_count,
    }


def run_masked_replay(
    initial_state: ArticulatedState,
    actions_normalized: List[np.ndarray],
    env: LocalParkingEnv,
):
    state = ArticulatedState(
        x_front=initial_state.x_front,
        y_front=initial_state.y_front,
        theta_front=initial_state.theta_front,
        theta_rear=initial_state.theta_rear,
    )
    env.state = state
    env.step_count = 0
    env.prev_motion_gear = None
    env.prev_gear_in_obs = 0.0
    env._update_sensors_and_mask()

    forced_stops = 0
    r_raw_values = []
    v_teacher_values = []
    v_masked_values = []
    gear_list = []
    phi_dot_bin_list = []
    total_steps = 0
    low_r_raw_steps = 0
    speed_scaled_steps = 0
    collision = False
    success = False
    final_overlap = 0.0
    final_heading_err = float("inf")
    min_clearance = float("inf")

    for action_norm in actions_normalized:
        if collision or success:
            break
        total_steps += 1
        decoded = env.action_mask.decode_safe_speed_and_cost(
            action_norm, env.current_mask, env.state.phi,
            dt=env.vehicle_params.dt, prev_motion_gear=env.prev_motion_gear,
            config=env.config,
        )
        executed = np.array([decoded["v_exec"], decoded["phi_dot_exec"]], dtype=np.float32)
        env.state = env.vehicle_model.step(env.state, executed)
        env.step_count += 1
        env.prev_motion_gear = decoded["prev_motion_gear"]
        env.prev_gear_in_obs = decoded["prev_gear_in_obs"]

        if decoded["forced_stop"]:
            forced_stops += 1

        r_raw = decoded["r_raw"]
        r_raw_values.append(r_raw)
        if r_raw < 0.15:
            low_r_raw_steps += 1

        v_teacher = action_norm[0] * (
            env.vehicle_params.parking_v_forward_max if action_norm[0] >= 0
            else env.vehicle_params.parking_v_reverse_max
        )
        v_teacher_values.append(abs(v_teacher))
        v_masked_values.append(abs(executed[0]))
        if abs(v_teacher) > 1e-6 and abs(executed[0]) / abs(v_teacher) < 0.5:
            speed_scaled_steps += 1

        gear_list.append(decoded["gear"])
        pd = decoded["phi_dot_exec"]
        phi_dot_bins = env.action_mask.phi_dot_bins
        if len(phi_dot_bins) > 0:
            pd_bin = int(np.argmin(np.abs(phi_dot_bins - pd)))
            phi_dot_bin_list.append(pd_bin)

        metrics = env._boxes_and_metrics()
        front_box = metrics["front_box"]
        rear_box = metrics["rear_box"]
        try:
            fc = front_box.distance(env.scene.obstacle_union)
            rc = rear_box.distance(env.scene.obstacle_union)
            min_clearance = min(min_clearance, float(fc), float(rc))
        except Exception:
            pass

        if env._state_collides(env.state):
            collision = True
            break

        final_overlap = metrics["front_overlap"]
        final_heading_err = abs(metrics["heading_error"])

        if final_overlap >= 0.80 and final_heading_err <= math.radians(15.0):
            success = True
            break

    mask_interfered = False
    if not success or collision:
        mask_interfered = True

    gear_phi_dot_hist = defaultdict(list)
    for g, pd_bin in zip(gear_list, phi_dot_bin_list):
        gear_phi_dot_hist[(g, pd_bin)].append(1)

    return {
        "success": success,
        "collision": collision,
        "mask_interfered": mask_interfered,
        "total_steps": total_steps,
        "forced_stops": forced_stops,
        "forced_stop_rate": forced_stops / max(1, total_steps),
        "mean_r_raw": float(np.mean(r_raw_values)) if r_raw_values else 0.0,
        "low_r_raw_fraction": low_r_raw_steps / max(1, total_steps),
        "speed_scaled_fraction": speed_scaled_steps / max(1, total_steps),
        "final_overlap": final_overlap,
        "final_heading_error": final_heading_err,
        "min_clearance": min_clearance,
        "r_raw_values": r_raw_values,
    }


def evaluate_teacher(
    teacher,
    teacher_name: str,
    stages: List[int],
    families: List[str],
    seeds_per_family: int,
    base_seed: int,
    args,
) -> List[dict]:
    results = []
    for stage in stages:
        for family in families:
            for s_idx in range(seeds_per_family):
                seed_val = base_seed + s_idx * 100 + args.seed_offset
                env_config = replace(DEFAULT_ENV_CONFIG, curriculum_stage=stage)
                env_config = replace(
                    env_config,
                    scene_pool_size=1,
                    scene_family_schedule=(family,),
                    use_hybrid_astar=False,
                    rs_potential_enabled=False,
                )
                env = LocalParkingEnv(
                    config=env_config,
                    seed=seed_val,
                )
                try:
                    obs, info = env.reset()
                except Exception as exc:
                    if args.verbose:
                        print(f"  [SKIP] seed={seed_val} stage={stage} family={family}: {exc}")
                    continue

                initial_state = env.state
                scene = env.scene
                slot = env.slot
                vehicle_model = env.vehicle_model

                entry = {
                    "teacher": teacher_name,
                    "stage": stage,
                    "family": family,
                    "seed": int(scene.metadata.get("seed", seed_val)),
                    "scene_seed": seed_val,
                    "initial_x": float(initial_state.x_front),
                    "initial_y": float(initial_state.y_front),
                    "initial_theta": float(initial_state.theta_front),
                    "initial_phi": float(initial_state.phi),
                    "goal_x": float(slot.x_goal),
                    "goal_y": float(slot.y_goal),
                    "goal_theta": float(slot.theta_goal),
                }

                if args.verbose:
                    print(f"  [{teacher_name}] stage={stage} family={family} seed={seed_val}")

                t0 = time.perf_counter()
                try:
                    result = teacher.plan_from_state(initial_state, scene, slot, vehicle_model)
                except Exception as exc:
                    entry.update({
                        "success": False,
                        "fail_reason": f"exception: {exc}",
                        "planning_time_ms": (time.perf_counter() - t0) * 1000.0,
                        "num_steps": 0,
                        "num_gear_switches": 0,
                        "path_length": 0.0,
                    })
                    results.append(entry)
                    continue

                planning_time = result.planning_time_ms
                if planning_time < 1e-3:
                    planning_time = (time.perf_counter() - t0) * 1000.0

                entry.update({
                    "success": result.success,
                    "fail_reason": result.fail_reason,
                    "planning_time_ms": planning_time,
                    "num_steps": result.num_steps,
                    "num_gear_switches": result.num_gear_switches,
                    "num_zero_speed_steps": result.num_zero_speed_steps,
                    "path_length": result.path_length,
                    "total_cost": result.total_cost,
                    "final_position_error": result.final_position_error,
                    "final_heading_error": result.final_heading_error,
                    "final_phi": result.final_phi,
                    "final_overlap": result.final_overlap,
                    "min_clearance": result.min_clearance,
                })

                if result.success and result.actions_physical:
                    unmasked = run_unmasked_replay(
                        initial_state, result.actions_physical,
                        scene, vehicle_model, slot,
                    )
                    entry["unmasked_success"] = unmasked["success"]
                    entry["unmasked_collision"] = unmasked["collision"]
                    entry["unmasked_final_overlap"] = unmasked["final_overlap"]
                    entry["unmasked_min_clearance"] = unmasked["min_clearance"]
                    env_success = unmasked["success"]
                    clean_phi = abs(result.final_phi) <= math.radians(20.0)
                    entry["clean_success"] = unmasked["success"] and clean_phi
                else:
                    entry["unmasked_success"] = False
                    entry["unmasked_collision"] = False
                    entry["clean_success"] = False

                if args.mask_test and result.success and result.actions_normalized:
                    masked = run_masked_replay(
                        initial_state, result.actions_normalized, env,
                    )
                    entry["masked_success"] = masked["success"]
                    entry["masked_collision"] = masked["collision"]
                    entry["masked_final_overlap"] = masked["final_overlap"]
                    entry["masked_min_clearance"] = masked["min_clearance"]
                    entry["mask_forced_stop_rate"] = masked["forced_stop_rate"]
                    entry["mask_mean_r_raw"] = masked["mean_r_raw"]
                    entry["mask_low_r_raw_fraction"] = masked["low_r_raw_fraction"]
                    entry["mask_speed_scaled_fraction"] = masked["speed_scaled_fraction"]
                    entry["mask_interfered"] = masked["mask_interfered"]

                results.append(entry)

    return results


def compute_summary(results: List[dict]) -> dict:
    summary = {}
    grouped = defaultdict(list)
    for r in results:
        key = (r["teacher"], r["stage"], r["family"])
        grouped[key].append(r)

    for key, entries in grouped.items():
        teacher, stage, family = key
        n = len(entries)
        successes = [e for e in entries if e.get("success")]
        sr = len(successes) / max(1, n)
        collisions = sum(1 for e in entries if e.get("unmasked_collision", False))
        timeouts = sum(1 for e in entries if "timeout" in str(e.get("fail_reason", "")).lower())
        planning_times = [e["planning_time_ms"] for e in entries]
        pt_sorted = sorted(planning_times)

        prefix = f"{teacher}_stage{stage}_{family}"
        summary[f"{prefix}_count"] = n
        summary[f"{prefix}_success_rate"] = sr
        summary[f"{prefix}_collision_rate"] = collisions / max(1, n)
        summary[f"{prefix}_timeout_rate"] = timeouts / max(1, n)
        summary[f"{prefix}_planning_p50_ms"] = pt_sorted[int(0.50 * n)] if n > 0 else 0
        summary[f"{prefix}_planning_p90_ms"] = pt_sorted[int(0.90 * n)] if n > 0 else 0
        summary[f"{prefix}_planning_p95_ms"] = pt_sorted[int(0.95 * n)] if n > 0 else 0

        if "unmasked_success" in entries[0]:
            unmasked_sr = sum(1 for e in entries if e.get("unmasked_success")) / max(1, n)
            summary[f"{prefix}_unmasked_success_rate"] = unmasked_sr

        if "clean_success" in entries[0]:
            clean_sr = sum(1 for e in entries if e.get("clean_success")) / max(1, n)
            summary[f"{prefix}_clean_success_rate"] = clean_sr

        if "mask_interfered" in entries[0]:
            masked_n = [e for e in entries if e.get("success")]
            interfered = sum(1 for e in masked_n if e.get("mask_interfered"))
            summary[f"{prefix}_mask_interference_rate"] = interfered / max(1, len(masked_n))
            frs = [e.get("mask_forced_stop_rate", 0) for e in entries if e.get("success")]
            if frs:
                summary[f"{prefix}_mean_forced_stop_rate"] = float(np.mean(frs))

        if successes:
            mean_steps = float(np.mean([e["num_steps"] for e in successes]))
            mean_switches = float(np.mean([e["num_gear_switches"] for e in successes]))
            mean_length = float(np.mean([e["path_length"] for e in successes]))
            mean_phi = float(np.mean([abs(e.get("final_phi", 0)) for e in successes]))
            summary[f"{prefix}_mean_steps"] = mean_steps
            summary[f"{prefix}_mean_gear_switches"] = mean_switches
            summary[f"{prefix}_mean_path_length"] = mean_length
            summary[f"{prefix}_mean_abs_final_phi"] = mean_phi

        fail_reasons = defaultdict(int)
        for e in entries:
            if not e.get("success"):
                fail_reasons[e.get("fail_reason", "unknown")] += 1
        for reason, count in fail_reasons.items():
            summary[f"{prefix}_fail_{reason}"] = count

    return summary


def write_mask_report(results: List[dict], output_dir: str):
    report_path = os.path.join(output_dir, "mask_compat_report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("ACTION MASK COMPATIBILITY REPORT")
    lines.append("=" * 70)
    lines.append("")

    mask_entries = [e for e in results if "mask_interfered" in e]
    if not mask_entries:
        lines.append("No mask test data collected. Run with --mask-test.")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        return

    grouped = defaultdict(list)
    for e in mask_entries:
        key = (e["teacher"], e["stage"], e["family"])
        grouped[key].append(e)

    for key in sorted(grouped.keys()):
        entries = grouped[key]
        teacher, stage, family = key
        lines.append(f"--- {teacher} | stage={stage} | family={family} ---")
        n = len(entries)
        successes = [e for e in entries if e.get("success")]
        n_success = len(successes)

        mask_inter = sum(1 for e in successes if e.get("mask_interfered"))
        lines.append(f"  Successfully planned: {n_success}/{n}")
        lines.append(f"  Mask interfered (success but masked fail): {mask_inter}/{max(1, n_success)}")
        if n_success > 0:
            lines.append(f"  Mask interference rate: {mask_inter / n_success:.1%}")

        frs = [e.get("mask_forced_stop_rate", 0) for e in successes]
        rraws = [e.get("mask_mean_r_raw", 0) for e in successes]
        lowr = [e.get("mask_low_r_raw_fraction", 0) for e in successes]
        speed_scaled = [e.get("mask_speed_scaled_fraction", 0) for e in successes]

        if frs:
            lines.append(f"  Mean forced-stop rate: {np.mean(frs):.3f}")
        if rraws:
            lines.append(f"  Mean r_raw: {np.mean(rraws):.3f}")
        if lowr:
            lines.append(f"  Mean fraction r_raw<0.15: {np.mean(lowr):.3f}")
        if speed_scaled:
            lines.append(f"  Mean fraction speed scaled >2x: {np.mean(speed_scaled):.3f}")
        lines.append("")

    if mask_inter > 0:
        lines.append("WARNING: Mask interference detected. Action mask may be over-conservative.")
        lines.append("Consider investigating gear+phi_dot combinations that are frequently masked.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Mask report: {report_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Evaluating teachers: {args.teachers}")
    print(f"Stages: {args.stages}, Families: {args.families}")
    print(f"Seeds per family: {args.seeds_per_family}")
    print(f"Output: {args.output_dir}")
    print()

    all_results = []

    for teacher_name in args.teachers:
        print(f"\n{'='*50}")
        print(f"Teacher: {teacher_name}")
        print(f"{'='*50}")
        teacher = TEACHER_REGISTRY[teacher_name]()
        results = evaluate_teacher(
            teacher=teacher,
            teacher_name=teacher_name,
            stages=args.stages,
            families=args.families,
            seeds_per_family=args.seeds_per_family,
            base_seed=args.stage_seed,
            args=args,
        )
        all_results.extend(results)

        n_success = sum(1 for r in results if r.get("success"))
        print(f"  Total: {len(results)}, Success: {n_success} ({n_success/max(1,len(results)):.1%})")

    jsonl_path = os.path.join(args.output_dir, "results.jsonl")
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nResults: {jsonl_path} ({len(all_results)} entries)")

    summary = compute_summary(all_results)
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(summary.items()):
            writer.writerow([k, v])
    print(f"Summary: {csv_path}")

    if args.mask_test:
        write_mask_report(all_results, args.output_dir)

    for task_family in args.families:
        for teacher_name in args.teachers:
            fam_results = [
                r for r in all_results
                if r.get("family") == task_family and r.get("teacher") == teacher_name
            ]
            successes = [r for r in fam_results if r.get("success")]
            if successes:
                times = sorted([r["planning_time_ms"] for r in fam_results])
                n = len(times)
                print(f"  {teacher_name} {task_family}: "
                      f"SR={len(successes)/max(1,len(fam_results)):.1%} "
                      f"p50={times[int(0.50*n)]:.0f}ms "
                      f"p90={times[int(0.90*n)]:.0f}ms "
                      f"p95={times[int(0.95*n)]:.0f}ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
