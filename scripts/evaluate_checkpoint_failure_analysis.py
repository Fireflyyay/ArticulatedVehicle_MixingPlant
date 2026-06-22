#!/usr/bin/env python3
"""Evaluate a checkpoint across stages 1-4 with detailed failure analysis.

For each stage, runs deterministic episodes, records full trajectories,
renders failed scenes to outputs/failures/, and prints a summary analysis.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG
from env.local_parking_env import LocalParkingEnv
from env.vehicle import ArticulatedState
from model.continuous_ppo import ContinuousPPOAgent
from train.curriculum import MultiStageScenePool

GEAR_DEADBAND = 0.10
PHI_DOT_MIN_ACTIVE = 0.05


# ---------------------------------------------------------------------------
#  plotting helpers (mirrored from existing render scripts)
# ---------------------------------------------------------------------------
def _plot_polygon(ax, polygon, facecolor, edgecolor, alpha=1.0, linewidth=1.0):
    coords = np.asarray(polygon.exterior.coords)
    ax.fill(
        coords[:, 0], coords[:, 1],
        facecolor=facecolor, edgecolor=edgecolor,
        alpha=alpha, linewidth=linewidth,
    )


def _plot_bay(ax, bay):
    is_target = bool(bay.is_target)
    _plot_polygon(
        ax, bay.polygon,
        "#f6bd60" if is_target else "#9ecae1",
        "#bc6c25" if is_target else "#3182bd",
        alpha=0.35 if is_target else 0.22,
        linewidth=2.2 if is_target else 1.4,
    )
    mouth = np.asarray(bay.mouth_segment, dtype=np.float64)
    ax.plot(
        mouth[:, 0], mouth[:, 1],
        color="#d62828" if is_target else "#3182bd",
        linewidth=3.0 if is_target else 1.5,
        linestyle="--", zorder=5,
    )
    center = np.asarray(bay.polygon.centroid.coords[0], dtype=np.float64)
    direction = np.asarray(
        [np.cos(bay.goal_heading), np.sin(bay.goal_heading)], dtype=np.float64,
    )
    ax.arrow(
        center[0], center[1],
        3.0 * direction[0], 3.0 * direction[1],
        width=0.12, head_width=0.8,
        color="#9b2226" if is_target else "#22577a",
        length_includes_head=True, zorder=6,
    )


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def _eval_config():
    return replace(
        DEFAULT_ENV_CONFIG,
        use_hybrid_astar=False,
        rs_potential_enabled=False,
        enable_hope_teacher=False,
        use_teacher_reward=False,
        enable_offpath_reset=False,
        enable_failure_aggregation=False,
    )


def _failure_type(info):
    if info.get("collision"):
        return "collision"
    if info.get("out_of_bounds"):
        return "out_of_bounds"
    if info.get("articulation_limit_violation"):
        return "articulation"
    if info.get("timeout"):
        return "timeout"
    return "other"


def _failure_label(info):
    label = _failure_type(info)
    if info.get("collision"):
        front_overlap = float(info.get("front_overlap", 0.0))
        dist = float(info.get("distance_to_goal", 0.0))
        phi = float(info.get("phi", 0.0))
        label += " ovlp={:.2f} dist={:.1f}m phi={:.1f}deg".format(
            front_overlap, dist, math.degrees(phi),
        )
    elif info.get("timeout"):
        front_overlap = float(info.get("front_overlap", 0.0))
        heading_err = float(info.get("heading_error_deg", 0.0))
        label += " ovlp={:.2f} head={:.1f}deg".format(front_overlap, heading_err)
    return label


def _state_to_dict(state):
    return {
        "x_front": float(state.x_front),
        "y_front": float(state.y_front),
        "theta_front": float(state.theta_front),
        "theta_rear": float(state.theta_rear),
        "v": float(state.v),
        "phi_dot": float(state.phi_dot),
        "phi": float(state.phi),
    }


def _dict_to_state(d):
    return ArticulatedState(
        x_front=float(d["x_front"]),
        y_front=float(d["y_front"]),
        theta_front=float(d["theta_front"]),
        theta_rear=float(d["theta_rear"]),
        v=float(d.get("v", 0.0)),
        phi_dot=float(d.get("phi_dot", 0.0)),
    )


# ---------------------------------------------------------------------------
#  render a single failure episode
# ---------------------------------------------------------------------------
def render_failure_episode(
    output_path,
    env,
    ep_states,
    final_info,
    reset_info,
    show_lidar=False,
):
    scene = env.scene
    slot = env.slot
    vehicle_model = env.vehicle_model

    fig, ax = plt.subplots(figsize=(10, 10))

    # obstacles
    for obstacle in scene.obstacle_polygons:
        _plot_polygon(ax, obstacle, "#777777", "#555555")

    # parking bays
    for bay in scene.parking_bays:
        _plot_bay(ax, bay)

    # target slot
    target_poly = slot.front_box()
    _plot_polygon(ax, target_poly, "#8fd175", "#207020", alpha=0.55, linewidth=2.0)

    # trajectory (gradient from blue-start to red-end)
    if len(ep_states) > 1:
        xs = [s["x_front"] for s in ep_states]
        ys = [s["y_front"] for s in ep_states]
        n_pts = len(xs)
        for i in range(n_pts - 1):
            frac = i / max(n_pts - 1, 1)
            ax.plot(
                xs[i:i + 2], ys[i:i + 2],
                color=plt.cm.coolwarm(1.0 - frac),
                linewidth=1.0, alpha=0.7,
            )
        ax.scatter(
            xs[-1], ys[-1],
            color="red", s=30, zorder=10, marker="X",
            linewidths=1.0, edgecolors="black",
        )

    # initial vehicle pose (translucent)
    s0 = ep_states[0]
    init_state = _dict_to_state(s0)
    front_box_0, rear_box_0 = vehicle_model.body_boxes(init_state)
    _plot_polygon(ax, front_box_0, "#3a86ff", "#164a91", alpha=0.30, linewidth=1.2)
    _plot_polygon(ax, rear_box_0, "#f4a261", "#9c4f15", alpha=0.30, linewidth=1.2)

    # final vehicle pose (solid)
    s_final = ep_states[-1]
    final_state = _dict_to_state(s_final)
    front_box_f, rear_box_f = vehicle_model.body_boxes(final_state)
    collision = bool(final_info.get("collision", False))
    _plot_polygon(
        ax, front_box_f,
        "#e63946" if collision else "#3a86ff",
        "#9b2226" if collision else "#164a91",
        alpha=0.85, linewidth=2.0,
    )
    _plot_polygon(
        ax, rear_box_f,
        "#e76f51" if collision else "#f4a261",
        "#bc6c25",
        alpha=0.85, linewidth=2.0,
    )

    # optional LiDAR at final state
    if show_lidar and hasattr(env, "last_front_lidar_m") and env.last_front_lidar_m is not None:
        rear_center = vehicle_model.rear_center(final_state)
        front_center = (final_state.x_front, final_state.y_front)
        for center, heading, distances, color in (
            (front_center, final_state.theta_front, env.last_front_lidar_m, "#2b6cb0"),
            (rear_center, final_state.theta_rear, env.last_rear_lidar_m, "#b45309"),
        ):
            for beam_angle, distance in zip(env.lidar.beam_angles, distances):
                angle = heading + beam_angle
                end = (
                    center[0] + distance * math.cos(angle),
                    center[1] + distance * math.sin(angle),
                )
                ax.plot(
                    [center[0], end[0]], [center[1], end[1]],
                    color=color, alpha=0.10, linewidth=0.5,
                )

    # layout
    xmin, ymin, xmax, ymax = scene.world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    metadata = scene.metadata
    title = (
        "stage={} | {} | {} | head_in | variant={}\n"
        "failure: {}"
    ).format(
        metadata.get("stage", "?"),
        reset_info.get("scenario_type", ""),
        metadata.get("approach_side_bucket", ""),
        metadata.get("obstacle_layout_variant", ""),
        _failure_label(final_info),
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_checkpoint_with_failures(
    checkpoint_path,
    episodes_per_family,
    seed,
    device,
    stages,
    output_dir,
    show_lidar=False,
):
    episodes_per_family = int(episodes_per_family)
    if episodes_per_family < 20:
        raise ValueError("episodes_per_family must be at least 20")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_stage = payload.get("extra", {}).get("stage", "?")
    checkpoint_episode = payload.get("extra", {}).get("episode", "?")
    print("=" * 72)
    print("Checkpoint: {}".format(checkpoint_path))
    print("  trained stage:  {}".format(checkpoint_stage))
    print("  trained episode: {}".format(checkpoint_episode))
    print("  episodes per family per stage: {}".format(episodes_per_family))
    print("  stages: {}".format(list(stages)))
    print("  output dir: {}".format(output_dir))
    print("=" * 72)

    agent = ContinuousPPOAgent(device=device)
    agent.network.load_state_dict(payload["network"])
    agent.network.eval()

    mode_config = _eval_config()
    multi_pool = MultiStageScenePool(
        pool_size=mode_config.scene_pool_size,
        base_seed=int(seed),
        scene_config=DEFAULT_SCENE_CONFIG,
        family_schedule=mode_config.scene_family_schedule,
    )
    env = LocalParkingEnv(
        config=mode_config,
        multi_stage_pool=multi_pool,
        seed=int(seed),
    )

    os.makedirs(output_dir, exist_ok=True)

    all_stage_summaries = {}
    all_failed_episodes = []

    for stage in stages:
        print("\n--- Stage {} ---".format(stage))
        env.set_active_stage(stage)

        stage_fail_dir = os.path.join(output_dir, "stage{}".format(stage))
        os.makedirs(stage_fail_dir, exist_ok=True)

        outcomes = []
        families = {"head_in": []}
        failed_episodes = []
        fail_index = 0

        reset_attempts = 0
        while min(len(entries) for entries in families.values()) < episodes_per_family:
            obs, reset_info = env.reset()
            reset_attempts += 1
            family = str(reset_info.get("task_family", "head_in"))
            if family not in families:
                families[family] = []
            if len(families[family]) >= episodes_per_family:
                continue

            done = False
            ep_states = []
            ep_actions_raw = []
            ep_actions_exec = []
            # stop-and-turn counters per episode
            stop_turn_policy = 0
            stop_turn_executed = 0
            stop_turn_masked_out = 0
            total_gear_steps = 0  # steps where gear != STOP_GEAR (policy intended motion)
            # mask-zero escape tracking
            mask_zero_steps = 0
            mask_zero_escapes_st = 0  # escapes via exec stop-turn
            mask_zero_escapes_any = 0  # all escapes from mask-zero
            mask_zero_max_streak = 0
            mask_zero_cur_streak = 0
            prev_mask_all_zero = False
            while not done:
                ep_states.append(_state_to_dict(env.state))
                raw_action, _, _ = agent.act(obs, deterministic=True)
                raw_arr = np.asarray(raw_action, dtype=np.float32)
                ep_actions_raw.append(raw_arr.tolist())
                obs, _reward, terminated, truncated, info = env.step(raw_action)
                exec_a = info.get("executed_action", np.zeros(2, dtype=np.float32))
                if hasattr(exec_a, "tolist"):
                    exec_a = exec_a.tolist()
                elif isinstance(exec_a, (list, tuple)):
                    exec_a = [float(v) for v in exec_a]
                else:
                    exec_a = [float(exec_a[0]), float(exec_a[1])]
                ep_actions_exec.append(exec_a)

                # ---- stop-and-turn detection ----
                v_cmd_norm = float(raw_arr[0])
                phi_cmd_norm = float(raw_arr[1])
                v_exec = float(exec_a[0])
                phi_exec = float(exec_a[1])
                forced_stop = bool(info.get("forced_stop", False))

                is_gear_step = abs(v_cmd_norm) >= GEAR_DEADBAND
                if is_gear_step:
                    total_gear_steps += 1

                # Policy intent: stay still on speed, but turn articulation
                policy_stop_turn = (
                    abs(v_cmd_norm) < GEAR_DEADBAND
                    and abs(phi_cmd_norm) > PHI_DOT_MIN_ACTIVE
                )
                if policy_stop_turn:
                    stop_turn_policy += 1

                # Execution result: actually stopped but turning
                exec_stop_turn = abs(v_exec) < 1e-6 and abs(phi_exec) > 1e-6
                if exec_stop_turn:
                    stop_turn_executed += 1

                # Mask blocked a stop-turn attempt
                if policy_stop_turn and forced_stop:
                    stop_turn_masked_out += 1

                # ---- mask-zero detection ----
                mask_all_zero = bool(info.get("mask_all_zero_before_floor", False))
                if mask_all_zero:
                    mask_zero_steps += 1
                    mask_zero_cur_streak += 1
                    mask_zero_max_streak = max(mask_zero_max_streak, mask_zero_cur_streak)
                else:
                    if prev_mask_all_zero:
                        # escaped from mask-zero this step
                        mask_zero_escapes_any += 1
                        if exec_stop_turn:
                            mask_zero_escapes_st += 1
                    mask_zero_cur_streak = 0
                prev_mask_all_zero = mask_all_zero

                done = terminated or truncated

            # capture final state after episode ends
            ep_states.append(_state_to_dict(env.state))

            final_info = dict(info)
            success = bool(final_info.get("success", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(final_info.get("timeout", False))
            articulation = bool(final_info.get("articulation_limit_violation", False))
            out_of_bounds = bool(final_info.get("out_of_bounds", False))

            ep_data = {
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "articulation": articulation,
                "out_of_bounds": out_of_bounds,
                "scenario": str(reset_info.get("scenario_type", "")),
                "task_family": family,
                "front_overlap": float(final_info.get("front_overlap", 0.0)),
                "heading_error_deg": float(final_info.get("heading_error_deg", 0.0)),
                "distance_to_goal": float(final_info.get("distance_to_goal", 0.0)),
                "episode_steps": len(ep_states) - 1,
                "states": ep_states,
                "actions_raw": ep_actions_raw,
                "actions_exec": ep_actions_exec,
                "final_info": final_info,
                "reset_info": {
                    k: (v if isinstance(v, (int, float, bool, str, type(None)))
                        else str(v))
                    for k, v in reset_info.items()
                },
                "scene_meta": {
                    k: (v if isinstance(v, (int, float, bool, str, type(None)))
                        else str(v))
                    for k, v in env.scene.metadata.items()
                },
                "failure_type": _failure_type(final_info) if not success else "success",
                "stop_turn_policy_count": stop_turn_policy,
                "stop_turn_exec_count": stop_turn_executed,
                "stop_turn_masked_out": stop_turn_masked_out,
                "total_gear_steps": total_gear_steps,
                "mask_zero_steps": mask_zero_steps,
                "mask_zero_escapes_st": mask_zero_escapes_st,
                "mask_zero_escapes_any": mask_zero_escapes_any,
                "mask_zero_max_streak": mask_zero_max_streak,
                "hit_mask_zero": mask_zero_steps > 0,
            }
            outcomes.append(ep_data)
            families[family].append(ep_data)

            if not success:
                fail_type = _failure_type(final_info)
                output_path = os.path.join(
                    stage_fail_dir,
                    "ep{:03d}_{}_{}.png".format(
                        fail_index, fail_type, ep_data["scenario"],
                    ),
                )
                render_failure_episode(
                    output_path, env, ep_states,
                    final_info, reset_info,
                    show_lidar=show_lidar,
                )
                failed_episodes.append(ep_data)
                fail_index += 1

            if len(outcomes) > 0 and len(outcomes) % 50 == 0:
                print(
                    "  stage {}  accepted {}/{}  resets={}  failures={}".format(
                        stage, len(outcomes),
                        episodes_per_family * len(families),
                        reset_attempts,
                        len(failed_episodes),
                    )
                )

        print(
            "  stage {} done: {} eps, {} failures, {} images in {}".format(
                stage, len(outcomes), len(failed_episodes),
                len(failed_episodes), stage_fail_dir,
            )
        )

        # --- compute stage summary ---
        n = len(outcomes)
        if n == 0:
            all_stage_summaries[stage] = {"episodes": 0, "success_rate": 0.0}
            continue

        success_rate = sum(1 for o in outcomes if o["success"]) / n
        collision_rate = sum(1 for o in outcomes if o["collision"]) / n
        timeout_rate = sum(1 for o in outcomes if o["timeout"]) / n
        articulation_rate = sum(1 for o in outcomes if o["articulation"]) / n
        oob_rate = sum(1 for o in outcomes if o["out_of_bounds"]) / n
        avg_overlap = float(np.mean([o["front_overlap"] for o in outcomes]))
        avg_heading = float(np.mean([o["heading_error_deg"] for o in outcomes]))
        avg_distance = float(np.mean([o["distance_to_goal"] for o in outcomes]))
        avg_steps = float(np.mean([o["episode_steps"] for o in outcomes]))

        # scenario breakdown
        scenario_outcomes = defaultdict(list)
        for o in outcomes:
            scenario_outcomes[o["scenario"]].append(o)
        scenario_stats = {}
        for sc, entries in sorted(scenario_outcomes.items()):
            sn = len(entries)
            scenario_stats[sc] = {
                "count": sn,
                "success_rate": sum(1 for e in entries if e["success"]) / sn if sn else 0.0,
                "collision_rate": sum(1 for e in entries if e["collision"]) / sn if sn else 0.0,
                "timeout_rate": sum(1 for e in entries if e["timeout"]) / sn if sn else 0.0,
            }

        # ---------- stop-and-turn analysis ----------
        succ_outs = [o for o in outcomes if o["success"]]
        fail_outs = [o for o in outcomes if not o["success"]]

        def _stop_turn_stats(eps, label):
            if not eps:
                return {"label": label, "count": 0}
            total_steps = sum(e["episode_steps"] for e in eps)
            st_policy = sum(e["stop_turn_policy_count"] for e in eps)
            st_exec = sum(e["stop_turn_exec_count"] for e in eps)
            st_masked = sum(e["stop_turn_masked_out"] for e in eps)
            gear_steps = sum(e["total_gear_steps"] for e in eps)
            return {
                "label": label,
                "count": len(eps),
                "total_steps": total_steps,
                "avg_steps": total_steps / len(eps) if eps else 0.0,
                "stop_turn_policy_total": st_policy,
                "stop_turn_exec_total": st_exec,
                "stop_turn_masked_total": st_masked,
                "stop_turn_policy_ratio": st_policy / total_steps if total_steps else 0.0,
                "stop_turn_exec_ratio": st_exec / total_steps if total_steps else 0.0,
                "stop_turn_masked_ratio": st_masked / max(1, st_policy) if st_policy else 0.0,
                "gear_step_ratio": gear_steps / total_steps if total_steps else 0.0,
            }

        st_all = _stop_turn_stats(outcomes, "all")
        st_succ = _stop_turn_stats(succ_outs, "success")
        st_fail = _stop_turn_stats(fail_outs, "failure")

        # ---------- mask-zero escape analysis ----------
        def _mask_zero_stats(eps, label):
            if not eps:
                return {"label": label, "count": 0}
            hit = [e for e in eps if e.get("hit_mask_zero")]
            total_steps = sum(e["episode_steps"] for e in eps)
            mz_steps = sum(e.get("mask_zero_steps", 0) for e in eps)
            mz_esc_st = sum(e.get("mask_zero_escapes_st", 0) for e in eps)
            mz_esc_any = sum(e.get("mask_zero_escapes_any", 0) for e in eps)
            mz_max_streaks = [e.get("mask_zero_max_streak", 0) for e in hit]
            return {
                "label": label,
                "count": len(eps),
                "hit_count": len(hit),
                "hit_ratio": len(hit) / len(eps) if eps else 0.0,
                "total_steps": total_steps,
                "mask_zero_steps": mz_steps,
                "mask_zero_ratio": mz_steps / total_steps if total_steps else 0.0,
                "escapes_via_stop_turn": mz_esc_st,
                "escapes_total": mz_esc_any,
                "escape_ratio": mz_esc_st / max(1, mz_esc_any) if mz_esc_any else 0.0,
                "avg_max_streak": np.mean(mz_max_streaks) if mz_max_streaks else 0.0,
                "max_streak": max(mz_max_streaks) if mz_max_streaks else 0,
            }

        mz_all = _mask_zero_stats(outcomes, "all")
        mz_succ = _mask_zero_stats(succ_outs, "success")
        mz_fail = _mask_zero_stats(fail_outs, "failure")

        # ---------- failure feature distributions ----------
        fail_type_counts = Counter()
        fail_approach_side = Counter()
        fail_complexity = Counter()
        fail_variant = Counter()
        for ep in failed_episodes:
            fail_type_counts[_failure_type(ep["final_info"])] += 1
            meta = ep.get("scene_meta", {})
            fail_approach_side[meta.get("approach_side_bucket", "?")] += 1
            fail_complexity[meta.get("scene_complexity_bucket", "?")] += 1
            fail_variant[int(meta.get("obstacle_layout_variant", -1))] += 1

        # collision details
        collisions = [ep for ep in failed_episodes if ep.get("collision")]
        timeouts = [ep for ep in failed_episodes if ep.get("timeout") and not ep.get("collision")]

        stage_summary = {
            "stage": stage,
            "episodes": n,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "articulation_rate": articulation_rate,
            "out_of_bounds_rate": oob_rate,
            "avg_front_overlap": avg_overlap,
            "avg_heading_error_deg": avg_heading,
            "avg_distance": avg_distance,
            "avg_steps": avg_steps,
            "scenario_stats": scenario_stats,
            "failure_type_counts": dict(fail_type_counts),
            "failure_approach_side": dict(fail_approach_side),
            "failure_complexity": dict(fail_complexity),
            "failure_variant": dict(fail_variant),
            "total_failures": len(failed_episodes),
            "stop_turn_all": st_all,
            "stop_turn_success": st_succ,
            "stop_turn_failure": st_fail,
            "mask_zero_all": mz_all,
            "mask_zero_success": mz_succ,
            "mask_zero_failure": mz_fail,
        }

        if collisions:
            stage_summary["collision_avg_distance"] = float(np.mean(
                [float(ep["final_info"]["distance_to_goal"]) for ep in collisions]
            ))
            stage_summary["collision_avg_overlap"] = float(np.mean(
                [float(ep["final_info"]["front_overlap"]) for ep in collisions]
            ))
            stage_summary["collision_avg_phi_deg"] = float(math.degrees(np.mean(
                [abs(float(ep["final_info"].get("phi", 0.0))) for ep in collisions]
            )))
            stage_summary["collision_avg_steps"] = float(np.mean(
                [ep["episode_steps"] for ep in collisions]
            ))

        if timeouts:
            stage_summary["timeout_avg_distance"] = float(np.mean(
                [float(ep["final_info"]["distance_to_goal"]) for ep in timeouts]
            ))
            stage_summary["timeout_avg_overlap"] = float(np.mean(
                [float(ep["final_info"]["front_overlap"]) for ep in timeouts]
            ))
            stage_summary["timeout_avg_heading_deg"] = float(np.mean(
                [float(ep["final_info"]["heading_error_deg"]) for ep in timeouts]
            ))

        all_stage_summaries[stage] = stage_summary
        all_failed_episodes.extend(failed_episodes)

    env.close() if hasattr(env, "close") else None

    # --- save summary JSON ---
    summary_path = os.path.join(output_dir, "failure_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_stage_summaries, f, indent=2, default=str)
    print("\nsaved summary to {}".format(summary_path))

    # --- print console report ---
    _print_console_report(all_stage_summaries, all_failed_episodes)

    return all_stage_summaries, all_failed_episodes


# ---------------------------------------------------------------------------
#  console report
# ---------------------------------------------------------------------------
def _print_console_report(stage_summaries, all_failures):
    print("\n" + "=" * 80)
    print("  EVALUATION REPORT")
    print("=" * 80)

    # per-stage table
    header = "{:<2}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>6}  {:>7}"
    print()
    print(header.format(
        "S", "eps", "succ%", "coll%", "tout%", "art%", "oob%",
        "ovlp", "steps", "dist(m)",
    ))
    print("-" * len(header.format(
        "S", "eps", "succ%", "coll%", "tout%", "art%", "oob%",
        "ovlp", "steps", "dist(m)",
    )))
    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        print(header.format(
            stage, s["episodes"],
            "{:.1f}".format(s["success_rate"] * 100),
            "{:.1f}".format(s["collision_rate"] * 100),
            "{:.1f}".format(s["timeout_rate"] * 100),
            "{:.1f}".format(s["articulation_rate"] * 100),
            "{:.1f}".format(s["out_of_bounds_rate"] * 100),
            "{:.3f}".format(s["avg_front_overlap"]),
            "{:.1f}".format(s["avg_steps"]),
            "{:.2f}".format(s["avg_distance"]),
        ))

    # scenario breakdown per stage
    for stage in sorted(stage_summaries.keys()):
        sc = stage_summaries[stage].get("scenario_stats", {})
        if len(sc) > 1:
            print("\nStage {} scenario breakdown:".format(stage))
            print("  {:<35}  {:>5}  {:>7}  {:>7}  {:>7}".format(
                "scenario", "cnt", "succ%", "coll%", "tout%",
            ))
            for name, stats in sorted(sc.items()):
                print("  {:<35}  {:>5}  {:>6.1f}  {:>6.1f}  {:>6.1f}".format(
                    name, stats["count"],
                    stats["success_rate"] * 100,
                    stats["collision_rate"] * 100,
                    stats["timeout_rate"] * 100,
                ))

    # failure mode distribution across stages
    print("\n" + "-" * 50)
    print("  FAILURE MODE BY STAGE")
    fail_type_total = Counter()
    fail_type_by_stage = defaultdict(Counter)
    for stage in sorted(stage_summaries.keys()):
        for ft, count in stage_summaries[stage].get("failure_type_counts", {}).items():
            fail_type_total[ft] += count
            fail_type_by_stage[stage][ft] += count

    stage_labels = sorted(stage_summaries.keys())
    row_fmt = "{:<12}  " + "  ".join("{:>5}" for _ in stage_labels) + "  {:>5}"
    print(row_fmt.format("failure", *stage_labels, "total"))
    sep = row_fmt.format("failure", *stage_labels, "total")
    print("-" * len(sep))
    for ft in ["collision", "timeout", "articulation", "out_of_bounds"]:
        counts = [fail_type_by_stage[s].get(ft, 0) for s in stage_labels]
        if sum(counts) > 0:
            print(row_fmt.format(ft, *counts, sum(counts)))

    # collision analysis
    collisions = [ep for ep in all_failures if ep.get("collision")]
    timeouts = [ep for ep in all_failures if ep.get("timeout") and not ep.get("collision")]

    if collisions:
        print("\n" + "=" * 60)
        print("  COLLISION ANALYSIS ({} events)".format(len(collisions)))
        c_stage = Counter(str(ep.get("scene_meta", {}).get("stage", "?")) for ep in collisions)
        print("  By stage: {}".format(dict(c_stage)))
        c_scenario = Counter(ep.get("scenario", "") for ep in collisions)
        print("  By scenario: {}".format(dict(c_scenario)))
        c_approach = Counter(
            ep.get("scene_meta", {}).get("approach_side_bucket", "") for ep in collisions
        )
        print("  By approach side: {}".format(dict(c_approach)))
        c_complexity = Counter(
            ep.get("scene_meta", {}).get("scene_complexity_bucket", "") for ep in collisions
        )
        print("  By complexity: {}".format(dict(c_complexity)))

        dists = [float(ep["final_info"].get("distance_to_goal", 0.0)) for ep in collisions]
        overlaps = [float(ep["final_info"].get("front_overlap", 0.0)) for ep in collisions]
        phis = [abs(float(ep["final_info"].get("phi", 0.0))) for ep in collisions]
        steps_c = [ep["episode_steps"] for ep in collisions]
        print("  Avg distance at collision: {:.2f}m".format(np.mean(dists) if dists else 0.0))
        print("  Avg overlap at collision: {:.3f}".format(np.mean(overlaps) if overlaps else 0.0))
        print("  Avg |phi| at collision: {:.1f}deg".format(
            math.degrees(np.mean(phis)) if phis else 0.0,
        ))
        print("  Avg steps at collision: {:.1f}".format(np.mean(steps_c) if steps_c else 0.0))

        # distance buckets
        if dists:
            buckets = {"<2m": 0, "2-5m": 0, "5-10m": 0, ">10m": 0}
            for d in dists:
                if d < 2.0:
                    buckets["<2m"] += 1
                elif d < 5.0:
                    buckets["2-5m"] += 1
                elif d < 10.0:
                    buckets["5-10m"] += 1
                else:
                    buckets[">10m"] += 1
            print("  Distance buckets: {}".format(dict(buckets)))

    # timeout analysis
    if timeouts:
        print("\n" + "=" * 60)
        print("  TIMEOUT ANALYSIS ({} events)".format(len(timeouts)))
        t_stage = Counter(str(ep.get("scene_meta", {}).get("stage", "?")) for ep in timeouts)
        print("  By stage: {}".format(dict(t_stage)))
        t_scenario = Counter(ep.get("scenario", "") for ep in timeouts)
        print("  By scenario: {}".format(dict(t_scenario)))
        t_ovl = [float(ep["final_info"].get("front_overlap", 0.0)) for ep in timeouts]
        t_head = [float(ep["final_info"].get("heading_error_deg", 0.0)) for ep in timeouts]
        t_dst = [float(ep["final_info"].get("distance_to_goal", 0.0)) for ep in timeouts]
        print("  Avg overlap at timeout: {:.3f}".format(np.mean(t_ovl) if t_ovl else 0.0))
        print("  Avg heading at timeout: {:.1f}deg".format(np.mean(t_head) if t_head else 0.0))
        print("  Avg distance at timeout: {:.2f}m".format(np.mean(t_dst) if t_dst else 0.0))

        if t_ovl:
            tbuckets = {"<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.8": 0}
            for o in t_ovl:
                if o < 0.3:
                    tbuckets["<0.3"] += 1
                elif o < 0.5:
                    tbuckets["0.3-0.5"] += 1
                elif o < 0.7:
                    tbuckets["0.5-0.7"] += 1
                else:
                    tbuckets["0.7-0.8"] += 1
            print("  Overlap buckets: {}".format(dict(tbuckets)))

    # feature correlations
    print("\n" + "=" * 60)
    print("  FAILURE FEATURE CORRELATIONS")
    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        if s["total_failures"] == 0:
            continue
        print("\n  Stage {} ({} failures / {} eps):".format(
            stage, s["total_failures"], s["episodes"],
        ))
        if s.get("failure_approach_side"):
            print("    Approach side: {}".format(s["failure_approach_side"]))
        if s.get("failure_complexity"):
            print("    Complexity: {}".format(s["failure_complexity"]))
        if s.get("failure_variant"):
            print("    Variant: {}".format(s["failure_variant"]))

    # ---------- stop-and-turn behavior analysis ----------
    _print_stop_turn_analysis(stage_summaries)

    # ---------- mask-zero escape analysis ----------
    _print_mask_zero_analysis(stage_summaries)

    # auto-generated conclusions
    print("\n" + "=" * 60)
    print("  ANALYSIS CONCLUSIONS")
    print("-" * 40)
    _generate_conclusions(stage_summaries, all_failures)


# ---------------------------------------------------------------------------
#  stop-and-turn behaviour analysis
# ---------------------------------------------------------------------------
def _print_stop_turn_analysis(stage_summaries):
    print("\n" + "=" * 60)
    print("  STOP-AND-TURN BEHAVIOR ANALYSIS")
    print("  (policy: v_cmd in deadband <{:.2f}, |phi_cmd| > {:.2f})".format(
        GEAR_DEADBAND, PHI_DOT_MIN_ACTIVE,
    ))
    print("-" * 50)

    row_fmt = "{:<3}  {:>5}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}"
    print()
    print(row_fmt.format(
        "S", "eps", "steps", "st_pol%", "st_exec%", "gear%",
        "s_succ%", "f_pol%", "mask%",
    ))
    print("-" * len(row_fmt.format(
        "S", "eps", "steps", "st_pol%", "st_exec%", "gear%",
        "s_succ%", "f_pol%", "mask%",
    )))
    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        all_st = s.get("stop_turn_all", {})
        succ_st = s.get("stop_turn_success", {})
        fail_st = s.get("stop_turn_failure", {})
        if not all_st.get("count"):
            continue
        print(row_fmt.format(
            stage,
            all_st["count"],
            int(all_st.get("avg_steps", 0)),
            "{:.1f}".format(all_st.get("stop_turn_policy_ratio", 0.0) * 100),
            "{:.1f}".format(all_st.get("stop_turn_exec_ratio", 0.0) * 100),
            "{:.1f}".format(all_st.get("gear_step_ratio", 0.0) * 100),
            "{:.1f}".format(succ_st.get("stop_turn_policy_ratio", 0.0) * 100),
            "{:.1f}".format(fail_st.get("stop_turn_policy_ratio", 0.0) * 100),
            "{:.1f}".format(all_st.get("stop_turn_masked_ratio", 0.0) * 100) if all_st.get("stop_turn_policy_total") else "0.0",
        ))

    # per-stage comparison detail
    print()
    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        all_st = s.get("stop_turn_all", {})
        succ_st = s.get("stop_turn_success", {})
        fail_st = s.get("stop_turn_failure", {})
        if not all_st.get("count"):
            continue
        st_pol_all = all_st.get("stop_turn_policy_total", 0)
        st_exec_all = all_st.get("stop_turn_exec_total", 0)
        st_mask_all = all_st.get("stop_turn_masked_total", 0)
        print(
            "  Stage {}: policy intent {} stop-turn steps, {} executed, {} masked-out "
            "({:.1f}% of intents blocked by mask)".format(
                stage, st_pol_all, st_exec_all, st_mask_all,
                st_mask_all / max(1, st_pol_all) * 100 if st_pol_all else 0.0,
            )
        )
        print(
            "    Success eps: {:.1f}% of steps were stop-turn policy intent".format(
                succ_st.get("stop_turn_policy_ratio", 0.0) * 100,
            )
        )
        print(
            "    Failure eps: {:.1f}% of steps were stop-turn policy intent".format(
                fail_st.get("stop_turn_policy_ratio", 0.0) * 100,
            )
        )

    # judgment
    print("\n  Judgment:")
    stages_sorted = sorted(stage_summaries.keys())
    has_learned = False
    for stage in stages_sorted:
        s = stage_summaries[stage]
        all_st = s.get("stop_turn_all", {})
        succ_st = s.get("stop_turn_success", {})
        fail_st = s.get("stop_turn_failure", {})
        if not all_st.get("count"):
            continue
        sp = all_st.get("stop_turn_policy_ratio", 0.0)
        if sp > 0.02:
            has_learned = True
        s_sp = succ_st.get("stop_turn_policy_ratio", 0.0)
        f_sp = fail_st.get("stop_turn_policy_ratio", 0.0)
        if f_sp > s_sp * 1.5 and fail_st.get("count", 0) >= 3:
            print(
                "  S{}: FAILURE eps show {:.1f}% stop-turn vs {:.1f}% in SUCCESS"
                " — policy over-uses stop-turn when stuck.".format(
                    stage, f_sp * 100, s_sp * 100,
                )
            )

    if has_learned:
        print(
            "  Policy HAS learned proactive stop-and-turn: {:.1f}% of overall steps"
            " show zero-speed + active articulation intent.".format(
                max(s.get("stop_turn_all", {}).get("stop_turn_policy_ratio", 0.0)
                    for s in stage_summaries.values()) * 100,
            )
        )
    else:
        print(
            "  Policy has NOT learned proactive stop-and-turn (<2% stop-turn steps)."
            " Vehicle relies on simultaneous speed+steering rather than in-place turning."
        )


# ---------------------------------------------------------------------------
#  mask-zero escape analysis
# ---------------------------------------------------------------------------
def _print_mask_zero_analysis(stage_summaries):
    print("\n" + "=" * 60)
    print("  MASK-ZERO ESCAPE ANALYSIS")
    print("  (Can articulation changes escape mask-all-zero deadlock?)")
    print("-" * 50)

    row_fmt = "{:<3}  {:>5}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}"
    print()
    print(row_fmt.format(
        "S", "eps", "mz_hit", "mz_hit%", "mz_st%", "esc_st", "esc_any",
        "esc_%st", "streak",
    ))
    print("-" * len(row_fmt.format(
        "S", "eps", "mz_hit", "mz_hit%", "mz_st%", "esc_st", "esc_any",
        "esc_%st", "streak",
    )))

    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        mz = s.get("mask_zero_all", {})
        if not mz.get("count"):
            continue
        print(row_fmt.format(
            stage,
            mz["count"],
            mz.get("hit_count", 0),
            "{:.1f}".format(mz.get("hit_ratio", 0.0) * 100),
            "{:.1f}".format(mz.get("mask_zero_ratio", 0.0) * 100),
            mz.get("escapes_via_stop_turn", 0),
            mz.get("escapes_total", 0),
            "{:.1f}".format(mz.get("escape_ratio", 0.0) * 100) if mz.get("escapes_total") else "0.0",
            "{:.0f}".format(mz.get("max_streak", 0)),
        ))

    # per-stage detail
    print()
    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        mz_all = s.get("mask_zero_all", {})
        mz_succ = s.get("mask_zero_success", {})
        mz_fail = s.get("mask_zero_failure", {})
        if not mz_all.get("count"):
            continue
        hit_all = mz_all.get("hit_count", 0)
        esc_st = mz_all.get("escapes_via_stop_turn", 0)
        esc_any = mz_all.get("escapes_total", 0)
        if hit_all == 0:
            print("  Stage {}: ZERO episodes hit mask-all-zero.".format(stage))
            continue
        print(
            "  Stage {}: {}/{} eps ({:.0f}%) hit mask-zero, {} escapes ({} via stop-turn, {:.0f}%"
            " of escapes)".format(
                stage, hit_all, mz_all["count"],
                mz_all["hit_ratio"] * 100,
                esc_any, esc_st,
                esc_st / max(1, esc_any) * 100 if esc_any else 0.0,
            )
        )
        print(
            "    Avg mask-zero steps per eps: {:.1f}, max consecutive streak: {:.0f}".format(
                mz_all.get("mask_zero_ratio", 0.0) * mz_all.get("total_steps", 0) / max(1, mz_all["count"]),
                mz_all.get("max_streak", 0),
            )
        )
        f_hit = mz_fail.get("hit_count", 0)
        s_hit = mz_succ.get("hit_count", 0)
        if f_hit > 0 or s_hit > 0:
            print(
                "    Success eps hit mz: {}/{} ({:.0f}%), Failure eps hit mz: {}/{} ({:.0f}%)".format(
                    s_hit, mz_succ.get("count", 0),
                    mz_succ.get("hit_ratio", 0.0) * 100,
                    f_hit, mz_fail.get("count", 0),
                    mz_fail.get("hit_ratio", 0.0) * 100,
                )
            )

    # overall judgment
    print("\n  Judgment:")
    any_escapes = False
    for s in stage_summaries.values():
        if s.get("mask_zero_all", {}).get("escapes_via_stop_turn", 0) > 0:
            any_escapes = True
            break

    if any_escapes:
        total_esc_st = sum(
            s.get("mask_zero_all", {}).get("escapes_via_stop_turn", 0)
            for s in stage_summaries.values()
        )
        total_esc_any = sum(
            s.get("mask_zero_all", {}).get("escapes_total", 0)
            for s in stage_summaries.values()
        )
        print(
            "  YES: articulation changes CAN escape mask-zero. {} of {} total escapes"
            " ({:.0f}%) happened immediately after an exec stop-turn step.".format(
                total_esc_st, total_esc_any,
                total_esc_st / max(1, total_esc_any) * 100 if total_esc_any else 0.0,
            )
        )
    else:
        # check if any episodes hit mask-zero at all
        total_hit = sum(
            s.get("mask_zero_all", {}).get("hit_count", 0)
            for s in stage_summaries.values()
        )
        if total_hit > 0:
            print(
                "  PARTIAL: {} episodes hit mask-zero, but ZERO escapes via stop-turn."
                " Articulation changes are NOT effective for mask escape in these scenes.".format(
                    total_hit,
                )
            )
        else:
            print(
                "  No episodes hit mask-all-zero. The action mask never fully blocks"
                " the vehicle in the tested scenes/checkpoint."
            )


def _generate_conclusions(stage_summaries, all_failures):
    lines = []

    total_eps = sum(s["episodes"] for s in stage_summaries.values())
    total_succ = int(sum(
        s["episodes"] * s["success_rate"] for s in stage_summaries.values()
    ))
    total_coll = int(sum(
        s["episodes"] * s["collision_rate"] for s in stage_summaries.values()
    ))
    total_tout = int(sum(
        s["episodes"] * s["timeout_rate"] for s in stage_summaries.values()
    ))
    total_fail = total_eps - total_succ

    lines.append(
        "Overall: {} episodes, {:.1f}% success, {} failures ({} collisions, {} timeouts)".format(
            total_eps,
            100.0 * total_succ / total_eps if total_eps else 0.0,
            total_fail,
            total_coll,
            total_tout,
        )
    )

    stages = sorted(stage_summaries.keys())
    succ_trend = [stage_summaries[s]["success_rate"] * 100 for s in stages]
    coll_trend = [stage_summaries[s]["collision_rate"] * 100 for s in stages]
    tout_trend = [stage_summaries[s]["timeout_rate"] * 100 for s in stages]

    worst_idx = succ_trend.index(min(succ_trend))
    lines.append(
        "Success trend S1->S{}: {} -> lowest: S{}={:.1f}%".format(
            stages[-1],
            " -> ".join("{:.0f}%".format(v) for v in succ_trend),
            stages[worst_idx],
            min(succ_trend),
        )
    )

    # Stage 4 focus
    if 4 in stage_summaries:
        s4 = stage_summaries[4]
        lines.append("")
        lines.append(
            "Stage 4 (hardest): {:.1f}% succ, {:.1f}% coll, {:.1f}% tout".format(
                s4["success_rate"] * 100,
                s4["collision_rate"] * 100,
                s4["timeout_rate"] * 100,
            )
        )
        if "collision_avg_distance" in s4:
            lines.append(
                "  Collision: avg {:.2f}m from goal, |phi|={:.1f}deg, overlap={:.3f}".format(
                    s4.get("collision_avg_distance", 0),
                    s4.get("collision_avg_phi_deg", 0),
                    s4.get("collision_avg_overlap", 0),
                )
            )

    # Collision patterns
    collisions = [ep for ep in all_failures if ep.get("collision")]
    if collisions:
        lines.append("")
        lines.append("Collision pattern ({} events):".format(len(collisions)))
        c_stages = Counter(
            str(ep.get("scene_meta", {}).get("stage", "?")) for ep in collisions
        )
        dominant = c_stages.most_common(1)[0]
        lines.append(
            "  {}% occur in Stage {}.".format(
                int(100.0 * dominant[1] / len(collisions)), dominant[0],
            )
        )
        c_scenarios = Counter(ep.get("scenario", "") for ep in collisions)
        if c_scenarios:
            top = c_scenarios.most_common(2)
            lines.append(
                "  Top scenarios: {}".format(
                    ", ".join("{} ({})".format(s, c) for s, c in top)
                )
            )

        c_dists = [float(ep["final_info"].get("distance_to_goal", 0.0)) for ep in collisions]
        lines.append(
            "  Mean distance to goal at collision: {:.2f}m (range {:.2f}-{:.2f}m)".format(
                np.mean(c_dists), np.min(c_dists), np.max(c_dists),
            )
        )
        c_approach = Counter(
            ep.get("scene_meta", {}).get("approach_side_bucket", "") for ep in collisions
        )
        if len(c_approach) > 1:
            lines.append("  Approach side skew: {}".format(dict(c_approach)))

    # Timeout patterns
    timeouts = [ep for ep in all_failures if ep.get("timeout") and not ep.get("collision")]
    if timeouts:
        lines.append("")
        lines.append("Timeout pattern ({} events):".format(len(timeouts)))
        t_ovl = [float(ep["final_info"].get("front_overlap", 0.0)) for ep in timeouts]
        t_head = [float(ep["final_info"].get("heading_error_deg", 0.0)) for ep in timeouts]
        lines.append(
            "  Avg overlap={:.3f}, avg heading={:.1f}deg".format(
                np.mean(t_ovl), np.mean(t_head),
            )
        )
        near = sum(1 for o in t_ovl if o >= 0.6)
        lines.append(
            "  {}/{} timeouts near success (overlap>=0.6). Success threshold is 0.8.".format(
                near, len(timeouts),
            )
        )

    # Articulation
    artic = [ep for ep in all_failures if ep.get("articulation") and not ep.get("collision")]
    if artic:
        lines.append("")
        lines.append("Articulation violations: {} events across stages.".format(len(artic)))

    # Hardest scenario per stage
    lines.append("")
    lines.append("Per-stage hardest scenarios:")
    for stage in sorted(stage_summaries.keys()):
        ss = stage_summaries[stage].get("scenario_stats", {})
        worst_name = None
        worst_fail = 0.0
        for sc_name, stats in ss.items():
            fail_rate = 1.0 - stats["success_rate"]
            if fail_rate > worst_fail:
                worst_fail = fail_rate
                worst_name = sc_name
        if worst_name and worst_fail > 0.0:
            lines.append(
                "  S{}: '{}' ({:.1f}% fail, {} eps)".format(
                    stage, worst_name, worst_fail * 100, ss[worst_name]["count"],
                )
            )

    # Stop-and-turn findings
    all_st_agg = {}
    for stage in sorted(stage_summaries.keys()):
        s = stage_summaries[stage]
        all_st = s.get("stop_turn_all", {})
        succ_st = s.get("stop_turn_success", {})
        fail_st = s.get("stop_turn_failure", {})
        all_st_agg[stage] = {
            "pol_ratio": all_st.get("stop_turn_policy_ratio", 0.0),
            "exec_ratio": all_st.get("stop_turn_exec_ratio", 0.0),
            "succ_pol": succ_st.get("stop_turn_policy_ratio", 0.0),
            "fail_pol": fail_st.get("stop_turn_policy_ratio", 0.0),
            "mask_block": all_st.get("stop_turn_masked_ratio", 0.0),
            "fail_count": fail_st.get("count", 0),
        }
    max_pol = max(v["pol_ratio"] for v in all_st_agg.values()) if all_st_agg else 0.0

    lines.append("")
    lines.append("Stop-and-turn behavior:")
    if max_pol > 0.02:
        lines.append(
            "  Policy HAS learned proactive stop-and-turn (peak {:.1f}% of steps).".format(
                max_pol * 100,
            )
        )
        # contrast between success and failure
        for stage, v in sorted(all_st_agg.items()):
            if v["fail_count"] >= 2:
                delta = v["fail_pol"] - v["succ_pol"]
                if abs(delta) > 0.01:
                    lines.append(
                        "  S{}: stop-turn in fail={:.1f}% vs success={:.1f}%"
                        " — {} in failures.".format(
                            stage,
                            v["fail_pol"] * 100, v["succ_pol"] * 100,
                            "more frequent" if delta > 0 else "less frequent",
                        )
                    )
        mask_stages = [s for s, v in all_st_agg.items() if v["mask_block"] > 0.1]
        if mask_stages:
            lines.append(
                "  Mask blocks {:.0f}% of stop-turn intents in stage(s) {}. Policy tries but mask denies.".format(
                    max(v["mask_block"] for v in all_st_agg.values()) * 100,
                    ", ".join(str(s) for s in mask_stages),
                )
            )
    else:
        lines.append(
            "  Policy has NOT learned proactive stop-and-turn (<2% stop-turn steps)."
            " Vehicle controls speed and articulation simultaneously rather than in-place turning."
        )

    for line in lines:
        print("  " + line)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint with detailed failure analysis and rendering."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Deterministic episodes per family per stage (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stages", type=int, nargs="+", default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "failures"),
    )
    parser.add_argument(
        "--show-lidar", action="store_true",
        help="Overlay LiDAR rays on failure images",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print("ERROR: checkpoint not found: {}".format(args.checkpoint))
        sys.exit(1)

    evaluate_checkpoint_with_failures(
        checkpoint_path=args.checkpoint,
        episodes_per_family=args.episodes,
        seed=args.seed,
        device=args.device,
        stages=args.stages,
        output_dir=args.output_dir,
        show_lidar=args.show_lidar,
    )


if __name__ == "__main__":
    main()
