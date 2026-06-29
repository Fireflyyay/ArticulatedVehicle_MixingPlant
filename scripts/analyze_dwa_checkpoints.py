#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEFAULT_ENV_CONFIG, DEFAULT_SCENE_CONFIG  # noqa: E402
from env.local_parking_env import LocalParkingEnv  # noqa: E402
from env.vehicle import ArticulatedState  # noqa: E402
from model.continuous_ppo import ContinuousPPOAgent  # noqa: E402
from train.curriculum import MultiStageScenePool  # noqa: E402
from visualize_local_parking_paths import (  # noqa: E402
    _plot_bay,
    _plot_obstacles,
    _plot_polygon,
    _plot_polygon_outline,
    _plot_scene_regions,
)


TASK_FAMILIES = ("head_in",)
EPISODE_RE = re.compile(r"checkpoint_episode_(\d+)\.pt$")


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return dict((str(k), _json_safe(v)) for k, v in value.items())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _scalar_dict(data):
    out = {}
    for key, value in data.items():
        safe = _json_safe(value)
        if isinstance(safe, (bool, int, float, str)) or safe is None:
            out[str(key)] = safe
    return out


def _state_to_dict(state):
    return {
        "x_front": float(state.x_front),
        "y_front": float(state.y_front),
        "theta_front": float(state.theta_front),
        "theta_rear": float(state.theta_rear),
        "phi": float(state.phi),
        "v": float(state.v),
        "phi_dot": float(state.phi_dot),
    }


def _dict_to_state(data):
    return ArticulatedState(
        x_front=float(data["x_front"]),
        y_front=float(data["y_front"]),
        theta_front=float(data["theta_front"]),
        theta_rear=float(data["theta_rear"]),
        v=float(data.get("v", 0.0)),
        phi_dot=float(data.get("phi_dot", 0.0)),
    )


def _status_label(info):
    if bool(info.get("success", False)):
        return "success"
    if bool(info.get("collision", False)):
        return "collision"
    if bool(info.get("deadlock", False)):
        return "deadlock"
    if bool(info.get("out_of_bounds", False)):
        return "out_of_bounds"
    if bool(info.get("articulation_limit_violation", False)):
        return "articulation"
    if bool(info.get("timeout", False)):
        return "timeout"
    return "done"


def _checkpoint_episode(path):
    match = EPISODE_RE.match(os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _checkpoint_sort_key(path):
    name = os.path.basename(path)
    episode = _checkpoint_episode(path)
    if episode is not None:
        return (1, episode, name)
    if name.startswith("checkpoint_best"):
        return (0, 0, name)
    return (2, 0, name)


def _select_checkpoints(run_dir, mode, stride, explicit_names):
    paths = []
    if explicit_names:
        for name in explicit_names:
            path = name if os.path.isabs(name) else os.path.join(run_dir, name)
            if not os.path.isfile(path):
                raise ValueError("checkpoint not found: {}".format(path))
            paths.append(os.path.abspath(path))
        return sorted(dict((p, None) for p in paths).keys(), key=_checkpoint_sort_key)

    all_paths = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.startswith("checkpoint") and name.endswith(".pt")
    ]
    all_paths = sorted(all_paths, key=_checkpoint_sort_key)
    if mode == "all":
        return [os.path.abspath(path) for path in all_paths]

    best_paths = [
        path
        for path in all_paths
        if os.path.basename(path).startswith("checkpoint_best")
    ]
    episode_paths = [
        path
        for path in all_paths
        if _checkpoint_episode(path) is not None
    ]
    selected = list(best_paths)
    if mode == "latest":
        if episode_paths:
            selected.append(episode_paths[-1])
    elif mode == "stride":
        stride = max(1, int(stride))
        for path in episode_paths:
            episode = _checkpoint_episode(path)
            if episode is not None and episode % stride == 0:
                selected.append(path)
        if episode_paths:
            selected.append(episode_paths[-1])
    else:
        raise ValueError("unsupported checkpoint mode: {}".format(mode))
    return sorted(dict((os.path.abspath(p), None) for p in selected).keys(), key=_checkpoint_sort_key)


def _load_agent(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = ContinuousPPOAgent(device=device)
    agent.network.load_state_dict(payload["network"])
    agent.network.eval()
    return agent, dict(payload.get("extra", {}))


def _env_config(stage, max_steps):
    kwargs = dict(
        curriculum_stage=int(stage),
        scene_family_schedule=TASK_FAMILIES,
        use_hybrid_astar=False,
        rs_potential_enabled=False,
        enable_hope_teacher=False,
        use_teacher_reward=False,
        enable_offpath_reset=False,
        enable_failure_aggregation=False,
        enable_dwa_recovery=True,
        dwa_override_policy_action=True,
        dwa_enable_deadlock_termination=False,
    )
    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)
    return replace(DEFAULT_ENV_CONFIG, **kwargs)


def _simulate_dwa_preview(vehicle_model, start_state, preview_action, horizon):
    action = np.asarray(preview_action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 2:
        return []
    current = replace(start_state)
    states = [_state_to_dict(current)]
    for _ in range(max(1, int(horizon))):
        current = vehicle_model.step(current, action[:2])
        states.append(_state_to_dict(current))
    return states


def _action_list(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        return [0.0, 0.0]
    return [float(arr[0]), float(arr[1])]


def _compact_step_info(info):
    keys = (
        "success",
        "collision",
        "timeout",
        "deadlock",
        "failure_type",
        "front_overlap",
        "rear_body_overlap",
        "heading_error_deg",
        "rear_heading_error_deg",
        "distance_to_goal",
        "min_lidar_distance",
        "max_safe_ratio",
        "mask_safe_ratio",
        "mask_safe_ratio_mean",
        "mask_zero_fraction",
        "forced_stop",
        "policy_forced_stop",
        "selected_action_masked",
        "gear",
        "policy_gear",
        "raw_safe_ratio",
        "exec_safe_ratio",
        "policy_raw_safe_ratio",
        "mask_all_zero_before_floor",
        "mask_max_before_floor",
        "dwa_enabled",
        "dwa_triggered",
        "dwa_used",
        "dwa_mode",
        "dwa_reason",
        "dwa_candidate_count",
        "dwa_valid_candidate_count",
        "dwa_unlock_success",
        "dwa_unlock_step",
        "dwa_deadlock",
        "dwa_final_max_safe_ratio",
        "dwa_override_policy_action",
        "dwa_policy_invalid_trigger",
        "dwa_low_safe_trigger",
        "dwa_all_zero_trigger",
    )
    return dict((key, _json_safe(info.get(key))) for key in keys if key in info)


def _rollout(env, agent, deterministic, horizon):
    obs, reset_info = env.reset()
    scene = env.scene
    slot = env.slot
    states = [_state_to_dict(env.state)]
    steps = []
    total_reward = 0.0
    final_info = dict(reset_info)
    terminated = False
    truncated = False
    while not bool(terminated or truncated):
        pre_state = replace(env.state)
        raw_action, _, _ = agent.act(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(raw_action)
        total_reward += float(reward)
        final_info = dict(info)
        post_state = replace(env.state)
        preview = info.get("dwa_executed_action_preview", np.zeros(2, dtype=np.float32))
        dwa_segment = []
        if bool(info.get("dwa_used", False)):
            dwa_segment = _simulate_dwa_preview(
                env.vehicle_model,
                pre_state,
                preview,
                horizon,
            )
        steps.append(
            {
                "step": int(env.step_count),
                "pre_state": _state_to_dict(pre_state),
                "post_state": _state_to_dict(post_state),
                "policy_raw_action": _action_list(info.get("policy_raw_action", raw_action)),
                "execution_raw_action": _action_list(info.get("raw_action", raw_action)),
                "executed_action": _action_list(info.get("executed_action", np.zeros(2))),
                "dwa_policy_raw_action": _action_list(info.get("dwa_policy_raw_action", np.zeros(2))),
                "dwa_raw_action": _action_list(info.get("dwa_raw_action", np.zeros(2))),
                "dwa_executed_action_preview": _action_list(preview),
                "dwa_preview_states": dwa_segment,
                "info": _compact_step_info(info),
            }
        )
        states.append(_state_to_dict(post_state))
    return {
        "scene": scene,
        "slot": slot,
        "reset_info": dict(reset_info),
        "scene_meta": dict(scene.metadata),
        "states": states,
        "steps": steps,
        "final_info": final_info,
        "total_reward": float(total_reward),
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def _plot_failure(path, env, rollout, checkpoint_name, stage, episode_index):
    scene = rollout["scene"]
    slot = rollout["slot"]
    states = rollout["states"]
    steps = rollout["steps"]
    final_info = rollout["final_info"]
    reset_info = rollout["reset_info"]

    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_scene_regions(ax, scene)
    _plot_obstacles(ax, scene)
    for bay in scene.parking_bays:
        _plot_bay(ax, bay)

    target_rear = env.vehicle_model.target_rear_box(
        slot.x_goal,
        slot.y_goal,
        slot.theta_goal,
    )
    _plot_polygon(ax, target_rear, "#f4a261", "#9c4f15", alpha=0.35, linewidth=2.0, zorder=3)
    _plot_polygon(ax, slot.front_box(), "#8fd175", "#207020", alpha=0.48, linewidth=2.4, zorder=3)

    front = np.asarray([(s["x_front"], s["y_front"]) for s in states], dtype=np.float64)
    rear = np.asarray(
        [env.vehicle_model.rear_center(_dict_to_state(s)) for s in states],
        dtype=np.float64,
    )
    ax.plot(front[:, 0], front[:, 1], color="#005f73", linewidth=2.2, label="executed front path", zorder=7)
    ax.plot(rear[:, 0], rear[:, 1], color="#ca6702", linewidth=1.4, linestyle=":", label="executed rear path", zorder=6)
    ax.scatter([front[0, 0]], [front[0, 1]], color="#005f73", marker="o", s=42, edgecolors="black", zorder=9)
    ax.scatter([front[-1, 0]], [front[-1, 1]], color="#ae2012", marker="X", s=58, edgecolors="black", zorder=9)

    start_front, start_rear = env.vehicle_model.body_boxes(_dict_to_state(states[0]))
    end_front, end_rear = env.vehicle_model.body_boxes(_dict_to_state(states[-1]))
    _plot_polygon_outline(ax, start_rear, "#005f73", alpha=0.45, linewidth=1.2)
    _plot_polygon_outline(ax, start_front, "#005f73", alpha=0.75, linewidth=1.7)
    _plot_polygon_outline(ax, end_rear, "#ae2012", alpha=0.65, linewidth=1.3, linestyle="--")
    _plot_polygon_outline(ax, end_front, "#ae2012", alpha=0.95, linewidth=1.9, linestyle="--")

    plotted_dwa = False
    for step in steps:
        segment = step.get("dwa_preview_states") or []
        if len(segment) < 2:
            continue
        pts = np.asarray([(s["x_front"], s["y_front"]) for s in segment], dtype=np.float64)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#9b2226",
            linewidth=1.4,
            alpha=0.45,
            linestyle="-.",
            label="DWA selected preview" if not plotted_dwa else None,
            zorder=8,
        )
        plotted_dwa = True

    xmin, ymin, xmax, ymax = scene.world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    status = _status_label(final_info)
    dwa_steps = sum(1 for step in steps if bool(step["info"].get("dwa_triggered", False)))
    used_steps = sum(1 for step in steps if bool(step["info"].get("dwa_used", False)))
    title = "{} | stage {} ep {} | {} | scene_seed {}".format(
        checkpoint_name,
        int(stage),
        int(episode_index),
        status,
        int(final_info.get("scene_seed", reset_info.get("scene_seed", -1))),
    )
    ax.set_title(title, fontsize=9)
    scene_type = str(
        rollout["scene_meta"].get(
            "scene_type",
            reset_info.get("scene_type", "?"),
        )
    )
    requested_scene_type = str(
        rollout["scene_meta"].get(
            "requested_scene_type",
            reset_info.get("requested_scene_type", scene_type),
        )
    )
    details = [
        "scene_type={} requested={}".format(scene_type, requested_scene_type),
        "scenario={} steps={} reward={:.2f}".format(
            reset_info.get("scenario_type", "?"),
            max(0, len(states) - 1),
            float(rollout["total_reward"]),
        ),
        "overlap={:.3f} heading={:.1f}deg dist={:.2f}m".format(
            float(final_info.get("front_overlap", 0.0)),
            float(final_info.get("heading_error_deg", 0.0)),
            float(final_info.get("distance_to_goal", 0.0)),
        ),
        "DWA triggered={} used={} final_mode={} valid={}/{}".format(
            int(dwa_steps),
            int(used_steps),
            str(final_info.get("dwa_mode", "none")),
            int(final_info.get("dwa_valid_candidate_count", 0)),
            int(final_info.get("dwa_candidate_count", 0)),
        ),
    ]
    ax.text(
        0.01,
        0.01,
        "\n".join(details),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.80},
        zorder=20,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.86)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _episode_summary(checkpoint_path, checkpoint_extra, stage, episode_index, rollout, image_path):
    final_info = rollout["final_info"]
    steps = rollout["steps"]
    speeds = [abs(float(step["executed_action"][0])) for step in steps]
    phi_rates = [abs(float(step["executed_action"][1])) for step in steps]
    forced_stop_steps = sum(1 for step in steps if bool(step["info"].get("forced_stop", False)))
    policy_forced_stop_steps = sum(1 for step in steps if bool(step["info"].get("policy_forced_stop", False)))
    dwa_trigger_steps = sum(1 for step in steps if bool(step["info"].get("dwa_triggered", False)))
    dwa_used_steps = sum(1 for step in steps if bool(step["info"].get("dwa_used", False)))
    dwa_deadlock_steps = sum(1 for step in steps if bool(step["info"].get("dwa_deadlock", False)))
    mask_zero_steps = sum(1 for step in steps if bool(step["info"].get("mask_all_zero_before_floor", False)))
    valid_counts = [
        int(step["info"].get("dwa_valid_candidate_count", 0))
        for step in steps
        if bool(step["info"].get("dwa_triggered", False))
    ]
    return {
        "checkpoint": os.path.basename(checkpoint_path),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_episode": _checkpoint_episode(checkpoint_path),
        "checkpoint_extra_episode": checkpoint_extra.get("episode"),
        "checkpoint_extra_stage": checkpoint_extra.get("stage"),
        "stage": int(stage),
        "episode_index": int(episode_index),
        "status": _status_label(final_info),
        "success": bool(final_info.get("success", False)),
        "failure_type": str(final_info.get("failure_type", _status_label(final_info))),
        "scenario_type": str(rollout["reset_info"].get("scenario_type", "")),
        "scene_seed": int(final_info.get("scene_seed", rollout["reset_info"].get("scene_seed", -1))),
        "task_family": str(rollout["reset_info"].get("task_family", "")),
        "steps": max(0, len(rollout["states"]) - 1),
        "total_reward": float(rollout["total_reward"]),
        "front_overlap": float(final_info.get("front_overlap", 0.0)),
        "rear_body_overlap": float(final_info.get("rear_body_overlap", 0.0)),
        "heading_error_deg": float(final_info.get("heading_error_deg", 0.0)),
        "rear_heading_error_deg": float(final_info.get("rear_heading_error_deg", 0.0)),
        "distance_to_goal": float(final_info.get("distance_to_goal", 0.0)),
        "min_lidar_distance": float(final_info.get("min_lidar_distance", 0.0)),
        "max_safe_ratio": float(final_info.get("max_safe_ratio", 0.0)),
        "avg_abs_v_exec": float(np.mean(speeds)) if speeds else 0.0,
        "max_abs_v_exec": float(np.max(speeds)) if speeds else 0.0,
        "avg_abs_phi_dot_exec": float(np.mean(phi_rates)) if phi_rates else 0.0,
        "forced_stop_steps": int(forced_stop_steps),
        "policy_forced_stop_steps": int(policy_forced_stop_steps),
        "mask_zero_steps": int(mask_zero_steps),
        "dwa_trigger_steps": int(dwa_trigger_steps),
        "dwa_used_steps": int(dwa_used_steps),
        "dwa_deadlock_steps": int(dwa_deadlock_steps),
        "dwa_valid_candidate_steps": int(sum(1 for value in valid_counts if value > 0)),
        "dwa_avg_valid_candidates": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "image_path": image_path,
        "reset_info": _scalar_dict(rollout["reset_info"]),
        "scene_meta": _scalar_dict(rollout["scene_meta"]),
    }


def _aggregate(summaries):
    aggregate = {
        "episodes": len(summaries),
        "success_rate": 0.0,
        "status_counts": {},
        "failure_type_counts": {},
        "by_stage": {},
        "by_checkpoint": {},
        "by_scenario": {},
        "dwa": {},
        "scene_health": {},
    }
    if not summaries:
        return aggregate
    aggregate["success_rate"] = sum(1 for item in summaries if item["success"]) / float(len(summaries))
    aggregate["status_counts"] = dict(Counter(item["status"] for item in summaries))
    aggregate["failure_type_counts"] = dict(Counter(item["failure_type"] for item in summaries if not item["success"]))
    for key_name, target_key in (("stage", "by_stage"), ("checkpoint", "by_checkpoint"), ("scenario_type", "by_scenario")):
        grouped = defaultdict(list)
        for item in summaries:
            grouped[str(item[key_name])].append(item)
        for key, entries in sorted(grouped.items()):
            n = len(entries)
            aggregate[target_key][key] = {
                "episodes": int(n),
                "success_rate": sum(1 for item in entries if item["success"]) / float(n),
                "failure_type_counts": dict(Counter(item["failure_type"] for item in entries if not item["success"])),
                "dwa_trigger_episode_rate": sum(1 for item in entries if item["dwa_trigger_steps"] > 0) / float(n),
                "dwa_used_episode_rate": sum(1 for item in entries if item["dwa_used_steps"] > 0) / float(n),
                "avg_steps": float(np.mean([item["steps"] for item in entries])),
            }
    successes = [item for item in summaries if item["success"]]
    failures = [item for item in summaries if not item["success"]]
    for label, entries in (("success", successes), ("failure", failures), ("all", summaries)):
        n = len(entries)
        aggregate["dwa"][label] = {
            "episodes": int(n),
            "trigger_episode_rate": sum(1 for item in entries if item["dwa_trigger_steps"] > 0) / float(n) if n else 0.0,
            "used_episode_rate": sum(1 for item in entries if item["dwa_used_steps"] > 0) / float(n) if n else 0.0,
            "avg_trigger_steps": float(np.mean([item["dwa_trigger_steps"] for item in entries])) if n else 0.0,
            "avg_used_steps": float(np.mean([item["dwa_used_steps"] for item in entries])) if n else 0.0,
            "avg_forced_stop_steps": float(np.mean([item["forced_stop_steps"] for item in entries])) if n else 0.0,
            "avg_mask_zero_steps": float(np.mean([item["mask_zero_steps"] for item in entries])) if n else 0.0,
            "avg_valid_candidates": float(np.mean([item["dwa_avg_valid_candidates"] for item in entries])) if n else 0.0,
        }
    bad_scene = []
    for item in summaries:
        feasible = item["scene_meta"].get("success_neighborhood_feasible_count")
        if feasible is not None and int(feasible) <= 0:
            bad_scene.append(item)
    aggregate["scene_health"] = {
        "zero_success_neighborhood_episodes": int(len(bad_scene)),
        "zero_success_neighborhood_rate": len(bad_scene) / float(len(summaries)),
        "unique_zero_success_neighborhood_seeds": sorted(set(int(item["scene_seed"]) for item in bad_scene)),
    }
    return aggregate


def _percent(value):
    return "{:.1f}%".format(100.0 * float(value))


def _write_analysis(path, args, aggregate, checkpoint_summaries):
    lines = []
    lines.append("# DWA checkpoint failure analysis")
    lines.append("")
    lines.append("run_dir: `{}`".format(os.path.abspath(args.run_dir)))
    lines.append("episodes_per_stage: `{}`".format(int(args.episodes_per_stage)))
    lines.append("stages: `{}`".format(",".join(str(s) for s in args.stages)))
    lines.append("DWA: enabled, override policy action, deadlock termination disabled")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("- episodes: {}".format(int(aggregate["episodes"])))
    lines.append("- success_rate: {}".format(_percent(aggregate["success_rate"])))
    lines.append("- status_counts: `{}`".format(aggregate["status_counts"]))
    lines.append("- failure_type_counts: `{}`".format(aggregate["failure_type_counts"]))
    lines.append("- scene_zero_success_neighborhood_rate: {}".format(
        _percent(aggregate["scene_health"]["zero_success_neighborhood_rate"])
    ))
    lines.append("")
    lines.append("## Stage summary")
    lines.append("")
    for stage, item in sorted(aggregate["by_stage"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            "- stage {}: episodes={} success={} failures={} dwa_used_ep={}".format(
                stage,
                int(item["episodes"]),
                _percent(item["success_rate"]),
                item["failure_type_counts"],
                _percent(item["dwa_used_episode_rate"]),
            )
        )
    lines.append("")
    lines.append("## Best checkpoints by success rate")
    lines.append("")
    ranked = sorted(
        aggregate["by_checkpoint"].items(),
        key=lambda kv: (float(kv[1]["success_rate"]), -float(kv[1]["avg_steps"]), kv[0]),
        reverse=True,
    )
    for name, item in ranked[:12]:
        lines.append(
            "- {}: success={} episodes={} failures={} dwa_used_ep={}".format(
                name,
                _percent(item["success_rate"]),
                int(item["episodes"]),
                item["failure_type_counts"],
                _percent(item["dwa_used_episode_rate"]),
            )
        )
    lines.append("")
    lines.append("## DWA motion diagnostics")
    lines.append("")
    for label in ("success", "failure", "all"):
        item = aggregate["dwa"][label]
        lines.append(
            "- {}: trigger_ep={} used_ep={} avg_trigger_steps={:.2f} avg_used_steps={:.2f} "
            "avg_forced_stop_steps={:.2f} avg_mask_zero_steps={:.2f} avg_valid_candidates={:.2f}".format(
                label,
                _percent(item["trigger_episode_rate"]),
                _percent(item["used_episode_rate"]),
                float(item["avg_trigger_steps"]),
                float(item["avg_used_steps"]),
                float(item["avg_forced_stop_steps"]),
                float(item["avg_mask_zero_steps"]),
                float(item["avg_valid_candidates"]),
            )
        )
    lines.append("")
    lines.append("## Evidence-backed interpretation")
    lines.append("")
    failure_dwa = aggregate["dwa"]["failure"]
    scene_rate = float(aggregate["scene_health"]["zero_success_neighborhood_rate"])
    timeout_count = int(aggregate["failure_type_counts"].get("timeout", 0))
    collision_count = int(aggregate["failure_type_counts"].get("collision", 0))
    deadlock_count = int(aggregate["failure_type_counts"].get("deadlock", 0))
    if scene_rate > 0.0:
        lines.append(
            "- Some evaluated scenes report empty success neighborhoods. Inspect those seeds first; they may be ill-conditioned for the configured success threshold."
        )
    else:
        lines.append(
            "- The sampled scenes did not expose empty success neighborhoods, so the dominant failures are more likely policy/controller limitations than impossible target scenes."
        )
    if timeout_count > 0:
        lines.append(
            "- Timeouts indicate the policy/DWA stack can stay collision-free but still fail terminal overlap or heading. This points to weak fine alignment and stopping behavior near the bay."
        )
    if collision_count > 0:
        lines.append(
            "- Collisions indicate the DWA trigger is reactive and local: it can select a short collision-free preview yet still leave the vehicle in a bad later pose."
        )
    if deadlock_count > 0 or failure_dwa["avg_valid_candidates"] < 1.0:
        lines.append(
            "- Low valid DWA candidate counts in failures show that strict mask constraints often leave little usable recovery action once the vehicle is already boxed in."
        )
    if failure_dwa["used_episode_rate"] > 0.5:
        lines.append(
            "- DWA is frequently used in failed episodes, so it is acting as a late rescue layer rather than solving the nominal parking objective."
        )
    lines.append("")
    lines.append("## Next steps to raise success rate")
    lines.append("")
    lines.append("- Add a terminal alignment controller or train a dedicated near-goal recovery slice for high-overlap timeouts.")
    lines.append("- Trigger DWA before all-zero or forced-stop streaks accumulate, using falling safe-ratio and progress stagnation as early signals.")
    lines.append("- Let DWA score multi-step terminal overlap/heading more strongly, not only safe-ratio recovery, so local rescue actions point toward the parking objective.")
    lines.append("- Re-evaluate bad seeds with saved failure JSON and rendered DWA preview segments before changing scene generation.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `episode_summary.jsonl`: one row per evaluated episode")
    lines.append("- `checkpoint_stage_summary.jsonl`: per checkpoint/stage aggregate rows")
    lines.append("- `failure_trajectories.jsonl`: full state/action/DWA data for failed episodes")
    lines.append("- `failures/`: rendered failed scenes with executed path and DWA selected previews")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_analysis(args):
    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir)
    failures_dir = os.path.join(output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    checkpoints = _select_checkpoints(
        run_dir,
        args.checkpoint_mode,
        args.checkpoint_stride,
        args.checkpoints,
    )
    if not checkpoints:
        raise ValueError("no checkpoints selected from {}".format(run_dir))

    episode_records = []
    failure_records = []
    stage_records = []
    checkpoint_extras = {}
    print("selected {} checkpoints".format(len(checkpoints)), flush=True)
    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = os.path.basename(checkpoint_path)
        agent, extra = _load_agent(checkpoint_path, args.device)
        checkpoint_extras[checkpoint_name] = _json_safe(extra)
        print(
            "[{}/{}] {}".format(checkpoint_index + 1, len(checkpoints), checkpoint_name),
            flush=True,
        )
        for stage in args.stages:
            stage = int(stage)
            config = _env_config(stage, args.max_steps)
            scene_config = replace(DEFAULT_SCENE_CONFIG, scene_type=str(args.scene_type))
            pool = MultiStageScenePool(
                pool_size=int(config.scene_pool_size),
                base_seed=int(args.seed),
                scene_config=scene_config,
                family_schedule=config.scene_family_schedule,
            )
            env = LocalParkingEnv(
                config=config,
                scene_config=scene_config,
                multi_stage_pool=pool,
                seed=int(args.seed),
            )
            env.set_active_stage(stage)
            stage_episode_records = []
            rendered_failures = 0
            for episode_index in range(int(args.episodes_per_stage)):
                rollout = _rollout(
                    env,
                    agent,
                    deterministic=not bool(args.stochastic),
                    horizon=int(config.dwa_horizon_steps),
                )
                final_info = rollout["final_info"]
                status = _status_label(final_info)
                image_path = ""
                is_failure = not bool(final_info.get("success", False))
                if is_failure:
                    failure_name = "{}_stage{}_ep{:04d}_{}_seed{}.png".format(
                        os.path.splitext(checkpoint_name)[0],
                        stage,
                        episode_index + 1,
                        status,
                        int(final_info.get("scene_seed", rollout["reset_info"].get("scene_seed", -1))),
                    )
                    image_path = os.path.join(failures_dir, failure_name)
                    if rendered_failures < int(args.max_failure_images_per_stage):
                        _plot_failure(
                            image_path,
                            env,
                            rollout,
                            checkpoint_name,
                            stage,
                            episode_index + 1,
                        )
                        rendered_failures += 1
                    else:
                        image_path = ""
                    failure_record = {
                        "checkpoint": checkpoint_name,
                        "checkpoint_path": os.path.abspath(checkpoint_path),
                        "stage": stage,
                        "episode_index": int(episode_index),
                        "status": status,
                        "image_path": image_path,
                        "reset_info": _scalar_dict(rollout["reset_info"]),
                        "scene_meta": _scalar_dict(rollout["scene_meta"]),
                        "final_info": _compact_step_info(final_info),
                        "total_reward": float(rollout["total_reward"]),
                        "states": rollout["states"],
                        "steps": rollout["steps"],
                    }
                    failure_records.append(failure_record)
                summary = _episode_summary(
                    checkpoint_path,
                    extra,
                    stage,
                    episode_index,
                    rollout,
                    image_path,
                )
                episode_records.append(summary)
                stage_episode_records.append(summary)
                if (episode_index + 1) % int(args.progress_interval) == 0:
                    successes = sum(1 for item in stage_episode_records if item["success"])
                    print(
                        "  stage {} {}/{} success {:.1f}% failures {}".format(
                            stage,
                            episode_index + 1,
                            int(args.episodes_per_stage),
                            100.0 * successes / float(episode_index + 1),
                            (episode_index + 1) - successes,
                        ),
                        flush=True,
                    )
            env.close() if hasattr(env, "close") else None
            stage_aggregate = _aggregate(stage_episode_records)
            stage_record = {
                "checkpoint": checkpoint_name,
                "stage": int(stage),
                "episodes": int(stage_aggregate["episodes"]),
                "success_rate": float(stage_aggregate["success_rate"]),
                "status_counts": stage_aggregate["status_counts"],
                "failure_type_counts": stage_aggregate["failure_type_counts"],
                "dwa": stage_aggregate["dwa"],
                "rendered_failures": int(rendered_failures),
            }
            stage_records.append(stage_record)
            print(
                "  stage {} done success {:.1f}% status {}".format(
                    stage,
                    100.0 * stage_record["success_rate"],
                    stage_record["status_counts"],
                ),
                flush=True,
            )

    aggregate = _aggregate(episode_records)
    aggregate["run_dir"] = run_dir
    aggregate["output_dir"] = output_dir
    aggregate["checkpoint_count"] = int(len(checkpoints))
    aggregate["selected_checkpoints"] = [os.path.basename(path) for path in checkpoints]
    aggregate["checkpoint_extras"] = checkpoint_extras
    aggregate["args"] = _json_safe(vars(args))

    _write_jsonl(os.path.join(output_dir, "episode_summary.jsonl"), episode_records)
    _write_jsonl(os.path.join(output_dir, "checkpoint_stage_summary.jsonl"), stage_records)
    _write_jsonl(os.path.join(output_dir, "failure_trajectories.jsonl"), failure_records)
    with open(os.path.join(output_dir, "aggregate_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(aggregate), handle, ensure_ascii=False, indent=2, sort_keys=True)
    _write_analysis(
        os.path.join(output_dir, "analysis.md"),
        args,
        aggregate,
        stage_records,
    )
    print("wrote {}".format(output_dir), flush=True)
    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate checkpoints with DWA enabled and export failed paths."
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(REPO_ROOT, "runs", "local_parking_20260627_143716_seed0"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "dwa_failures"),
    )
    parser.add_argument("--checkpoint-mode", choices=("all", "stride", "latest"), default="stride")
    parser.add_argument("--checkpoint-stride", type=int, default=1000)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--episodes-per-stage", type=int, default=30)
    parser.add_argument("--max-failure-images-per-stage", type=int, default=6)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--scene-type", default=DEFAULT_SCENE_CONFIG.scene_type)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    if int(args.episodes_per_stage) <= 0:
        raise ValueError("--episodes-per-stage must be positive")
    if int(args.progress_interval) <= 0:
        raise ValueError("--progress-interval must be positive")
    if int(args.max_failure_images_per_stage) < 0:
        raise ValueError("--max-failure-images-per-stage must be non-negative")
    run_analysis(args)


if __name__ == "__main__":
    main()
