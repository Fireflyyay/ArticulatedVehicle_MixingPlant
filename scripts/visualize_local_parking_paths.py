#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, replace
import os
import sys

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

from config import DEFAULT_ENV_CONFIG  # noqa: E402
from env.local_parking_env import LocalParkingEnv  # noqa: E402
from model.continuous_ppo import ContinuousPPOAgent  # noqa: E402


@dataclass
class PathRollout:
    seed: int
    scene: object
    slot: object
    reset_info: dict
    states: list
    final_info: dict
    total_reward: float
    rs_guide_path: object = None
    rs_guide_step: int = -1
    rs_guide_state: object = None
    rs_guide_info: object = None
    rs_guide_source: str = ""


def _plot_polygon(ax, polygon, facecolor, edgecolor, alpha=1.0, linewidth=1.0, **kwargs):
    coords = np.asarray(polygon.exterior.coords)
    ax.fill(
        coords[:, 0],
        coords[:, 1],
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        **kwargs
    )


def _plot_polygon_outline(ax, polygon, color, alpha=1.0, linewidth=1.0, linestyle="-"):
    coords = np.asarray(polygon.exterior.coords)
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def _plot_bay(ax, bay):
    is_target = bool(bay.is_target)
    _plot_polygon(
        ax,
        bay.polygon,
        "#f6bd60" if is_target else "#9ecae1",
        "#bc6c25" if is_target else "#3182bd",
        alpha=0.35 if is_target else 0.16,
        linewidth=2.2 if is_target else 1.2,
        zorder=2,
    )
    mouth = np.asarray(bay.mouth_segment, dtype=np.float64)
    ax.plot(
        mouth[:, 0],
        mouth[:, 1],
        color="#d62828" if is_target else "#3182bd",
        linewidth=3.0 if is_target else 1.3,
        linestyle="--",
        zorder=4,
    )


def _load_agent(checkpoint_path, device):
    agent = ContinuousPPOAgent(device=device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=agent.device,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "network" in checkpoint:
        state_dict = checkpoint["network"]
        extra = dict(checkpoint.get("extra", {}))
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        extra = {}
    else:
        raise ValueError("unsupported checkpoint payload: {}".format(type(checkpoint)))
    agent.network.load_state_dict(state_dict)
    agent.network.eval()
    return agent, extra


def _rs_path_length(path):
    array = np.asarray(path, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1)))


def _fixed_rs_guide_info(path, source, step, reason="success", extra=None):
    info = {
        "source": str(source),
        "step": int(step),
        "reason": str(reason),
        "valid": path is not None,
        "sample_count": 0,
        "total_length": 0.0,
    }
    if path is not None:
        array = np.asarray(path, dtype=np.float64)
        info["sample_count"] = int(array.shape[0])
        info["total_length"] = _rs_path_length(array)
    if extra:
        info.update(extra)
    return info


def _plan_fixed_rs_guide(env, state, step):
    planner = getattr(env.rs_potential, "planner", None)
    if planner is None:
        return None, _fixed_rs_guide_info(
            None,
            "manual",
            step,
            reason="rs_planner_disabled",
        )
    try:
        result = planner.plan(env.scene, state, env.slot)
    except Exception as exc:
        return None, _fixed_rs_guide_info(
            None,
            "manual",
            step,
            reason="planner_exception: {}".format(type(exc).__name__),
        )
    extra = {
        "candidate_count": int(getattr(result, "candidate_count", 0)),
        "checked_candidates": int(getattr(result, "checked_candidates", 0)),
        "collision_checks": int(getattr(result, "collision_checks", 0)),
        "generation_time_ms": float(getattr(result, "generation_time_ms", 0.0)),
        "collision_time_ms": float(getattr(result, "collision_time_ms", 0.0)),
        "plan_time_ms": float(getattr(result, "total_time_ms", 0.0)),
    }
    if not bool(getattr(result, "valid", False)) or getattr(result, "path", None) is None:
        return None, _fixed_rs_guide_info(
            None,
            "manual",
            step,
            reason=str(getattr(result, "reason", "invalid")),
            extra=extra,
        )
    path = np.asarray(result.path, dtype=np.float64).copy()
    return path, _fixed_rs_guide_info(
        path,
        "manual",
        step,
        reason=str(getattr(result, "reason", "success")),
        extra=extra,
    )


def _rollout_path(env, agent, seed, deterministic, rs_guide_step=None):
    observation, reset_info = env.reset(seed=seed)
    states = [env.state]
    scene = env.scene
    slot = env.slot
    total_reward = 0.0
    final_info = dict(reset_info)
    done = False
    rs_guide_path = None
    rs_guide_step_found = -1
    rs_guide_state = None
    rs_guide_info = None
    rs_guide_source = "manual" if rs_guide_step is not None else "latch"
    if rs_guide_step is not None:
        requested_step = int(rs_guide_step)
        if requested_step < 0:
            raise ValueError("--rs-guide-step must be non-negative")
        if requested_step == 0:
            rs_guide_path, rs_guide_info = _plan_fixed_rs_guide(
                env,
                env.state,
                requested_step,
            )
            rs_guide_step_found = requested_step
            rs_guide_state = replace(env.state)
    while not done:
        raw_action, _, _ = agent.act(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(raw_action)
        states.append(env.state)
        total_reward += float(reward)
        final_info = info
        done = bool(terminated or truncated)
        if (
            rs_guide_step is not None
            and rs_guide_info is None
            and env.step_count == int(rs_guide_step)
        ):
            rs_guide_path, rs_guide_info = _plan_fixed_rs_guide(
                env,
                env.state,
                int(rs_guide_step),
            )
            rs_guide_step_found = int(rs_guide_step)
            rs_guide_state = replace(env.state)
        elif (
            rs_guide_step is None
            and rs_guide_path is None
            and getattr(env.rs_potential, "rs_latched", False)
            and getattr(env.rs_potential, "rs_path", None) is not None
        ):
            rs_guide_path = np.asarray(env.rs_potential.rs_path, dtype=np.float64).copy()
            rs_guide_step_found = int(env.step_count)
            rs_guide_state = replace(env.state)
            rs_guide_info = _fixed_rs_guide_info(
                rs_guide_path,
                "latch",
                rs_guide_step_found,
                extra={
                    "candidate_count": int(info.get("rs_candidate_count", 0)),
                    "checked_candidates": int(info.get("rs_checked_candidates", 0)),
                    "collision_checks": int(info.get("rs_collision_checks", 0)),
                    "plan_time_ms": float(info.get("rs_plan_time_ms_mean", 0.0)),
                },
            )
    if rs_guide_info is None:
        if rs_guide_step is None:
            rs_guide_info = _fixed_rs_guide_info(
                None,
                "latch",
                -1,
                reason=str(final_info.get("rs_fail_reason", "not_latched")),
            )
        else:
            rs_guide_info = _fixed_rs_guide_info(
                None,
                "manual",
                int(rs_guide_step),
                reason="step_not_reached",
            )
    return PathRollout(
        seed=int(seed),
        scene=scene,
        slot=slot,
        reset_info=dict(reset_info),
        states=states,
        final_info=dict(final_info),
        total_reward=float(total_reward),
        rs_guide_path=rs_guide_path,
        rs_guide_step=int(rs_guide_step_found),
        rs_guide_state=rs_guide_state,
        rs_guide_info=rs_guide_info,
        rs_guide_source=rs_guide_source,
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
        return "articulation_limit"
    if bool(info.get("timeout", False)):
        return "timeout"
    return "done"


def _front_points(rollout):
    return np.asarray(
        [(state.x_front, state.y_front) for state in rollout.states],
        dtype=np.float64,
    )


def _rear_points(vehicle_model, rollout):
    return np.asarray(
        [vehicle_model.rear_center(state) for state in rollout.states],
        dtype=np.float64,
    )


def _plot_direction_arrow(ax, points, color):
    if len(points) < 2:
        return
    index = max(0, min(len(points) - 2, len(points) // 2))
    delta = points[index + 1] - points[index]
    if float(np.linalg.norm(delta)) <= 1e-8:
        return
    ax.arrow(
        points[index, 0],
        points[index, 1],
        delta[0],
        delta[1],
        color=color,
        width=0.05,
        head_width=0.55,
        length_includes_head=True,
        alpha=0.9,
        zorder=7,
    )


def _plot_rollout(ax, env, rollout, color, index):
    front_points = _front_points(rollout)
    rear_points = _rear_points(env.vehicle_model, rollout)
    status = _status_label(rollout.final_info)
    ax.plot(
        front_points[:, 0],
        front_points[:, 1],
        color=color,
        linewidth=2.2,
        alpha=0.95,
        label="path {} ({})".format(index + 1, status),
        zorder=6,
    )
    ax.plot(
        rear_points[:, 0],
        rear_points[:, 1],
        color=color,
        linewidth=1.2,
        linestyle=":",
        alpha=0.7,
        zorder=5,
    )
    ax.scatter(
        [front_points[0, 0]],
        [front_points[0, 1]],
        color=color,
        marker="o",
        s=36,
        edgecolors="black",
        linewidths=0.5,
        zorder=8,
    )
    ax.scatter(
        [front_points[-1, 0]],
        [front_points[-1, 1]],
        color=color,
        marker="X",
        s=48,
        edgecolors="black",
        linewidths=0.5,
        zorder=8,
    )
    _plot_direction_arrow(ax, front_points, color)

    start_front, start_rear = env.vehicle_model.body_boxes(rollout.states[0])
    end_front, end_rear = env.vehicle_model.body_boxes(rollout.states[-1])
    _plot_polygon_outline(ax, start_rear, color, alpha=0.45, linewidth=1.2)
    _plot_polygon_outline(ax, start_front, color, alpha=0.75, linewidth=1.6)
    _plot_polygon_outline(ax, end_rear, color, alpha=0.55, linewidth=1.2, linestyle="--")
    _plot_polygon_outline(ax, end_front, color, alpha=0.9, linewidth=1.8, linestyle="--")


def _plot_fixed_rs_guide(ax, vehicle_model, rollout):
    if rollout.rs_guide_path is None:
        return False
    path = np.asarray(rollout.rs_guide_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
        return False
    color = "#d00000"
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=2.8,
        linestyle="-.",
        alpha=0.95,
        label="fixed RS guide step {}".format(int(rollout.rs_guide_step)),
        zorder=9,
    )
    ax.scatter(
        [path[0, 0]],
        [path[0, 1]],
        color=color,
        marker="D",
        s=56,
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
    )
    ax.scatter(
        [path[-1, 0]],
        [path[-1, 1]],
        color=color,
        marker="*",
        s=96,
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
    )
    _plot_direction_arrow(ax, path[:, :2], color)
    if rollout.rs_guide_state is not None:
        start_front, start_rear = vehicle_model.body_boxes(rollout.rs_guide_state)
        _plot_polygon_outline(
            ax,
            start_rear,
            color,
            alpha=0.7,
            linewidth=1.4,
            linestyle="-.",
        )
        _plot_polygon_outline(
            ax,
            start_front,
            color,
            alpha=0.95,
            linewidth=1.8,
            linestyle="-.",
        )
    return True


def _plot_scene_and_path(
    env,
    rollout,
    checkpoint_path,
    output,
    stage,
    deterministic,
    path_index,
    total_paths,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    scene = rollout.scene
    slot = rollout.slot
    for obstacle in scene.obstacle_polygons:
        _plot_polygon(ax, obstacle, "#777777", "#555555", alpha=0.95, zorder=1)
    for bay in scene.parking_bays:
        _plot_bay(ax, bay)

    target_front = slot.front_box()
    target_rear = env.vehicle_model.target_rear_box(
        slot.x_goal,
        slot.y_goal,
        slot.theta_goal,
    )
    _plot_polygon(
        ax,
        target_rear,
        "#f4a261",
        "#9c4f15",
        alpha=0.35,
        linewidth=2.0,
        zorder=3,
    )
    _plot_polygon(
        ax,
        target_front,
        "#8fd175",
        "#207020",
        alpha=0.48,
        linewidth=2.4,
        zorder=3,
    )

    cmap = plt.get_cmap("tab20", max(1, total_paths))
    _plot_rollout(ax, env, rollout, cmap(path_index), path_index)
    _plot_fixed_rs_guide(ax, env.vehicle_model, rollout)

    xmin, ymin, xmax, ymax = scene.world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    checkpoint_name = os.path.basename(checkpoint_path)
    policy_mode = "deterministic" if deterministic else "stochastic"
    status = _status_label(rollout.final_info)
    info = rollout.final_info
    dwa_tag = "dwa={} used={} mode={} unlock={}".format(
        int(bool(info.get("dwa_triggered", False))),
        int(bool(info.get("dwa_used", False))),
        str(info.get("dwa_mode", "none")),
        int(bool(info.get("dwa_unlock_success", False))),
    )
    ax.set_title(
        "Local parking policy path {} / {} | stage={} | {} | {} | {}".format(
            path_index + 1,
            total_paths,
            stage,
            status,
            policy_mode,
            dwa_tag,
        )
    )

    rs_info = rollout.rs_guide_info or {}
    summary_lines = [
        "checkpoint: {}".format(checkpoint_name),
        "path={} seed={} {} steps={} overlap={:.2f} reward={:.2f}".format(
            path_index + 1,
            rollout.seed,
            status,
            max(0, len(rollout.states) - 1),
            float(info.get("front_overlap", 0.0)),
            rollout.total_reward,
        )
    ]
    if bool(info.get("dwa_triggered", False)):
        summary_lines.append(
            "dwa: mode={} override={} unlock={} step={} deadlock={} valid={}/{} r={:.3f}".format(
                str(info.get("dwa_mode", "none")),
                int(bool(info.get("dwa_override_policy_action", False))),
                int(bool(info.get("dwa_unlock_success", False))),
                int(info.get("dwa_unlock_step", -1)),
                int(bool(info.get("deadlock", False))),
                int(info.get("dwa_valid_candidate_count", 0)),
                int(info.get("dwa_candidate_count", 0)),
                float(info.get("dwa_final_max_safe_ratio", 0.0)),
            )
        )
    if bool(rs_info.get("valid", False)):
        summary_lines.append(
            "rs_guide: {} step={} len={:.1f} samples={}".format(
                rs_info.get("source", rollout.rs_guide_source),
                int(rs_info.get("step", rollout.rs_guide_step)),
                float(rs_info.get("total_length", 0.0)),
                int(rs_info.get("sample_count", 0)),
            )
        )
    elif rs_info:
        summary_lines.append(
            "rs_guide: {} unavailable ({})".format(
                rs_info.get("source", rollout.rs_guide_source),
                rs_info.get("reason", "unknown"),
            )
        )
    ax.text(
        0.01,
        0.01,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.78},
        zorder=20,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _path_output_path(base_output, path_index, total_paths):
    if int(total_paths) <= 1:
        return base_output
    stem, extension = os.path.splitext(base_output)
    if not extension:
        extension = ".png"
    return "{}_path{:03d}{}".format(stem, path_index + 1, extension)


def _default_output_path(stage, task_family, seed, checkpoint_path):
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    filename = "local_parking_paths_stage{}_{}_seed{}_{}.png".format(
        int(stage),
        str(task_family),
        int(seed),
        checkpoint_stem,
    )
    return os.path.join(REPO_ROOT, "outputs", "paths", filename)


def main():
    parser = argparse.ArgumentParser(
        description="Roll out a PPO checkpoint and render local parking paths."
    )
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument(
        "--task-family",
        choices=["head_in"],
        default="head_in",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a PPO checkpoint .pt file")
    parser.add_argument(
        "--num-paths",
        "--paths",
        type=int,
        default=4,
        help="Number of initial states and policy paths to render",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Scene seed; rollout seeds are seed, seed+1, ...",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic policy mean",
    )
    parser.add_argument(
        "--allow-stage-mismatch",
        action="store_true",
        help="Run even when checkpoint stage differs from --stage",
    )
    parser.add_argument(
        "--rs-guide-step",
        type=int,
        default=None,
        help=(
            "Plan and render one fixed RS guide from this rollout step; "
            "omit to render the actual RS potential latch when it occurs"
        ),
    )
    parser.add_argument(
        "--enable-dwa-recovery",
        action="store_true",
        help="Enable strict-mask DWA recovery diagnostics during rollout",
    )
    parser.add_argument(
        "--dwa-override-policy-action",
        action="store_true",
        help="Allow DWA recovery to replace the policy action",
    )
    parser.add_argument(
        "--dwa-deadlock-termination",
        action="store_true",
        help="Terminate repeated DWA no-candidate deadlocks",
    )
    args = parser.parse_args()

    if args.num_paths <= 0:
        raise ValueError("--num-paths must be positive")
    if args.rs_guide_step is not None and int(args.rs_guide_step) < 0:
        raise ValueError("--rs-guide-step must be non-negative")
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    output = args.output or _default_output_path(
        args.stage,
        args.task_family,
        args.seed,
        checkpoint_path,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    agent, extra = _load_agent(checkpoint_path, args.device)
    checkpoint_stage = extra.get("stage")
    if checkpoint_stage is not None and int(checkpoint_stage) != int(args.stage):
        msg = (
            "checkpoint stage {} differs from requested stage {}".format(
                checkpoint_stage,
                args.stage,
            )
        )
        if not bool(args.allow_stage_mismatch):
            raise SystemExit(msg)
        print("warning: {}".format(msg))

    env_config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=int(args.stage),
        scene_pool_size=1,
        scene_family_schedule=(args.task_family,),
        use_hybrid_astar=False,
        enable_dwa_recovery=bool(args.enable_dwa_recovery),
        dwa_override_policy_action=bool(args.dwa_override_policy_action),
        dwa_enable_deadlock_termination=bool(args.dwa_deadlock_termination),
    )
    env = LocalParkingEnv(config=env_config, seed=int(args.seed))
    deterministic = not bool(args.stochastic)

    rollouts = []
    for index in range(int(args.num_paths)):
        rollout_seed = int(args.seed) + index
        rollout = _rollout_path(
            env,
            agent,
            rollout_seed,
            deterministic,
            rs_guide_step=args.rs_guide_step,
        )
        rollouts.append(rollout)
        rs_info = rollout.rs_guide_info or {}
        if bool(rs_info.get("valid", False)):
            rs_text = "rs_guide={} step={} len={:.2f} samples={}".format(
                rs_info.get("source", rollout.rs_guide_source),
                int(rs_info.get("step", rollout.rs_guide_step)),
                float(rs_info.get("total_length", 0.0)),
                int(rs_info.get("sample_count", 0)),
            )
        else:
            rs_text = "rs_guide={} unavailable reason={}".format(
                rs_info.get("source", rollout.rs_guide_source),
                rs_info.get("reason", "unknown"),
            )
        print(
            "path={} seed={} status={} steps={} reward={:.3f} "
            "dwa_triggered={} dwa_override={} dwa_mode={} unlock={} {}".format(
                index + 1,
                rollout_seed,
                _status_label(rollout.final_info),
                max(0, len(rollout.states) - 1),
                rollout.total_reward,
                int(bool(rollout.final_info.get("dwa_triggered", False))),
                int(bool(rollout.final_info.get("dwa_override_policy_action", False))),
                str(rollout.final_info.get("dwa_mode", "none")),
                int(bool(rollout.final_info.get("dwa_unlock_success", False))),
                rs_text,
            )
        )

    for index, rollout in enumerate(rollouts):
        path_output = _path_output_path(output, index, len(rollouts))
        _plot_scene_and_path(
            env=env,
            rollout=rollout,
            checkpoint_path=checkpoint_path,
            output=path_output,
            stage=int(args.stage),
            deterministic=deterministic,
            path_index=index,
            total_paths=len(rollouts),
        )
        print("saved {}".format(path_output))

    scenarios = {}
    for rollout in rollouts:
        stype = rollout.reset_info.get("scenario_type", "unknown")
        status = _status_label(rollout.final_info)
        if stype not in scenarios:
            scenarios[stype] = {
                "total": 0,
                "success": 0,
                "collision": 0,
                "timeout": 0,
                "deadlock": 0,
            }
        scenarios[stype]["total"] += 1
        if status == "success":
            scenarios[stype]["success"] += 1
        elif status == "collision":
            scenarios[stype]["collision"] += 1
        elif status == "deadlock":
            scenarios[stype]["deadlock"] += 1
        elif status == "timeout":
            scenarios[stype]["timeout"] += 1
    if len(scenarios) > 1:
        print("scenario summary:")
        for stype in sorted(scenarios):
            s = scenarios[stype]
            print(
                "  {}: total={} success={} collision={} timeout={} deadlock={}".format(
                    stype,
                    s["total"],
                    s["success"],
                    s["collision"],
                    s["timeout"],
                    s["deadlock"],
                )
            )


if __name__ == "__main__":
    main()
