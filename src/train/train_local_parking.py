import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_MASK_CONFIG,
    DEFAULT_PPO_CONFIG,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_VEHICLE_PARAMS,
)
from env.local_parking_env import LocalParkingEnv
from env.mixing_plant_scene import (
    SUPPORTED_SCENE_TYPES,
    TASK_FAMILIES as SCENE_TASK_FAMILIES,
    normalize_family_schedule,
)
from env.vehicle import ArticulatedState
from model.continuous_ppo import ContinuousPPOAgent, RolloutBuffer
from planning.passenger_hybrid_astar import PassengerHybridAStar
from train.curriculum import (
    CurriculumStageSelector,
    MultiStageScenePool,
    UniformStageSelector,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK_FAMILIES = SCENE_TASK_FAMILIES
MIN_EVAL_EPISODES_PER_FAMILY = 20


def _add_config_bool_argument(parser, name, default, help_enable, help_disable):
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    disable_name = name[len("enable-") :] if name.startswith("enable-") else name
    group.add_argument(
        "--{}".format(name),
        dest=dest,
        action="store_true",
        default=bool(default),
        help=help_enable,
    )
    group.add_argument(
        "--disable-{}".format(disable_name),
        dest=dest,
        action="store_false",
        help=help_disable,
    )


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _linear_schedule_value(initial, final, start_episode, end_episode, episode):
    initial = float(initial)
    final = float(final)
    start_episode = int(start_episode)
    end_episode = int(end_episode)
    episode = int(episode)
    if end_episode <= start_episode:
        return final if episode >= end_episode else initial
    if episode <= start_episode:
        return initial
    if episode >= end_episode:
        return final
    progress = (episode - start_episode) / float(end_episode - start_episode)
    return float(initial + progress * (final - initial))


def _conditional_rate(items, value_key, condition_key):
    selected = [
        float(item.get(value_key, False))
        for item in items
        if bool(item.get(condition_key, False))
    ]
    return _safe_mean(selected)


def _task_family(reset_info, scene_metadata):
    task_family = str(reset_info.get("task_family", ""))
    if task_family in TASK_FAMILIES:
        return task_family
    mode = str(reset_info.get("goal_orientation_mode", ""))
    if mode == "head_in":
        return "head_in"
    raise ValueError("unsupported goal orientation mode: {}".format(mode))


def _scene_type_key(reset_info, scene_metadata):
    requested = str(
        reset_info.get(
            "requested_scene_type",
            scene_metadata.get("requested_scene_type", ""),
        )
    )
    if requested:
        return requested
    actual = str(reset_info.get("scene_type", scene_metadata.get("scene_type", "")))
    if actual:
        return actual
    return str(DEFAULT_SCENE_CONFIG.scene_type)


def _parse_family_schedule(value):
    return normalize_family_schedule(value)


def _parse_float_tuple(value):
    if isinstance(value, (tuple, list)):
        items = value
    else:
        items = str(value).split(",")
    result = tuple(float(item) for item in items if str(item).strip())
    if not result:
        raise ValueError("expected at least one numeric value")
    return result


def _success_by_family(completed):
    result = {}
    for family in TASK_FAMILIES:
        selected = [
            float(item["success"])
            for item in completed
            if str(item.get("task_family", "")) == family
        ]
        result[family] = _safe_mean(selected)
        result["{}_count".format(family)] = len(selected)
    return result


def _eval_family_value(evaluation, eval_mode, policy_mode, family):
    mode_block = evaluation.get(eval_mode, {})
    if isinstance(mode_block, dict):
        policy_block = mode_block.get(policy_mode, {})
        if isinstance(policy_block, dict) and family in policy_block:
            return float(policy_block[family])
    policy_block = evaluation.get(policy_mode, {})
    if isinstance(policy_block, dict) and family in policy_block:
        return float(policy_block[family])
    return 0.0


def _eval_mode_success_mean(evaluation, eval_mode, policy_mode="deterministic"):
    values = [
        _eval_family_value(evaluation, eval_mode, policy_mode, family)
        for family in TASK_FAMILIES
    ]
    return _safe_mean(values)


def _write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _rs_metrics_by_mode(infos):
    grouped = {}
    for mode in ("head_in",):
        selected = [
            item
            for item in infos
            if str(item.get("goal_orientation_mode", "")) == mode
        ]
        if not selected:
            continue
        grouped[mode] = {
            "step_count": len(selected),
            "latched_rate": _safe_mean(
                [float(item.get("rs_latched", False)) for item in selected]
            ),
            "valid_rate": _safe_mean(
                [float(item.get("rs_valid_rate", 0.0)) for item in selected]
            ),
            "reward_mean": _safe_mean(
                [float(item.get("rs_reward", 0.0)) for item in selected]
            ),
            "plan_time_ms_mean": _safe_mean(
                [float(item.get("rs_plan_time_ms_mean", 0.0)) for item in selected]
            ),
            "plan_time_ms_max": max(
                float(item.get("rs_plan_time_ms_max", 0.0)) for item in selected
            ),
        }
    return grouped


def _copy_articulated_state(state):
    return ArticulatedState(
        x_front=float(state.x_front),
        y_front=float(state.y_front),
        theta_front=float(state.theta_front),
        theta_rear=float(state.theta_rear),
        v=0.0,
        phi_dot=0.0,
    )


class HardCaseReplayBuffer:
    def __init__(self, capacity, tail_steps, replay_ratio, rng):
        self.capacity = max(0, int(capacity))
        self.tail_steps = max(1, int(tail_steps))
        self.replay_ratio = float(np.clip(replay_ratio, 0.0, 1.0))
        self.rng = rng
        self._entries = []
        self._next_index = 0

    def __len__(self):
        return len(self._entries)

    def _append(self, entry):
        if self.capacity <= 0:
            return
        if len(self._entries) < self.capacity:
            self._entries.append(entry)
            return
        self._entries[self._next_index] = entry
        self._next_index = (self._next_index + 1) % self.capacity

    def record_failure(
        self,
        scene,
        slot,
        tail_states,
        final_info,
        reset_info,
        stage,
        episode_index,
    ):
        if self.capacity <= 0:
            return 0
        if bool(final_info.get("success", False)):
            return 0
        if bool(final_info.get("rs_latched", False)):
            return 0
        collision = bool(final_info.get("collision", False))
        timeout = bool(final_info.get("timeout", False))
        deadlock = bool(final_info.get("deadlock", False))
        if not (collision or timeout or deadlock):
            return 0
        selected_states = list(tail_states)[-self.tail_steps:]
        if not selected_states:
            return 0
        failure_type = str(final_info.get("failure_type", ""))
        if not failure_type:
            failure_type = "collision" if collision else ("deadlock" if deadlock else "timeout")
        count = 0
        for state in selected_states:
            self._append(
                {
                    "scene": scene,
                    "slot": slot,
                    "state": _copy_articulated_state(state),
                    "stage": int(stage),
                    "episode": int(episode_index),
                    "scene_seed": int(reset_info.get("scene_seed", -1)),
                    "scenario_type": str(reset_info.get("scenario_type", "")),
                    "task_family": str(reset_info.get("task_family", "")),
                    "failure_type": failure_type,
                }
            )
            count += 1
        return count

    def sample(self):
        if not self._entries:
            return None
        if self.replay_ratio <= 0.0:
            return None
        if float(self.rng.random()) >= self.replay_ratio:
            return None
        index = int(self.rng.integers(0, len(self._entries)))
        return self._entries[index]


def _resolve_output_dir(output_dir, seed, timestamp=None):
    if output_dir:
        return os.path.abspath(output_dir)
    run_time = timestamp or datetime.now()
    run_name = "local_parking_{}_seed{}".format(
        run_time.strftime("%Y%m%d_%H%M%S"),
        int(seed),
    )
    return os.path.join(REPO_ROOT, "runs", run_name)


def _write_config_snapshot(
    path,
    args,
    scene_config_or_env_config,
    env_config_or_ppo_config,
    ppo_config=None,
):
    if ppo_config is None:
        scene_config = DEFAULT_SCENE_CONFIG
        env_config = scene_config_or_env_config
        ppo_config = env_config_or_ppo_config
    else:
        scene_config = scene_config_or_env_config
        env_config = env_config_or_ppo_config
    sections = (
        ("training_arguments", vars(args)),
        ("vehicle", asdict(DEFAULT_VEHICLE_PARAMS)),
        ("action_mask", asdict(DEFAULT_MASK_CONFIG)),
        ("scene", asdict(scene_config)),
        ("environment", asdict(env_config)),
        ("ppo", asdict(ppo_config)),
    )
    with open(path, "w", encoding="utf-8") as handle:
        for section_name, values in sections:
            handle.write("[{}]\n".format(section_name))
            for key in sorted(values):
                handle.write("{} = {}\n".format(key, repr(values[key])))
            handle.write("\n")


def _update_reward_plot(path, episode_rewards):
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.asarray(episode_rewards, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, rewards, color="#2b6cb0", linewidth=1.2, label="Episode reward")
    if len(rewards) >= 10:
        kernel = np.ones(10, dtype=np.float64) / 10.0
        moving_average = np.convolve(rewards, kernel, mode="valid")
        ax.plot(
            episodes[9:],
            moving_average,
            color="#c53030",
            linewidth=2.0,
            label="10-episode mean",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Local Parking Training Reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    temporary_path = path + ".tmp.png"
    fig.savefig(temporary_path, dpi=150)
    plt.close(fig)
    os.replace(temporary_path, path)


def _build_update_record(
    buffer,
    infos,
    completed,
    update_stats,
    global_step,
    episode_index,
    update_index,
    start_time,
    global_log_std,
    gear_deadband,
    latest_evaluation,
):
    pre_tanh_actions = np.asarray(buffer.pre_tanh_actions)
    raw_actions = np.asarray(buffer.raw_actions)
    executed_actions = np.asarray(buffer.executed_actions)
    family_success = _success_by_family(completed)
    no_guide_success = _eval_mode_success_mean(latest_evaluation, "no_guide")
    guided_success = _eval_mode_success_mean(latest_evaluation, "guided")
    dwa_triggered_infos = [
        item for item in infos if bool(item.get("dwa_triggered", False))
    ]
    failure_types = sorted(
        set(str(item.get("failure_type", "")) for item in completed)
    )
    return {
        "global_step": global_step,
        "episode": episode_index,
        "update": update_index,
        "rollout_size": len(buffer),
        "steps_per_second": global_step / max(time.perf_counter() - start_time, 1e-6),
        **update_stats,
        "success/head_in": family_success["head_in"],
        "episodes/head_in": family_success["head_in_count"],
        "global_log_std": np.asarray(global_log_std, dtype=np.float32).tolist(),
        "pre_tanh_abs_mean": float(np.mean(np.abs(pre_tanh_actions))),
        "raw_action_saturation_rate": float(
            np.mean(np.abs(raw_actions) > 0.95)
        ),
        "speed_deadband_rate": float(
            np.mean(np.abs(raw_actions[:, 0]) < float(gear_deadband))
        ),
        "gear_switch_count": int(
            sum(int(item.get("gear_switch_count", 0)) for item in completed)
        ),
        "deterministic_eval_success_by_family": dict(
            latest_evaluation.get("deterministic", {})
        ),
        "stochastic_eval_success_by_family": dict(
            latest_evaluation.get("stochastic", {})
        ),
        "weighted_score": float(latest_evaluation.get("weighted_score", 0.0)),
        "checkpoint_selection_score": float(
            latest_evaluation.get("checkpoint_selection_score", 0.0)
        ),
        "stage3_no_latch_success": float(
            latest_evaluation.get("stage3_no_latch_success", 0.0)
        ),
        "stage4_recovery_success": float(
            latest_evaluation.get("stage4_recovery_success", 0.0)
        ),
        "eval_collision_rate": float(latest_evaluation.get("collision_rate", 0.0)),
        "eval_timeout_rate": float(latest_evaluation.get("timeout_rate", 0.0)),
        "eval_deadlock_rate": float(latest_evaluation.get("deadlock_rate", 0.0)),
        "dwa_assisted_eval_success_rate": float(
            latest_evaluation.get("dwa_assisted_success_rate", 0.0)
        ),
        "raw_action_mean": raw_actions.mean(axis=0).tolist(),
        "raw_action_std": raw_actions.std(axis=0).tolist(),
        "executed_action_mean": executed_actions.mean(axis=0).tolist(),
        "executed_action_std": executed_actions.std(axis=0).tolist(),
        "speed_clip_rate": _safe_mean([item["speed_clip_rate"] for item in infos]),
        "mask_invalid_rate": _safe_mean([item["mask_invalid_rate"] for item in infos]),
        "mask_zero_fraction": _safe_mean([item["mask_zero_fraction"] for item in infos]),
        "mask_safe_ratio_mean": _safe_mean([item["mask_safe_ratio"] for item in infos]),
        "mask_safe_ratio_min": float(min(item["mask_safe_ratio"] for item in infos)),
        "front_overlap": _safe_mean([item["front_overlap"] for item in infos]),
        "best_front_overlap": _safe_mean(
            [item["best_front_overlap"] for item in infos]
        ),
        "rear_body_overlap": _safe_mean(
            [item["rear_body_overlap"] for item in infos]
        ),
        "heading_error_deg": _safe_mean(
            [item["heading_error_deg"] for item in infos]
        ),
        "rear_heading_error_deg": _safe_mean(
            [item["rear_heading_error_deg"] for item in infos]
        ),
        "distance_to_goal": _safe_mean(
            [item["distance_to_goal"] for item in infos]
        ),
        "phi_mean": _safe_mean([item["phi"] for item in infos]),
        "phi_abs_mean": _safe_mean([abs(item["phi"]) for item in infos]),
        "min_lidar_distance": float(
            min(item["min_lidar_distance"] for item in infos)
        ),
        "success_rate": _safe_mean([float(item["success"]) for item in completed]),
        "guided_success_rate": guided_success,
        "no_guide_success_rate": no_guide_success,
        "success_gap_guided_minus_noguide": guided_success - no_guide_success,
        "collision_rate": _safe_mean(
            [float(item["collision"]) for item in completed]
        ),
        "timeout_rate": _safe_mean([float(item["timeout"]) for item in completed]),
        "deadlock_rate": _safe_mean(
            [float(item.get("deadlock", False)) for item in completed]
        ),
        "failure_type_distribution": dict(
            (
                failure_type,
                sum(
                    1
                    for item in completed
                    if str(item.get("failure_type", "")) == failure_type
                ),
            )
            for failure_type in failure_types
            if failure_type
        ),
        "hybrid_astar_valid_rate": _safe_mean(
            [item["hybrid_astar_valid_rate"] for item in infos]
        ),
        "hybrid_astar_episode_valid_rate": _safe_mean(
            [float(item.get("hybrid_astar_valid_rate", 0.0)) for item in completed]
        ),
        "planner_valid_rate": _safe_mean(
            [float(item.get("planner_valid", False)) for item in infos]
        ),
        "planner_cost_mean": _safe_mean(
            [float(item.get("planner_cost", 0.0)) for item in infos]
        ),
        "planner_potential_reward_mean": _safe_mean(
            [float(item.get("planner_potential_reward", 0.0)) for item in infos]
        ),
        "planner_fallback_used_rate": _safe_mean(
            [float(item.get("planner_fallback_used", False)) for item in infos]
        ),
        "rs_attempt_count": sum(
            int(item.get("rs_attempt_count", 0)) for item in completed
        ),
        "rs_success_count": sum(
            int(item.get("rs_success_count", 0)) for item in completed
        ),
        "rs_latched_rate": _safe_mean(
            [float(item.get("rs_latched", False)) for item in infos]
        ),
        "rs_valid_rate": _safe_mean(
            [float(item.get("rs_valid_rate", 0.0)) for item in completed]
        ),
        "rs_plan_time_ms_mean": _safe_mean(
            [float(item.get("rs_plan_time_ms_mean", 0.0)) for item in completed]
        ),
        "rs_plan_time_ms_max": max(
            [float(item.get("rs_plan_time_ms_max", 0.0)) for item in completed]
            or [0.0]
        ),
        "rs_reward_mean": _safe_mean(
            [float(item.get("rs_reward", 0.0)) for item in infos]
        ),
        "rs_cost_mean": _safe_mean(
            [float(item.get("rs_cost", 0.0)) for item in infos]
        ),
        "rs_remaining_length_mean": _safe_mean(
            [float(item.get("rs_remaining_length", 0.0)) for item in infos]
        ),
        "rs_projection_error_mean": _safe_mean(
            [float(item.get("rs_projection_error", 0.0)) for item in infos]
        ),
        "rs_heading_error_mean": _safe_mean(
            [float(item.get("rs_heading_error", 0.0)) for item in infos]
        ),
        "rs_fail_reasons": dict(
            (reason, sum(
                1
                for item in completed
                if str(item.get("rs_fail_reason", "")) == reason
            ))
            for reason in sorted(
                set(str(item.get("rs_fail_reason", "")) for item in completed)
            )
            if reason
        ),
        "rs_by_goal_orientation_mode": _rs_metrics_by_mode(infos),
        "forced_stop_rate": _safe_mean(
            [float(item.get("forced_stop", False)) for item in infos]
        ),
        "dwa_trigger_rate": _safe_mean(
            [float(item.get("dwa_triggered", False)) for item in infos]
        ),
        "dwa_used_rate": _safe_mean(
            [float(item.get("dwa_used", False)) for item in infos]
        ),
        "dwa_override_policy_action_rate": _safe_mean(
            [float(item.get("dwa_override_policy_action", False)) for item in infos]
        ),
        "dwa_teacher_action_valid_rate": _safe_mean(
            [float(item.get("dwa_teacher_action_valid", False)) for item in infos]
        ),
        "dwa_policy_loss_weight_mean": _safe_mean(
            [float(item.get("dwa_policy_loss_weight", 1.0)) for item in infos]
        ),
        "dwa_unlock_success_rate": _conditional_rate(
            infos,
            "dwa_unlock_success",
            "dwa_triggered",
        ),
        "dwa_deadlock_rate": _conditional_rate(
            infos,
            "dwa_deadlock",
            "dwa_triggered",
        ),
        "dwa_valid_candidate_count_mean": _safe_mean(
            [
                float(item.get("dwa_valid_candidate_count", 0))
                for item in dwa_triggered_infos
            ]
        ),
        "dwa_final_max_safe_ratio_mean": _safe_mean(
            [
                float(item.get("dwa_final_max_safe_ratio", 0.0))
                for item in dwa_triggered_infos
            ]
        ),
        "normal_mask_max_mean": _safe_mean(
            [float(item.get("normal_mask_max", 0.0)) for item in infos]
        ),
        "recovery_mask_applied_rate": _safe_mean(
            [float(item.get("recovery_mask_applied", False)) for item in infos]
        ),
        "recovery_mask_nonzero_count_mean": _safe_mean(
            [float(item.get("recovery_mask_nonzero_count", 0)) for item in infos]
        ),
        "recovery_mask_max_mean": _safe_mean(
            [float(item.get("recovery_mask_max", 0.0)) for item in infos]
        ),
        "effective_mask_max_mean": _safe_mean(
            [float(item.get("effective_mask_max", 0.0)) for item in infos]
        ),
        "degenerate_mask_rate": _safe_mean(
            [float(item.get("degenerate_mask", False)) for item in infos]
        ),
        "initial_degenerate_mask_rate": _safe_mean(
            [float(item.get("initial_degenerate_mask", False)) for item in completed]
        ),
        "mask_floor_applied_rate": _safe_mean(
            [float(item.get("mask_floor_applied", False)) for item in infos]
        ),
        "mean_mask_max_before_floor": _safe_mean(
            [float(item.get("mask_max_before_floor", 0.0)) for item in infos]
        ),
        "collision_after_mask_floor_rate": _conditional_rate(
            infos,
            "collision_after_mask_floor",
            "mask_floor_applied",
        ),
        "success_after_mask_floor_rate": _conditional_rate(
            infos,
            "success_after_mask_floor",
            "mask_floor_applied",
        ),
        "episode_mask_floor_applied_rate": _safe_mean(
            [float(item.get("episode_mask_floor_applied", False)) for item in completed]
        ),
        "episode_collision_after_mask_floor_rate": _conditional_rate(
            completed,
            "episode_collision_after_mask_floor",
            "episode_mask_floor_applied",
        ),
        "episode_success_after_mask_floor_rate": _conditional_rate(
            completed,
            "episode_success_after_mask_floor",
            "episode_mask_floor_applied",
        ),
        "low_safe_abs_rate": _safe_mean(
            [float(item.get("raw_safe_ratio", 0.0) < 0.15) for item in infos]
        ),
        "mean_raw_safe_ratio": _safe_mean(
            [item.get("raw_safe_ratio", 0.0) for item in infos]
        ),
        "mean_max_safe_ratio": _safe_mean(
            [item.get("max_safe_ratio", 0.0) for item in infos]
        ),
        "clip_rate": _safe_mean(
            [float(item.get("clip_ratio", 0.0) > 0.01) for item in infos]
        ),
        "mask_cost_mean": _safe_mean(
            [float(item.get("mask_cost", 0.0)) for item in infos]
        ),
        "gear_switch_rate": float(
            sum(int(item.get("gear_switch_count", 0)) for item in completed)
            / max(len(infos), 1)
        ),
        "mask_loss_ratio_abs": float(
            abs(update_stats.get("aux_mask_loss", 0.0))
            / max(abs(update_stats.get("policy_loss", 0.0)), 1e-8)
        ),
        "hope_teacher_enabled": _safe_mean(
            [float(item.get("hope_teacher_enabled", False)) for item in completed]
        ),
        "hope_teacher_available": _safe_mean(
            [float(item.get("hope_teacher_available", False)) for item in completed]
        ),
        "hope_plan_success": _safe_mean(
            [float(item.get("hope_plan_success", False)) for item in completed]
        ),
        "hope_cache_hit_rate": _safe_mean(
            [float(item.get("hope_cache_hit", False)) for item in completed]
        ),
        "hope_path_length": _safe_mean(
            [float(item.get("hope_path_length", 0.0)) for item in completed]
        ),
        "hope_path_valid_after_articulated_check": _safe_mean(
            [
                float(item.get("hope_path_valid_after_articulated_check", False))
                for item in completed
            ]
        ),
        "hope_plan_fail_reasons": dict(
            (reason, sum(
                1
                for item in completed
                if str(item.get("hope_plan_fail_reason", "")) == reason
            ))
            for reason in sorted(
                set(str(item.get("hope_plan_fail_reason", "")) for item in completed)
            )
            if reason
        ),
        "guide_weight_current": _safe_mean(
            [float(item.get("guide_weight_current", 0.0)) for item in completed]
        ),
        "guide_dropout_rate": _safe_mean(
            [float(item.get("guide_dropout_rate", 0.0)) for item in completed]
        ),
        "guide_reward_mean": _safe_mean(
            [float(item.get("guide_reward", 0.0)) for item in infos]
        ),
        "guide_progress_reward_mean": _safe_mean(
            [float(item.get("guide_progress_reward", 0.0)) for item in infos]
        ),
        "guide_lateral_error_mean": _safe_mean(
            [float(item.get("guide_lateral_error", 0.0)) for item in infos]
        ),
        "guide_heading_error_mean": _safe_mean(
            [float(item.get("guide_heading_error", 0.0)) for item in infos]
        ),
        "guide_anchor_error_mean": _safe_mean(
            [float(item.get("guide_anchor_error", 0.0)) for item in infos]
        ),
        "guide_gear_agreement_rate": _safe_mean(
            [float(item.get("guide_gear_agreement", 0.0)) for item in infos]
        ),
        "large_heading_error_rate": _safe_mean(
            [
                float(float(item.get("heading_error_deg", 0.0)) >= 45.0)
                for item in completed
            ]
        ),
        "large_articulation_error_rate": _safe_mean(
            [
                float(abs(float(item.get("phi", 0.0))) >= np.deg2rad(20.0))
                for item in completed
            ]
        ),
    }


def _write_tensorboard_update(writer, record):
    if writer is None:
        return
    episode = record["episode"]
    for key, value in record.items():
        if isinstance(value, (int, float)):
            writer.add_scalar("update/{}".format(key), value, episode)
    for field in (
        "global_log_std",
        "raw_action_mean",
        "raw_action_std",
        "executed_action_mean",
        "executed_action_std",
    ):
        for index, value in enumerate(record[field]):
            writer.add_scalar(
                "update/{}/{}".format(field, index),
                value,
                episode,
            )
    for mode in ("deterministic", "stochastic"):
        field = "{}_eval_success_by_family".format(mode)
        for family, value in record.get(field, {}).items():
            writer.add_scalar(
                "eval/{}/{}".format(mode, family),
                value,
                episode,
            )


def _weighted_checkpoint_score(success_by_family, ppo_config):
    weights = {
        "head_in": float(ppo_config.checkpoint_score_weight_head_in),
    }
    denominator = sum(weights.values())
    if denominator <= 0.0:
        raise ValueError("checkpoint score weights must sum to a positive value")
    return float(
        sum(
            weights[family] * float(success_by_family[family])
            for family in TASK_FAMILIES
        )
        / denominator
    )


def _summarize_eval_outcomes(outcomes):
    outcomes = list(outcomes)
    no_latch = [
        item for item in outcomes if not bool(item.get("rs_latched", False))
    ]
    dwa_triggered = [
        item for item in outcomes if bool(item.get("dwa_triggered", False))
    ]
    failure_types = sorted(
        set(str(item.get("failure_type", "")) for item in outcomes)
    )
    scenario_success = {}
    scenario_counts = {}
    for scenario in sorted(set(str(item.get("scenario_type", "")) for item in outcomes)):
        selected = [
            item
            for item in outcomes
            if str(item.get("scenario_type", "")) == scenario
        ]
        if not selected:
            continue
        scenario_success[scenario] = _safe_mean(
            [float(item.get("success", False)) for item in selected]
        )
        scenario_counts[scenario] = len(selected)
    scene_type_success = {}
    scene_type_counts = {}
    scene_type_no_latch_success = {}
    scene_type_scenario_success = {}
    for scene_type in sorted(set(str(item.get("scene_type", "")) for item in outcomes)):
        selected = [
            item
            for item in outcomes
            if str(item.get("scene_type", "")) == scene_type
        ]
        if not selected:
            continue
        selected_no_latch = [
            item for item in selected if not bool(item.get("rs_latched", False))
        ]
        scene_type_success[scene_type] = _safe_mean(
            [float(item.get("success", False)) for item in selected]
        )
        scene_type_counts[scene_type] = len(selected)
        scene_type_no_latch_success[scene_type] = _safe_mean(
            [float(item.get("success", False)) for item in selected_no_latch]
        )
        scenario_rates = {}
        for scenario in sorted(set(str(item.get("scenario_type", "")) for item in selected)):
            scenario_selected = [
                item
                for item in selected
                if str(item.get("scenario_type", "")) == scenario
            ]
            if scenario_selected:
                scenario_rates[scenario] = _safe_mean(
                    [float(item.get("success", False)) for item in scenario_selected]
                )
        scene_type_scenario_success[scene_type] = scenario_rates
    return {
        "episode_count": len(outcomes),
        "success_rate": _safe_mean(
            [float(item.get("success", False)) for item in outcomes]
        ),
        "scene_type_equal_success_rate": _safe_mean(
            [float(value) for value in scene_type_success.values()]
        ),
        "collision_rate": _safe_mean(
            [float(item.get("collision", False)) for item in outcomes]
        ),
        "timeout_rate": _safe_mean(
            [float(item.get("timeout", False)) for item in outcomes]
        ),
        "deadlock_rate": _safe_mean(
            [float(item.get("deadlock", False)) for item in outcomes]
        ),
        "failure_type_distribution": dict(
            (
                failure_type,
                sum(
                    1
                    for item in outcomes
                    if str(item.get("failure_type", "")) == failure_type
                ),
            )
            for failure_type in failure_types
            if failure_type
        ),
        "dwa_trigger_rate": _safe_mean(
            [float(item.get("dwa_triggered", False)) for item in outcomes]
        ),
        "dwa_used_rate": _safe_mean(
            [float(item.get("dwa_used", False)) for item in outcomes]
        ),
        "dwa_unlock_success_rate": _conditional_rate(
            outcomes,
            "dwa_unlock_success",
            "dwa_triggered",
        ),
        "dwa_deadlock_rate": _conditional_rate(
            outcomes,
            "dwa_deadlock",
            "dwa_triggered",
        ),
        "dwa_valid_candidate_count_mean": _safe_mean(
            [
                float(item.get("dwa_valid_candidate_count", 0))
                for item in dwa_triggered
            ]
        ),
        "dwa_final_max_safe_ratio_mean": _safe_mean(
            [
                float(item.get("dwa_final_max_safe_ratio", 0.0))
                for item in dwa_triggered
            ]
        ),
        "rs_latched_rate": _safe_mean(
            [float(item.get("rs_latched", False)) for item in outcomes]
        ),
        "no_latch_count": len(no_latch),
        "no_latch_success_rate": _safe_mean(
            [float(item.get("success", False)) for item in no_latch]
        ),
        "scene_type_success": scene_type_success,
        "scene_type_counts": scene_type_counts,
        "scene_type_no_latch_success": scene_type_no_latch_success,
        "scene_type_scenario_success": scene_type_scenario_success,
        "scenario_success": scenario_success,
        "scenario_counts": scenario_counts,
    }


def _aggregate_success_by_family(stage_results, policy_mode):
    aggregate = {}
    for family in TASK_FAMILIES:
        aggregate[family] = _safe_mean(
            [
                float(result.get(policy_mode, {}).get(family, 0.0))
                for result in stage_results.values()
            ]
        )
    return aggregate


def _aggregate_success_by_scene_type(stage_results, policy_mode):
    key = "{}_by_scene_type".format(policy_mode)
    scene_types = sorted(
        {
            str(scene_type)
            for result in stage_results.values()
            for scene_type in result.get(key, {}).keys()
        }
    )
    return dict(
        (
            scene_type,
            _safe_mean(
                [
                    float(result.get(key, {}).get(scene_type, 0.0))
                    for result in stage_results.values()
                ]
            ),
        )
        for scene_type in scene_types
    )


def _scene_type_equal_from_summary(summary, per_scene_key, fallback_key):
    per_scene = summary.get(per_scene_key, {})
    if isinstance(per_scene, dict) and per_scene:
        return _safe_mean([float(value) for value in per_scene.values()])
    return float(summary.get(fallback_key, 0.0))


def _scene_type_equal_scenario_from_summary(summary, scenario, fallback_key):
    scene_type_scenario = summary.get("scene_type_scenario_success", {})
    scene_type_success = summary.get("scene_type_success", {})
    values = []
    if isinstance(scene_type_scenario, dict):
        for scene_type in sorted(scene_type_scenario.keys()):
            scenario_rates = scene_type_scenario.get(scene_type, {})
            if isinstance(scenario_rates, dict) and scenario in scenario_rates:
                values.append(float(scenario_rates[scenario]))
            elif isinstance(scene_type_success, dict) and scene_type in scene_type_success:
                values.append(float(scene_type_success[scene_type]))
    if values:
        return _safe_mean(values)
    return float(summary.get(fallback_key, 0.0))


def _aggregate_eval_summaries(stage_results, policy_mode):
    summary_key = "{}_summary".format(policy_mode)
    summaries = [
        result.get(summary_key, {})
        for result in stage_results.values()
        if isinstance(result.get(summary_key, {}), dict)
    ]
    scene_types = sorted(
        {
            str(scene_type)
            for summary in summaries
            for scene_type in summary.get("scene_type_success", {}).keys()
        }
    )
    scene_type_success = dict(
        (
            scene_type,
            _safe_mean(
                [
                    float(summary.get("scene_type_success", {}).get(scene_type, 0.0))
                    for summary in summaries
                ]
            ),
        )
        for scene_type in scene_types
    )
    scene_type_no_latch_success = dict(
        (
            scene_type,
            _safe_mean(
                [
                    float(
                        summary.get("scene_type_no_latch_success", {}).get(
                            scene_type,
                            0.0,
                        )
                    )
                    for summary in summaries
                ]
            ),
        )
        for scene_type in scene_types
    )
    return {
        "episode_count": int(sum(int(item.get("episode_count", 0)) for item in summaries)),
        "success_rate": _safe_mean(
            [float(item.get("success_rate", 0.0)) for item in summaries]
        ),
        "scene_type_equal_success_rate": _safe_mean(
            [float(value) for value in scene_type_success.values()]
        ),
        "collision_rate": _safe_mean(
            [float(item.get("collision_rate", 0.0)) for item in summaries]
        ),
        "timeout_rate": _safe_mean(
            [float(item.get("timeout_rate", 0.0)) for item in summaries]
        ),
        "deadlock_rate": _safe_mean(
            [float(item.get("deadlock_rate", 0.0)) for item in summaries]
        ),
        "dwa_trigger_rate": _safe_mean(
            [float(item.get("dwa_trigger_rate", 0.0)) for item in summaries]
        ),
        "dwa_used_rate": _safe_mean(
            [float(item.get("dwa_used_rate", 0.0)) for item in summaries]
        ),
        "rs_latched_rate": _safe_mean(
            [float(item.get("rs_latched_rate", 0.0)) for item in summaries]
        ),
        "no_latch_success_rate": _safe_mean(
            [float(item.get("no_latch_success_rate", 0.0)) for item in summaries]
        ),
        "scene_type_success": scene_type_success,
        "scene_type_no_latch_success": scene_type_no_latch_success,
    }


def _checkpoint_selection_score(evaluation):
    stage3 = float(
        evaluation.get(
            "stage3_scene_type_equal_no_latch_success",
            evaluation.get("stage3_no_latch_success", 0.0),
        )
    )
    stage4 = float(
        evaluation.get(
            "stage4_scene_type_equal_recovery_success",
            evaluation.get("stage4_recovery_success", 0.0),
        )
    )
    return min(stage3, stage4)


def _evaluate_policy_by_family(
    agent,
    env_config,
    scene_config,
    stage,
    seed,
    episodes_per_family,
    eval_modes=("no_guide",),
    scene_type_schedule=None,
):
    episodes_per_family = int(episodes_per_family)
    if episodes_per_family < MIN_EVAL_EPISODES_PER_FAMILY:
        raise ValueError(
            "episodes_per_family must be at least {}".format(
                MIN_EVAL_EPISODES_PER_FAMILY
            )
        )
    eval_modes = tuple(eval_modes or ("no_guide",))
    scene_type_schedule = tuple(scene_type_schedule or ())
    if not scene_type_schedule:
        scene_type_schedule = (
            str(getattr(scene_config, "scene_type", DEFAULT_SCENE_CONFIG.scene_type)),
        )
    eval_pool_size = max(1, int(env_config.scene_pool_size))
    schedule_size = max(
        1,
        len(tuple(env_config.scene_family_schedule)) * len(scene_type_schedule),
    )
    if schedule_size > 1:
        remainder = eval_pool_size % schedule_size
        if remainder:
            eval_pool_size += schedule_size - remainder
    base_seed = ((int(seed) + 100_000) // eval_pool_size) * eval_pool_size
    cuda_devices = []
    if agent.device.type == "cuda":
        cuda_devices = [
            agent.device.index
            if agent.device.index is not None
            else torch.cuda.current_device()
        ]

    def make_eval_config(eval_mode):
        enable_teacher = eval_mode == "guided" and bool(env_config.enable_hope_teacher)
        use_teacher_reward = enable_teacher and bool(env_config.use_teacher_reward)
        enable_dwa = eval_mode == "dwa_assisted_eval"
        return replace(
            env_config,
            curriculum_stage=int(stage),
            use_hybrid_astar=False,
            rs_potential_enabled=False,
            scene_family_schedule=TASK_FAMILIES,
            enable_hope_teacher=enable_teacher,
            use_teacher_reward=use_teacher_reward,
            enable_offpath_reset=False,
            enable_failure_aggregation=False,
            enable_dwa_recovery=bool(enable_dwa),
            dwa_recovery_mode="teacher_override",
            dwa_override_policy_action=bool(enable_dwa),
        )

    def run_mode(eval_mode):
        eval_config = make_eval_config(eval_mode)
        mode_results = {}
        for deterministic in (True, False):
            multi_pool = None
            if len(scene_type_schedule) > 1:
                multi_pool = MultiStageScenePool(
                    pool_size=eval_pool_size,
                    base_seed=base_seed,
                    scene_config=scene_config,
                    family_schedule=TASK_FAMILIES,
                    scene_type_schedule=scene_type_schedule,
                )
            env = LocalParkingEnv(
                config=eval_config,
                scene_config=scene_config,
                seed=base_seed,
                multi_stage_pool=multi_pool,
            )
            if multi_pool is not None:
                env.set_active_stage(stage)
            group_keys = [
                (scene_type, family)
                for scene_type in scene_type_schedule
                for family in TASK_FAMILIES
            ]
            outcomes_by_group = dict((key, []) for key in group_keys)
            while min(len(values) for values in outcomes_by_group.values()) < episodes_per_family:
                observation, reset_info = env.reset()
                family = _task_family(reset_info, env.scene.metadata)
                scene_type = _scene_type_key(reset_info, env.scene.metadata)
                group_key = (scene_type, family)
                if group_key not in outcomes_by_group:
                    raise RuntimeError(
                        "unexpected eval group scene_type={} family={}; expected {}".format(
                            scene_type,
                            family,
                            group_keys,
                        )
                    )
                if len(outcomes_by_group[group_key]) >= episodes_per_family:
                    continue
                done = False
                final_info = None
                while not done:
                    raw_action, _, _ = agent.act(
                        observation,
                        deterministic=deterministic,
                    )
                    observation, _, terminated, truncated, final_info = env.step(
                        raw_action
                    )
                    done = terminated or truncated
                outcomes_by_group[group_key].append(
                    {
                        "success": bool(final_info["success"]),
                        "collision": bool(final_info["collision"]),
                        "timeout": bool(final_info["timeout"]),
                        "deadlock": bool(final_info.get("deadlock", False)),
                        "failure_type": str(final_info.get("failure_type", "")),
                        "rs_latched": bool(final_info.get("rs_latched", False)),
                        "dwa_triggered": bool(final_info.get("dwa_triggered", False)),
                        "dwa_used": bool(final_info.get("dwa_used", False)),
                        "dwa_unlock_success": bool(
                            final_info.get("dwa_unlock_success", False)
                        ),
                        "dwa_unlock_step": int(final_info.get("dwa_unlock_step", -1)),
                        "dwa_deadlock": bool(final_info.get("dwa_deadlock", False)),
                        "dwa_valid_candidate_count": int(
                            final_info.get("dwa_valid_candidate_count", 0)
                        ),
                        "dwa_final_max_safe_ratio": float(
                            final_info.get("dwa_final_max_safe_ratio", 0.0)
                        ),
                        "scenario_type": str(reset_info.get("scenario_type", "")),
                        "scene_type": str(scene_type),
                        "requested_scene_type": str(
                            reset_info.get("requested_scene_type", scene_type)
                        ),
                        "task_family": family,
                    }
                )
            mode = "deterministic" if deterministic else "stochastic"
            mode_results[mode] = dict(
                (
                    family,
                    _safe_mean(
                        [
                            _safe_mean(
                                [
                                    float(item.get("success", False))
                                    for item in outcomes_by_group.get(
                                        (scene_type, family),
                                        (),
                                    )
                                ]
                            )
                            for scene_type in scene_type_schedule
                        ]
                    ),
                )
                for family in TASK_FAMILIES
            )
            mode_results["{}_by_scene_type".format(mode)] = dict(
                (
                    scene_type,
                    _safe_mean(
                        [
                            _safe_mean(
                                [
                                    float(item.get("success", False))
                                    for item in outcomes_by_group.get(
                                        (scene_type, family),
                                        (),
                                    )
                                ]
                            )
                            for family in TASK_FAMILIES
                        ]
                    ),
                )
                for scene_type in scene_type_schedule
            )
            mode_results["{}_by_scene_type_family".format(mode)] = dict(
                (
                    "{}|{}".format(scene_type, family),
                    _safe_mean(
                        [
                            float(item.get("success", False))
                            for item in outcomes_by_group.get((scene_type, family), ())
                        ]
                    ),
                )
                for scene_type in scene_type_schedule
                for family in TASK_FAMILIES
            )
            all_outcomes = []
            for scene_type, family in group_keys:
                all_outcomes.extend(outcomes_by_group.get((scene_type, family), ()))
            mode_results["{}_summary".format(mode)] = _summarize_eval_outcomes(
                all_outcomes
            )
        return mode_results

    grouped_results = {}
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(base_seed + 7)
        if agent.device.type == "cuda":
            torch.cuda.manual_seed_all(base_seed + 7)
        for eval_mode in eval_modes:
            grouped_results[eval_mode] = run_mode(eval_mode)

    primary_mode = "no_guide" if "no_guide" in grouped_results else eval_modes[0]
    results = dict(grouped_results[primary_mode])
    results.update(grouped_results)
    results["weighted_score"] = _weighted_checkpoint_score(
        results["deterministic"],
        agent.config,
    )
    results["guided_success_rate"] = _eval_mode_success_mean(results, "guided")
    results["no_guide_success_rate"] = _eval_mode_success_mean(results, "no_guide")
    results["dwa_assisted_success_rate"] = _eval_mode_success_mean(
        results,
        "dwa_assisted_eval",
    )
    results["success_gap_guided_minus_noguide"] = (
        results["guided_success_rate"] - results["no_guide_success_rate"]
    )
    deterministic_summary = results.get("deterministic_summary", {})
    results["collision_rate"] = float(deterministic_summary.get("collision_rate", 0.0))
    results["timeout_rate"] = float(deterministic_summary.get("timeout_rate", 0.0))
    results["deadlock_rate"] = float(deterministic_summary.get("deadlock_rate", 0.0))
    results["stage"] = int(stage)
    results["episodes_per_family"] = episodes_per_family
    results["eval_modes"] = list(eval_modes)
    results["scene_type_schedule"] = list(scene_type_schedule)
    return results


def _evaluate_policy_across_stages(
    agent,
    env_config,
    scene_config,
    stages,
    seed,
    episodes_per_family,
    eval_modes=("no_guide",),
    scene_type_schedule=None,
):
    scene_type_schedule = tuple(scene_type_schedule or ())
    if not scene_type_schedule:
        scene_type_schedule = (
            str(getattr(scene_config, "scene_type", DEFAULT_SCENE_CONFIG.scene_type)),
        )
    stage_results = {}
    for stage in stages:
        result = _evaluate_policy_by_family(
            agent=agent,
            env_config=env_config,
            scene_config=scene_config,
            stage=int(stage),
            seed=int(seed) + 10_000 * int(stage),
            episodes_per_family=episodes_per_family,
            eval_modes=eval_modes,
            scene_type_schedule=scene_type_schedule,
        )
        stage_results[str(int(stage))] = result

    primary_mode = "no_guide" if "no_guide" in eval_modes else tuple(eval_modes)[0]
    primary_stage_results = {}
    for stage_key, result in stage_results.items():
        primary_stage_results[stage_key] = result

    aggregate = {
        "stages": stage_results,
        "eval_stages": [int(stage) for stage in stages],
        "episodes_per_family": int(episodes_per_family),
        "eval_modes": list(eval_modes),
        "scene_type_schedule": list(scene_type_schedule),
        "deterministic": _aggregate_success_by_family(
            primary_stage_results,
            "deterministic",
        ),
        "stochastic": _aggregate_success_by_family(
            primary_stage_results,
            "stochastic",
        ),
        "deterministic_by_scene_type": _aggregate_success_by_scene_type(
            primary_stage_results,
            "deterministic",
        ),
        "stochastic_by_scene_type": _aggregate_success_by_scene_type(
            primary_stage_results,
            "stochastic",
        ),
        "deterministic_summary": _aggregate_eval_summaries(
            primary_stage_results,
            "deterministic",
        ),
        "stochastic_summary": _aggregate_eval_summaries(
            primary_stage_results,
            "stochastic",
        ),
    }
    aggregate[primary_mode] = {
        "deterministic": dict(aggregate["deterministic"]),
        "stochastic": dict(aggregate["stochastic"]),
        "deterministic_by_scene_type": dict(aggregate["deterministic_by_scene_type"]),
        "stochastic_by_scene_type": dict(aggregate["stochastic_by_scene_type"]),
        "deterministic_summary": dict(aggregate["deterministic_summary"]),
        "stochastic_summary": dict(aggregate["stochastic_summary"]),
    }
    for eval_mode in eval_modes:
        if eval_mode == primary_mode:
            continue
        eval_stage_results = {
            stage: result[eval_mode] if eval_mode in result else result
            for stage, result in stage_results.items()
        }
        aggregate[eval_mode] = {
            "deterministic": _aggregate_success_by_family(
                eval_stage_results,
                "deterministic",
            ),
            "stochastic": _aggregate_success_by_family(
                eval_stage_results,
                "stochastic",
            ),
            "deterministic_by_scene_type": _aggregate_success_by_scene_type(
                eval_stage_results,
                "deterministic",
            ),
            "stochastic_by_scene_type": _aggregate_success_by_scene_type(
                eval_stage_results,
                "stochastic",
            ),
            "deterministic_summary": _aggregate_eval_summaries(
                eval_stage_results,
                "deterministic",
            ),
            "stochastic_summary": _aggregate_eval_summaries(
                eval_stage_results,
                "stochastic",
            ),
        }

    stage3 = stage_results.get("3", {})
    stage4 = stage_results.get("4", {})
    stage3_summary = stage3.get("deterministic_summary", {})
    stage4_summary = stage4.get("deterministic_summary", {})
    aggregate["stage3_success"] = _scene_type_equal_from_summary(
        stage3_summary,
        "scene_type_success",
        "success_rate",
    )
    aggregate["stage4_success"] = _scene_type_equal_from_summary(
        stage4_summary,
        "scene_type_success",
        "success_rate",
    )
    aggregate["stage3_scene_type_equal_no_latch_success"] = (
        _scene_type_equal_from_summary(
            stage3_summary,
            "scene_type_no_latch_success",
            "no_latch_success_rate",
        )
    )
    aggregate["stage4_scene_type_equal_recovery_success"] = (
        _scene_type_equal_scenario_from_summary(
            stage4_summary,
            "recovery",
            "success_rate",
        )
    )
    aggregate["stage3_no_latch_success"] = float(
        aggregate["stage3_scene_type_equal_no_latch_success"]
    )
    aggregate["stage4_recovery_success"] = float(
        aggregate["stage4_scene_type_equal_recovery_success"]
    )
    aggregate["collision_rate"] = float(
        aggregate["deterministic_summary"].get("collision_rate", 0.0)
    )
    aggregate["timeout_rate"] = float(
        aggregate["deterministic_summary"].get("timeout_rate", 0.0)
    )
    aggregate["deadlock_rate"] = float(
        aggregate["deterministic_summary"].get("deadlock_rate", 0.0)
    )
    aggregate["family_weighted_score"] = _weighted_checkpoint_score(
        aggregate["deterministic"],
        agent.config,
    )
    aggregate["checkpoint_selection_score"] = _checkpoint_selection_score(aggregate)
    aggregate["weighted_score"] = float(aggregate["checkpoint_selection_score"])
    aggregate["guided_success_rate"] = _eval_mode_success_mean(aggregate, "guided")
    aggregate["no_guide_success_rate"] = _eval_mode_success_mean(aggregate, "no_guide")
    aggregate["dwa_assisted_success_rate"] = _eval_mode_success_mean(
        aggregate,
        "dwa_assisted_eval",
    )
    aggregate["success_gap_guided_minus_noguide"] = (
        aggregate["guided_success_rate"] - aggregate["no_guide_success_rate"]
    )
    return aggregate


def _save_best_checkpoints(
    agent,
    output_dir,
    evaluation,
    best_scores,
    checkpoint_extra,
):
    metrics = dict(evaluation["deterministic"])
    metrics["weighted_score"] = float(evaluation["weighted_score"])
    metrics["checkpoint_selection_score"] = float(
        evaluation.get("checkpoint_selection_score", evaluation["weighted_score"])
    )
    metrics["stage3_no_latch_success"] = float(
        evaluation.get("stage3_no_latch_success", 0.0)
    )
    metrics["stage4_recovery_success"] = float(
        evaluation.get("stage4_recovery_success", 0.0)
    )
    metrics["stage3_scene_type_equal_no_latch_success"] = float(
        evaluation.get("stage3_scene_type_equal_no_latch_success", 0.0)
    )
    metrics["stage4_scene_type_equal_recovery_success"] = float(
        evaluation.get("stage4_scene_type_equal_recovery_success", 0.0)
    )
    for name, score in metrics.items():
        if name not in best_scores:
            best_scores[name] = -float("inf")
        if float(score) <= float(best_scores[name]):
            continue
        best_scores[name] = float(score)
        extra = dict(checkpoint_extra)
        extra.update(
            {
                "best_metric": name,
                "best_score": float(score),
                "evaluation": dict(evaluation),
            }
        )
        agent.save(
            os.path.join(output_dir, "checkpoint_best_{}.pt".format(name)),
            extra=extra,
        )


def train(args):
    if args.total_episodes <= 0:
        raise ValueError("--total-episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    if args.eval_episodes_per_family < MIN_EVAL_EPISODES_PER_FAMILY:
        raise ValueError(
            "--eval-episodes-per-family must be at least {}".format(
                MIN_EVAL_EPISODES_PER_FAMILY
            )
        )
    if args.ppo_epochs <= 0:
        raise ValueError("--ppo-epochs must be positive")
    if not 0.0 < args.clip_range < 1.0:
        raise ValueError("--clip-range must be inside (0, 1)")
    if args.actor_lr <= 0.0 or args.critic_lr <= 0.0:
        raise ValueError("actor and critic learning rates must be positive")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    policy_weights = (
        args.policy_loss_weight_head_in,
    )
    if any(float(weight) < 0.0 for weight in policy_weights):
        raise ValueError("policy loss weights must be non-negative")
    if args.dwa_bc_coef_initial < 0.0 or args.dwa_bc_coef_final < 0.0:
        raise ValueError("DWA BC coefficients must be non-negative")
    if args.dwa_bc_anneal_end_episode < args.dwa_bc_anneal_start_episode:
        raise ValueError("--dwa-bc-anneal-end-episode must be >= start episode")
    family_schedule = _parse_family_schedule(args.scene_family_schedule)
    if args.guide_anneal_end_episode < args.guide_anneal_start_episode:
        raise ValueError("--guide-anneal-end-episode must be >= start episode")
    if not 0.0 <= args.guide_dropout_initial <= 1.0:
        raise ValueError("--guide-dropout-initial must be inside [0, 1]")
    if not 0.0 <= args.guide_dropout_final <= 1.0:
        raise ValueError("--guide-dropout-final must be inside [0, 1]")
    if args.teacher_corridor_width <= 0.0:
        raise ValueError("--teacher-corridor-width must be positive")
    if args.teacher_reward_clip <= 0.0:
        raise ValueError("--teacher-reward-clip must be positive")
    if args.no_guide_eval_interval < 0:
        raise ValueError("--no-guide-eval-interval must be non-negative")
    if not 0.0 <= args.hard_case_replay_ratio <= 1.0:
        raise ValueError("--hard-case-replay-ratio must be inside [0, 1]")
    if args.hard_case_replay_capacity < 0:
        raise ValueError("--hard-case-replay-capacity must be non-negative")
    if args.hard_case_replay_tail_steps <= 0:
        raise ValueError("--hard-case-replay-tail-steps must be positive")
    if args.hard_case_replay_attempts <= 0:
        raise ValueError("--hard-case-replay-attempts must be positive")
    if args.hard_case_replay_xy_std < 0.0:
        raise ValueError("--hard-case-replay-xy-std must be non-negative")
    if args.hard_case_replay_heading_std_deg < 0.0:
        raise ValueError("--hard-case-replay-heading-std-deg must be non-negative")
    if args.hard_case_replay_phi_std_deg < 0.0:
        raise ValueError("--hard-case-replay-phi-std-deg must be non-negative")
    if args.scene_pool_size <= 0:
        raise ValueError("--scene-pool-size must be positive")
    if args.mask_cost_coef_final < 0.0:
        raise ValueError("--mask-cost-coef-final must be non-negative")
    if args.mask_degenerate_eps < 0.0:
        raise ValueError("--mask-degenerate-eps must be non-negative")
    if not 0.0 < args.mask_floor_value <= 1.0:
        raise ValueError("--mask-floor-value must be inside (0, 1]")
    if args.dwa_all_zero_eps < 0.0:
        raise ValueError("--dwa-all-zero-eps must be non-negative")
    if args.dwa_low_safe_ratio < 0.0:
        raise ValueError("--dwa-low-safe-ratio must be non-negative")
    if args.dwa_unlock_safe_ratio < 0.0:
        raise ValueError("--dwa-unlock-safe-ratio must be non-negative")
    if args.dwa_unlock_min_safe_ratio_improvement < 0.0:
        raise ValueError("--dwa-unlock-min-safe-ratio-improvement must be non-negative")
    if not 0.0 <= args.dwa_override_policy_loss_weight <= 1.0:
        raise ValueError("--dwa-override-policy-loss-weight must be inside [0, 1]")
    if args.dwa_forced_stop_patience < 0:
        raise ValueError("--dwa-forced-stop-patience must be non-negative")
    if args.dwa_no_progress_patience < 0:
        raise ValueError("--dwa-no-progress-patience must be non-negative")
    if args.dwa_deadlock_patience <= 0:
        raise ValueError("--dwa-deadlock-patience must be positive")
    if args.dwa_horizon_steps <= 0:
        raise ValueError("--dwa-horizon-steps must be positive")
    dwa_speed_ratios = _parse_float_tuple(args.dwa_speed_ratios)
    if any(float(ratio) <= 0.0 or float(ratio) > 1.0 for ratio in dwa_speed_ratios):
        raise ValueError("--dwa-speed-ratios entries must be inside (0, 1]")
    if not 0.0 < args.dwa_recovery_max_speed_ratio <= 1.0:
        raise ValueError("--dwa-recovery-max-speed-ratio must be inside (0, 1]")
    if args.dwa_recovery_phi_bin_radius < 0:
        raise ValueError("--dwa-recovery-phi-bin-radius must be non-negative")
    dwa_unlock_speed_ratios = _parse_float_tuple(args.dwa_unlock_speed_ratios)
    if any(
        float(ratio) <= 0.0 or float(ratio) > args.dwa_recovery_max_speed_ratio
        for ratio in dwa_unlock_speed_ratios
    ):
        raise ValueError(
            "--dwa-unlock-speed-ratios entries must be inside "
            "(0, --dwa-recovery-max-speed-ratio]"
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scene_config = replace(
        DEFAULT_SCENE_CONFIG,
        scene_type=str(args.scene_type),
    )
    curriculum_scene_types = (
        tuple(SUPPORTED_SCENE_TYPES)
        if bool(args.curriculum)
        else (str(args.scene_type),)
    )
    args.curriculum_scene_types = ",".join(curriculum_scene_types)
    env_config = replace(
        DEFAULT_ENV_CONFIG,
        max_steps=args.max_steps,
        curriculum_stage=args.stage,
        scene_pool_size=args.scene_pool_size,
        scene_family_schedule=family_schedule,
        use_hybrid_astar=bool(args.use_hybrid_astar),
        rs_potential_enabled=bool(args.enable_rs_potential),
        enable_hope_teacher=bool(args.enable_hope_teacher),
        hope_code_dir=args.hope_code_dir,
        hope_weight_path=args.hope_weight_path,
        hope_cache_dir=args.hope_cache_dir,
        use_teacher_reward=bool(args.use_teacher_reward),
        guide_weight_initial=args.guide_weight_initial,
        guide_weight_final=args.guide_weight_final,
        guide_anneal_start_episode=args.guide_anneal_start_episode,
        guide_anneal_end_episode=args.guide_anneal_end_episode,
        guide_dropout_initial=args.guide_dropout_initial,
        guide_dropout_final=args.guide_dropout_final,
        teacher_corridor_width=args.teacher_corridor_width,
        teacher_anchor_weight=args.teacher_anchor_weight,
        teacher_heading_weight=args.teacher_heading_weight,
        teacher_progress_weight=args.teacher_progress_weight,
        teacher_gear_weight=args.teacher_gear_weight,
        teacher_reward_clip=args.teacher_reward_clip,
        enable_offpath_reset=bool(args.enable_offpath_reset),
        enable_failure_aggregation=bool(args.enable_failure_aggregation),
        no_guide_eval_interval=args.no_guide_eval_interval,
        hard_case_replay_enabled=not bool(args.disable_hard_case_replay),
        hard_case_replay_ratio=args.hard_case_replay_ratio,
        hard_case_replay_capacity=args.hard_case_replay_capacity,
        hard_case_replay_tail_steps=args.hard_case_replay_tail_steps,
        hard_case_replay_attempts=args.hard_case_replay_attempts,
        hard_case_replay_xy_std=args.hard_case_replay_xy_std,
        hard_case_replay_heading_std_deg=args.hard_case_replay_heading_std_deg,
        hard_case_replay_phi_std_deg=args.hard_case_replay_phi_std_deg,
        mask_cost_coef_final=args.mask_cost_coef_final,
        disable_mask_observation=bool(args.disable_mask_observation),
        rear_lidar_observation_mode=str(args.rear_lidar_observation_mode),
        disable_action_mask_execution=bool(args.disable_action_mask_execution),
        enable_mask_floor_fallback=bool(args.enable_mask_floor_fallback)
        and not bool(args.disable_mask_floor_fallback),
        mask_degenerate_eps=args.mask_degenerate_eps,
        mask_floor_value=args.mask_floor_value,
        apply_floor_only_when_all_zero=bool(args.apply_floor_only_when_all_zero),
        enable_dwa_recovery=bool(args.enable_dwa_recovery),
        dwa_recovery_mode=str(args.dwa_recovery_mode),
        dwa_override_policy_action=bool(args.dwa_override_policy_action),
        dwa_override_policy_loss_weight=args.dwa_override_policy_loss_weight,
        dwa_enable_deadlock_termination=bool(args.dwa_deadlock_termination),
        dwa_all_zero_eps=args.dwa_all_zero_eps,
        dwa_low_safe_ratio=args.dwa_low_safe_ratio,
        dwa_unlock_safe_ratio=args.dwa_unlock_safe_ratio,
        dwa_unlock_min_safe_ratio_improvement=args.dwa_unlock_min_safe_ratio_improvement,
        dwa_forced_stop_patience=args.dwa_forced_stop_patience,
        dwa_no_progress_patience=args.dwa_no_progress_patience,
        dwa_deadlock_patience=args.dwa_deadlock_patience,
        dwa_horizon_steps=args.dwa_horizon_steps,
        dwa_speed_ratios=dwa_speed_ratios,
        dwa_unlock_speed_ratios=dwa_unlock_speed_ratios,
        dwa_recovery_max_speed_ratio=args.dwa_recovery_max_speed_ratio,
        dwa_recovery_phi_bin_radius=args.dwa_recovery_phi_bin_radius,
    )
    ppo_config = replace(
        DEFAULT_PPO_CONFIG,
        rollout_steps=args.rollout_steps,
        batch_size=min(args.batch_size, args.rollout_steps),
        clip_range=args.clip_range,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        target_kl=args.target_kl,
        kl_early_stop_multiplier=args.kl_early_stop_multiplier,
        log_std_init=args.log_std_init,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        policy_loss_weight_head_in=args.policy_loss_weight_head_in,
        checkpoint_score_weight_head_in=args.checkpoint_score_weight_head_in,
        dwa_bc_coef_initial=args.dwa_bc_coef_initial,
        dwa_bc_coef_final=args.dwa_bc_coef_final,
        dwa_bc_anneal_start_episode=args.dwa_bc_anneal_start_episode,
        dwa_bc_anneal_end_episode=args.dwa_bc_anneal_end_episode,
    )
    output_dir = _resolve_output_dir(args.output_dir, args.seed)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    _write_config_snapshot(
        os.path.join(output_dir, "config.txt"),
        args,
        scene_config,
        env_config,
        ppo_config,
    )

    curriculum = bool(args.curriculum) and str(args.curriculum_mode) != "fixed"
    stage_selector = None
    multi_pool = None
    if curriculum:
        multi_pool = MultiStageScenePool(
            pool_size=env_config.scene_pool_size,
            base_seed=args.seed,
            scene_config=scene_config,
            family_schedule=env_config.scene_family_schedule,
            scene_type_schedule=curriculum_scene_types,
        )
        if str(args.curriculum_mode) == "uniform":
            stage_selector = UniformStageSelector(seed=int(args.seed) + 3571)
        else:
            stage_selector = CurriculumStageSelector(
                target_success_rate=float(args.curriculum_target_success),
                seed=int(args.seed) + 3571,
            )
        actual_stage = stage_selector.select_stage(0)
    else:
        actual_stage = args.stage

    planner = PassengerHybridAStar() if args.use_hybrid_astar else None
    env = LocalParkingEnv(
        config=env_config,
        hybrid_planner=planner,
        scene_config=scene_config,
        seed=args.seed,
        multi_stage_pool=multi_pool,
    )
    if curriculum:
        env.set_active_stage(actual_stage)
    agent = ContinuousPPOAgent(config=ppo_config, device=args.device)

    if env.hybrid_reward.planner is not None:
        env.hybrid_reward._gamma = float(ppo_config.gamma)
    env.rs_potential.gamma = float(ppo_config.gamma)

    update_jsonl_path = os.path.join(output_dir, "training_metrics.jsonl")
    episode_jsonl_path = os.path.join(output_dir, "episode_metrics.jsonl")
    evaluation_jsonl_path = os.path.join(output_dir, "evaluation_metrics.jsonl")
    reward_plot_path = os.path.join(output_dir, "reward_curve.png")
    writer = (
        SummaryWriter(os.path.join(output_dir, "tensorboard"))
        if SummaryWriter is not None
        else None
    )

    observation, reset_info = env.reset(seed=args.seed)
    global_step = 0
    episode_index = 0
    update_index = 0
    last_done = False
    episode_rewards = []
    buffer = RolloutBuffer()
    rollout_infos = []
    rollout_completed = []
    latest_evaluation = {}
    best_scores = dict((name, -float("inf")) for name in TASK_FAMILIES)
    best_scores["weighted_score"] = -float("inf")
    best_scores["checkpoint_selection_score"] = -float("inf")
    best_scores["stage3_no_latch_success"] = -float("inf")
    best_scores["stage4_recovery_success"] = -float("inf")
    hard_case_replay = HardCaseReplayBuffer(
        capacity=env_config.hard_case_replay_capacity
        if bool(env_config.hard_case_replay_enabled)
        else 0,
        tail_steps=env_config.hard_case_replay_tail_steps,
        replay_ratio=env_config.hard_case_replay_ratio
        if bool(env_config.hard_case_replay_enabled)
        else 0.0,
        rng=np.random.default_rng(int(args.seed) + 97_531),
    )
    start_time = time.perf_counter()

    while episode_index < args.total_episodes:
        mask_coef_final = float(env_config.mask_cost_coef_final)
        if episode_index < 20:
            current_mask_coef = 0.0
        elif episode_index < 220:
            progress = (episode_index - 20) / 200.0
            current_mask_coef = mask_coef_final * progress
        else:
            current_mask_coef = mask_coef_final
        current_dwa_bc_coef = _linear_schedule_value(
            ppo_config.dwa_bc_coef_initial,
            ppo_config.dwa_bc_coef_final,
            ppo_config.dwa_bc_anneal_start_episode,
            ppo_config.dwa_bc_anneal_end_episode,
            episode_index,
        )

        episode_reward = 0.0
        episode_steps = 0
        episode_gear_switch_count = 0
        previous_motion_gear = None
        final_info = None
        done = False
        episode_tail_states = []
        episode_degenerate_mask = False
        episode_mask_floor_applied = False
        episode_collision_after_mask_floor = False
        episode_success_after_mask_floor = False
        task_family = _task_family(reset_info, env.scene.metadata)
        while not done:
            raw_action, pre_tanh_action, log_prob, value = agent.act_with_pre_tanh(
                observation
            )
            next_observation, reward, terminated, truncated, info = env.step(raw_action)
            done = terminated or truncated
            current_gear = int(info.get("gear", -1))
            if (
                previous_motion_gear in (0, 1)
                and current_gear in (0, 1)
                and current_gear != previous_motion_gear
            ):
                episode_gear_switch_count += 1
            if current_gear in (0, 1):
                previous_motion_gear = current_gear
            buffer.add(
                observation=observation,
                raw_action=raw_action,
                executed_action=info["executed_action"],
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                pre_tanh_action=pre_tanh_action,
                mask_cost=float(info.get("mask_cost", 0.0)),
                task_family=task_family,
                dwa_raw_action=info.get("dwa_raw_action", None),
                dwa_teacher_action_valid=bool(
                    info.get("dwa_teacher_action_valid", False)
                ),
                dwa_used=bool(info.get("dwa_used", False)),
                dwa_policy_loss_weight=float(
                    info.get("dwa_policy_loss_weight", 1.0)
                ),
                recovery_mask_applied=bool(
                    info.get("recovery_mask_applied", False)
                ),
                recovery_mask_nonzero_count=int(
                    info.get("recovery_mask_nonzero_count", 0)
                ),
                recovery_mask_max=float(info.get("recovery_mask_max", 0.0)),
            )
            info["task_family"] = task_family
            episode_degenerate_mask = bool(
                episode_degenerate_mask or info.get("degenerate_mask", False)
            )
            episode_mask_floor_applied = bool(
                episode_mask_floor_applied or info.get("mask_floor_applied", False)
            )
            episode_collision_after_mask_floor = bool(
                episode_collision_after_mask_floor
                or info.get("collision_after_mask_floor", False)
            )
            episode_success_after_mask_floor = bool(
                episode_success_after_mask_floor
                or info.get("success_after_mask_floor", False)
            )
            rollout_infos.append(info)
            observation = next_observation
            global_step += 1
            episode_steps += 1
            episode_reward += float(reward)
            final_info = info
            episode_tail_states.append(_copy_articulated_state(env.state))
            if len(episode_tail_states) > env_config.hard_case_replay_tail_steps:
                episode_tail_states.pop(0)
            last_done = done

        episode_index += 1
        episode_rewards.append(episode_reward)
        final_info["task_family"] = task_family
        final_info["gear_switch_count"] = int(episode_gear_switch_count)
        final_info["episode_degenerate_mask"] = bool(episode_degenerate_mask)
        final_info["episode_mask_floor_applied"] = bool(episode_mask_floor_applied)
        final_info["episode_collision_after_mask_floor"] = bool(
            episode_mask_floor_applied and final_info.get("collision", False)
        )
        final_info["episode_success_after_mask_floor"] = bool(
            episode_mask_floor_applied and final_info.get("success", False)
        )
        final_info["step_collision_after_mask_floor"] = bool(
            episode_collision_after_mask_floor
        )
        final_info["step_success_after_mask_floor"] = bool(
            episode_success_after_mask_floor
        )
        final_info["initial_degenerate_mask"] = bool(
            reset_info.get("initial_degenerate_mask", False)
        )
        final_info["initial_mask_floor_applied"] = bool(
            reset_info.get("initial_mask_floor_applied", False)
        )
        final_info["initial_mask_max_before_floor"] = float(
            reset_info.get("initial_mask_max_before_floor", 0.0)
        )
        rollout_completed.append(final_info)
        hard_case_recorded_count = hard_case_replay.record_failure(
            scene=env.scene,
            slot=env.slot,
            tail_states=episode_tail_states,
            final_info=final_info,
            reset_info=reset_info,
            stage=actual_stage,
            episode_index=episode_index,
        )
        episode_record = {
            "episode": episode_index,
            "global_step": global_step,
            "episode_steps": episode_steps,
            "episode_reward": episode_reward,
            "success": bool(final_info["success"]),
            "collision": bool(final_info["collision"]),
            "timeout": bool(final_info["timeout"]),
            "deadlock": bool(final_info.get("deadlock", False)),
            "failure_type": str(final_info.get("failure_type", "")),
            "front_overlap": float(final_info["front_overlap"]),
            "best_front_overlap": float(final_info["best_front_overlap"]),
            "heading_error_deg": float(final_info["heading_error_deg"]),
            "distance_to_goal": float(final_info["distance_to_goal"]),
            "scenario_type": reset_info.get("scenario_type", ""),
            "scene_seed": int(reset_info.get("scene_seed", -1)),
            "scene_type": str(reset_info.get("scene_type", "")),
            "requested_scene_type": str(
                reset_info.get("requested_scene_type", reset_info.get("scene_type", ""))
            ),
            "goal_orientation_mode": str(reset_info.get("goal_orientation_mode", "")),
            "task_family": task_family,
            "bay_count": int(reset_info.get("bay_count", 0)),
            "bay_width": float(reset_info.get("bay_width", 0.0)),
            "bay_depth": float(reset_info.get("bay_depth", 0.0)),
            "corridor_width": float(reset_info.get("corridor_width", 0.0)),
            "target_bay_index": int(reset_info.get("target_bay_index", -1)),
            "initial_bay_index": int(reset_info.get("initial_bay_index", -1)),
            "initial_spawn_region": str(reset_info.get("initial_spawn_region", "")),
            "corridor_outer_wall_exists": bool(
                reset_info.get("corridor_outer_wall_exists", False)
            ),
            "reset_feasible_mask_available": bool(
                reset_info.get("reset_feasible_mask_available", False)
            ),
            "world_length": float(reset_info.get("world_length", 0.0)),
            "world_width": float(reset_info.get("world_width", 0.0)),
            "truck_in_front": bool(reset_info.get("truck_in_front", False)),
            "truck_perpendicular": bool(
                reset_info.get("truck_perpendicular", False)
            ),
            "discrete_obstacle_count": int(
                reset_info.get("discrete_obstacle_count", 0)
            ),
            "obstacle_count": int(reset_info.get("obstacle_count", 0)),
            "obstacle_exclusion_valid": bool(
                reset_info.get("obstacle_exclusion_valid", False)
            ),
            "clearance_bucket": str(reset_info.get("clearance_bucket", "")),
            "approach_side_bucket": str(reset_info.get("approach_side_bucket", "")),
            "scene_complexity_bucket": str(
                reset_info.get("scene_complexity_bucket", "")
            ),
            "difficulty_label": str(reset_info.get("difficulty_label", "")),
            "hard_case_replay_attempted": bool(
                reset_info.get("hard_case_replay_attempted", False)
            ),
            "hard_case_replay_used": bool(
                reset_info.get("hard_case_replay_used", False)
            ),
            "hard_case_replay_source_episode": int(
                reset_info.get("hard_case_replay_source_episode", -1)
            ),
            "hard_case_replay_source_stage": int(
                reset_info.get("hard_case_replay_source_stage", -1)
            ),
            "hard_case_replay_source_failure": str(
                reset_info.get("hard_case_replay_source_failure", "")
            ),
            "hard_case_replay_recorded_count": int(hard_case_recorded_count),
            "hard_case_replay_buffer_size": int(len(hard_case_replay)),
            "nominal_target_collision": bool(
                reset_info.get("nominal_target_collision", False)
            ),
            "nominal_target_clearance_m": float(
                reset_info.get("nominal_target_clearance_m", 0.0)
            ),
            "success_neighborhood_feasible_count": int(
                reset_info.get("success_neighborhood_feasible_count", 0)
            ),
            "constructed_obstacle_feature_count": int(
                reset_info.get("constructed_obstacle_feature_count", 0)
            ),
            "constructed_wall_feature_count": int(
                reset_info.get("constructed_wall_feature_count", 0)
            ),
            "scene_generation_attempt_count": int(
                reset_info.get("scene_generation_attempt_count", 1)
            ),
            "scene_generation_attempts": int(
                reset_info.get(
                    "scene_generation_attempts",
                    reset_info.get("scene_generation_attempt_count", 1),
                )
            ),
            "initial_distance_min": float(
                reset_info.get("initial_distance_min", 0.0)
            ),
            "initial_distance_max": float(
                reset_info.get("initial_distance_max", 0.0)
            ),
            "initial_lateral_range": float(
                reset_info.get("initial_lateral_range", 0.0)
            ),
            "initial_heading_range_deg": float(
                reset_info.get("initial_heading_range_deg", 0.0)
            ),
            "initial_phi_range_deg": float(
                reset_info.get("initial_phi_range_deg", 0.0)
            ),
            "reset_initial_mask_max": float(
                reset_info.get("reset_initial_mask_max", 0.0)
            ),
            "initial_degenerate_mask": bool(
                reset_info.get("initial_degenerate_mask", False)
            ),
            "initial_mask_floor_applied": bool(
                reset_info.get("initial_mask_floor_applied", False)
            ),
            "initial_mask_max_before_floor": float(
                reset_info.get("initial_mask_max_before_floor", 0.0)
            ),
            "reset_initial_mask_degenerate": bool(
                reset_info.get("reset_initial_mask_degenerate", False)
            ),
            "reset_initial_mask_all_zero": bool(
                reset_info.get("reset_initial_mask_all_zero", False)
            ),
            "reset_initial_mask_required": float(
                reset_info.get("reset_initial_mask_required", 0.0)
            ),
            "reset_initial_body_clearance_m": float(
                reset_info.get("reset_initial_body_clearance_m", 0.0)
            ),
            "reset_candidate_reject_mask_count": int(
                reset_info.get("reset_candidate_reject_mask_count", 0)
            ),
            "reset_candidate_reject_collision_count": int(
                reset_info.get("reset_candidate_reject_collision_count", 0)
            ),
            "reset_scene_retry_count": int(
                reset_info.get("reset_scene_retry_count", 0)
            ),
            "reset_scene_last_failed_seed": int(
                reset_info.get("reset_scene_last_failed_seed", -1)
            ),
            "reset_scene_success_seed": int(
                reset_info.get(
                    "reset_scene_success_seed",
                    reset_info.get("scene_seed", -1),
                )
            ),
            "fallback_used": bool(reset_info.get("fallback_used", False)),
            "initial_collision": bool(reset_info.get("initial_collision", False)),
            "hybrid_astar_valid_rate": float(
                final_info.get("hybrid_astar_valid_rate", 0.0)
            ),
            "planner_valid": bool(final_info.get("planner_valid", False)),
            "planner_fallback_used": bool(final_info.get("planner_fallback_used", False)),
            "planner_fail_reason": str(final_info.get("planner_fail_reason", "")),
            "planner_source": str(final_info.get("planner_source", "")),
            "rs_attempt_count": int(final_info.get("rs_attempt_count", 0)),
            "rs_success_count": int(final_info.get("rs_success_count", 0)),
            "rs_latched": bool(final_info.get("rs_latched", False)),
            "rs_valid_rate": float(final_info.get("rs_valid_rate", 0.0)),
            "rs_plan_time_ms_mean": float(
                final_info.get("rs_plan_time_ms_mean", 0.0)
            ),
            "rs_plan_time_ms_max": float(
                final_info.get("rs_plan_time_ms_max", 0.0)
            ),
            "rs_reward": float(final_info.get("rs_reward", 0.0)),
            "rs_cost": float(final_info.get("rs_cost", 0.0)),
            "rs_remaining_length": float(
                final_info.get("rs_remaining_length", 0.0)
            ),
            "rs_projection_error": float(
                final_info.get("rs_projection_error", 0.0)
            ),
            "rs_heading_error": float(final_info.get("rs_heading_error", 0.0)),
            "rs_fail_reason": str(final_info.get("rs_fail_reason", "")),
            "hope_teacher_enabled": bool(
                final_info.get("hope_teacher_enabled", False)
            ),
            "hope_teacher_available": bool(
                final_info.get("hope_teacher_available", False)
            ),
            "hope_plan_success": bool(final_info.get("hope_plan_success", False)),
            "hope_plan_fail_reason": str(
                final_info.get("hope_plan_fail_reason", "")
            ),
            "hope_cache_hit": bool(final_info.get("hope_cache_hit", False)),
            "hope_path_length": float(final_info.get("hope_path_length", 0.0)),
            "hope_path_valid_after_articulated_check": bool(
                final_info.get("hope_path_valid_after_articulated_check", False)
            ),
            "hope_collision_margin": float(
                final_info.get("hope_collision_margin", 0.0)
            ),
            "hope_terminal_heading_error": float(
                final_info.get("hope_terminal_heading_error", 0.0)
            ),
            "hope_reward_mode": str(final_info.get("hope_reward_mode", "")),
            "guide_weight_current": float(
                final_info.get("guide_weight_current", 0.0)
            ),
            "guide_dropout_rate": float(final_info.get("guide_dropout_rate", 0.0)),
            "guide_dropped": bool(final_info.get("guide_dropped", True)),
            "guide_reward": float(final_info.get("guide_reward", 0.0)),
            "guide_progress_reward": float(
                final_info.get("guide_progress_reward", 0.0)
            ),
            "guide_lateral_error": float(final_info.get("guide_lateral_error", 0.0)),
            "guide_heading_error": float(final_info.get("guide_heading_error", 0.0)),
            "guide_anchor_error": float(final_info.get("guide_anchor_error", 0.0)),
            "guide_gear_agreement": float(
                final_info.get("guide_gear_agreement", 0.0)
            ),
            "hope_failure_aggregation_recorded": bool(
                final_info.get("hope_failure_aggregation_recorded", False)
            ),
            "large_heading_error": bool(
                float(final_info.get("heading_error_deg", 0.0)) >= 45.0
            ),
            "large_articulation_error": bool(
                abs(float(final_info.get("phi", 0.0))) >= np.deg2rad(20.0)
            ),
            "mask_cost": float(final_info.get("mask_cost", 0.0)),
            "forced_stop": int(final_info.get("forced_stop", False)),
            "dwa_enabled": bool(final_info.get("dwa_enabled", False)),
            "dwa_triggered": bool(final_info.get("dwa_triggered", False)),
            "dwa_used": bool(final_info.get("dwa_used", False)),
            "dwa_mode": str(final_info.get("dwa_mode", "none")),
            "dwa_reason": str(final_info.get("dwa_reason", "")),
            "dwa_candidate_count": int(final_info.get("dwa_candidate_count", 0)),
            "dwa_valid_candidate_count": int(
                final_info.get("dwa_valid_candidate_count", 0)
            ),
            "dwa_unlock_success": bool(
                final_info.get("dwa_unlock_success", False)
            ),
            "dwa_unlock_step": int(final_info.get("dwa_unlock_step", -1)),
            "dwa_deadlock": bool(final_info.get("dwa_deadlock", False)),
            "dwa_final_max_safe_ratio": float(
                final_info.get("dwa_final_max_safe_ratio", 0.0)
            ),
            "dwa_override_policy_action": bool(
                final_info.get("dwa_override_policy_action", False)
            ),
            "dwa_teacher_action_valid": bool(
                final_info.get("dwa_teacher_action_valid", False)
            ),
            "dwa_policy_loss_weight": float(
                final_info.get("dwa_policy_loss_weight", 1.0)
            ),
            "normal_mask_max": float(final_info.get("normal_mask_max", 0.0)),
            "recovery_mask_applied": bool(
                final_info.get("recovery_mask_applied", False)
            ),
            "recovery_mask_nonzero_count": int(
                final_info.get("recovery_mask_nonzero_count", 0)
            ),
            "recovery_mask_max": float(final_info.get("recovery_mask_max", 0.0)),
            "effective_mask_max": float(final_info.get("effective_mask_max", 0.0)),
            "degenerate_mask": bool(final_info.get("degenerate_mask", False)),
            "mask_floor_applied": bool(final_info.get("mask_floor_applied", False)),
            "mask_max_before_floor": float(
                final_info.get("mask_max_before_floor", 0.0)
            ),
            "episode_degenerate_mask": bool(
                final_info.get("episode_degenerate_mask", False)
            ),
            "episode_mask_floor_applied": bool(
                final_info.get("episode_mask_floor_applied", False)
            ),
            "episode_collision_after_mask_floor": bool(
                final_info.get("episode_collision_after_mask_floor", False)
            ),
            "episode_success_after_mask_floor": bool(
                final_info.get("episode_success_after_mask_floor", False)
            ),
            "step_collision_after_mask_floor": bool(
                final_info.get("step_collision_after_mask_floor", False)
            ),
            "step_success_after_mask_floor": bool(
                final_info.get("step_success_after_mask_floor", False)
            ),
            "collision_after_mask_floor": bool(
                final_info.get("collision_after_mask_floor", False)
            ),
            "success_after_mask_floor": bool(
                final_info.get("success_after_mask_floor", False)
            ),
            "raw_safe_ratio_final": float(final_info.get("raw_safe_ratio", 0.0)),
            "mask_coef": float(current_mask_coef),
            "dwa_bc_coef": float(current_dwa_bc_coef),
            "gear_switch_count": int(episode_gear_switch_count),
        }
        _write_jsonl(episode_jsonl_path, episode_record)
        if writer is not None:
            writer.add_scalar("episode/reward", episode_reward, episode_index)
            writer.add_scalar("episode/steps", episode_steps, episode_index)
            writer.add_scalar(
                "episode/success",
                float(final_info["success"]),
                episode_index,
            )
        if episode_index % 10 == 0 or episode_index == args.total_episodes:
            _update_reward_plot(reward_plot_path, episode_rewards)

        print(
            "episode={}/{} reward={:.3f} episode_steps={} global_step={} "
            "success={} collision={} timeout={} deadlock={}".format(
                episode_index,
                args.total_episodes,
                episode_reward,
                episode_steps,
                global_step,
                int(final_info["success"]),
                int(final_info["collision"]),
                int(final_info["timeout"]),
                int(final_info.get("deadlock", False)),
            )
        )

        checkpoint_due = episode_index % args.checkpoint_interval == 0
        evaluation_due = (
            episode_index % args.eval_interval == 0
            or episode_index == args.total_episodes
        )
        if (
            env_config.no_guide_eval_interval > 0
            and episode_index % env_config.no_guide_eval_interval == 0
        ):
            evaluation_due = True
        should_update = (
            len(buffer) >= ppo_config.rollout_steps
            or episode_index == args.total_episodes
        )
        update_stats = None
        if should_update:
            update_stats = agent.update(
                buffer,
                observation,
                last_done,
                mask_coef=current_mask_coef,
                dwa_bc_coef=current_dwa_bc_coef,
            )
            update_index += 1

        if evaluation_due:
            eval_modes = ["no_guide"]
            if bool(env_config.enable_hope_teacher):
                eval_modes = ["guided", "no_guide"]
            if bool(env_config.enable_dwa_recovery):
                eval_modes.append("dwa_assisted_eval")
            latest_evaluation = _evaluate_policy_across_stages(
                agent=agent,
                env_config=env_config,
                scene_config=scene_config,
                stages=(1, 2, 3, 4),
                seed=args.seed,
                episodes_per_family=args.eval_episodes_per_family,
                eval_modes=tuple(eval_modes),
                scene_type_schedule=curriculum_scene_types if curriculum else None,
            )
            evaluation_record = {
                "episode": episode_index,
                "global_step": global_step,
                **latest_evaluation,
            }
            _write_jsonl(evaluation_jsonl_path, evaluation_record)
            checkpoint_extra = {
                "global_step": global_step,
                "episode": episode_index,
                "stage": args.stage,
                "eval_stages": [1, 2, 3, 4],
                "scene_type_schedule": list(curriculum_scene_types),
            }
            _save_best_checkpoints(
                agent=agent,
                output_dir=output_dir,
                evaluation=latest_evaluation,
                best_scores=best_scores,
                checkpoint_extra=checkpoint_extra,
            )
            print(
                "evaluation episode={episode} det={det} stochastic={stochastic} "
                "stage3_no_latch={stage3:.3f} stage4_recovery={stage4:.3f} "
                "collision={collision:.3f} timeout={timeout:.3f} "
                "checkpoint_score={score:.3f}".format(
                    episode=episode_index,
                    det=latest_evaluation["deterministic"],
                    stochastic=latest_evaluation["stochastic"],
                    stage3=latest_evaluation["stage3_no_latch_success"],
                    stage4=latest_evaluation["stage4_recovery_success"],
                    collision=latest_evaluation["collision_rate"],
                    timeout=latest_evaluation["timeout_rate"],
                    score=latest_evaluation["checkpoint_selection_score"],
                )
            )

        if should_update:
            record = _build_update_record(
                buffer=buffer,
                infos=rollout_infos,
                completed=rollout_completed,
                update_stats=update_stats,
                global_step=global_step,
                episode_index=episode_index,
                update_index=update_index,
                start_time=start_time,
                global_log_std=agent.global_log_std(),
                gear_deadband=env_config.gear_deadband,
                latest_evaluation=latest_evaluation,
            )
            _write_jsonl(update_jsonl_path, record)
            _write_tensorboard_update(writer, record)
            print(
                "ppo_update={update} episode={episode} rollout={rollout} "
                "success_rate={success:.3f} kl_mean={kl_mean:.5f} "
                "kl_max={kl_max:.5f} epochs={epochs}".format(
                    update=update_index,
                    episode=episode_index,
                    rollout=record["rollout_size"],
                    success=record["success_rate"],
                    kl_mean=record["approx_kl_mean"],
                    kl_max=record["approx_kl_max"],
                    epochs=record["ppo_epochs_completed"],
                )
            )
            buffer = RolloutBuffer()
            rollout_infos = []
            rollout_completed = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if checkpoint_due:
            agent.save(
                os.path.join(
                    output_dir,
                    "checkpoint_episode_{:06d}.pt".format(episode_index),
                ),
                extra={
                    "global_step": global_step,
                    "episode": episode_index,
                    "stage": args.stage,
                    "latest_evaluation": dict(latest_evaluation),
                    "best_scores": dict(best_scores),
                    "scene_type_schedule": list(curriculum_scene_types),
                },
            )
        if curriculum:
            stage_selector.record(
                actual_stage,
                bool(final_info["success"]),
                task_family=task_family,
            )
            actual_stage = stage_selector.select_stage(episode_index)
            env.set_active_stage(actual_stage)

        if episode_index < args.total_episodes:
            replay_case = hard_case_replay.sample()
            if replay_case is not None:
                replay_stage = int(replay_case.get("stage", actual_stage))
                if curriculum and replay_stage != int(actual_stage):
                    actual_stage = replay_stage
                    env.set_active_stage(actual_stage)
            observation, reset_info = env.reset(replay_case=replay_case)

    agent.save(
        os.path.join(output_dir, "checkpoint_final.pt"),
        extra={
            "global_step": global_step,
            "episode": episode_index,
            "stage": args.stage,
            "latest_evaluation": dict(latest_evaluation),
            "best_scores": dict(best_scores),
            "scene_type_schedule": list(curriculum_scene_types),
        },
    )
    if writer is not None:
        writer.close()
    print("training artifacts: {}".format(output_dir))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train continuous PPO for local parking.")
    parser.add_argument("--total-episodes", type=int, default=20_000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_ENV_CONFIG.max_steps,
        help="Maximum environment steps per episode",
    )
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=DEFAULT_PPO_CONFIG.ppo_epochs,
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=DEFAULT_PPO_CONFIG.clip_range,
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=DEFAULT_PPO_CONFIG.actor_lr,
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=DEFAULT_PPO_CONFIG.critic_lr,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_PPO_CONFIG.max_grad_norm,
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=DEFAULT_PPO_CONFIG.target_kl,
    )
    parser.add_argument(
        "--kl-early-stop-multiplier",
        type=float,
        default=DEFAULT_PPO_CONFIG.kl_early_stop_multiplier,
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=DEFAULT_PPO_CONFIG.log_std_init,
    )
    parser.add_argument(
        "--log-std-min",
        type=float,
        default=DEFAULT_PPO_CONFIG.log_std_min,
    )
    parser.add_argument(
        "--log-std-max",
        type=float,
        default=DEFAULT_PPO_CONFIG.log_std_max,
    )
    parser.add_argument(
        "--policy-loss-weight-head-in",
        type=float,
        default=DEFAULT_PPO_CONFIG.policy_loss_weight_head_in,
    )
    parser.add_argument(
        "--checkpoint-score-weight-head-in",
        type=float,
        default=DEFAULT_PPO_CONFIG.checkpoint_score_weight_head_in,
    )
    parser.add_argument(
        "--dwa-bc-coef-initial",
        type=float,
        default=DEFAULT_PPO_CONFIG.dwa_bc_coef_initial,
    )
    parser.add_argument(
        "--dwa-bc-coef-final",
        type=float,
        default=DEFAULT_PPO_CONFIG.dwa_bc_coef_final,
    )
    parser.add_argument(
        "--dwa-bc-anneal-start-episode",
        type=int,
        default=DEFAULT_PPO_CONFIG.dwa_bc_anneal_start_episode,
    )
    parser.add_argument(
        "--dwa-bc-anneal-end-episode",
        type=int,
        default=DEFAULT_PPO_CONFIG.dwa_bc_anneal_end_episode,
    )
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument(
        "--scene-type",
        choices=SUPPORTED_SCENE_TYPES,
        default=DEFAULT_SCENE_CONFIG.scene_type,
        help=(
            "Scene generator type for non-curriculum training; --curriculum "
            "uses all supported scene types"
        ),
    )
    parser.add_argument(
        "--scene-family-schedule",
        default=",".join(DEFAULT_ENV_CONFIG.scene_family_schedule),
        help="Comma-separated family schedule for cached scenes; currently only head_in",
    )
    parser.add_argument(
        "--scene-pool-size",
        type=int,
        default=DEFAULT_ENV_CONFIG.scene_pool_size,
        help="Cached scene variants per stage/scene type",
    )
    parser.add_argument("--use-hybrid-astar", action="store_true")
    _add_config_bool_argument(
        parser,
        "enable-rs-potential",
        DEFAULT_ENV_CONFIG.rs_potential_enabled,
        "Enable near-goal Reeds-Shepp potential shaping",
        "Disable near-goal Reeds-Shepp potential shaping",
    )
    parser.add_argument(
        "--mask-cost-coef-final",
        type=float,
        default=DEFAULT_ENV_CONFIG.mask_cost_coef_final,
        help="Final PPO mask-penalty coefficient; execution-layer action mask remains active",
    )
    parser.add_argument(
        "--disable-mask-observation",
        action="store_true",
        help="Zero the action-mask observation slice while keeping safe execution active",
    )
    parser.add_argument(
        "--rear-lidar-observation-mode",
        choices=("normal", "zero"),
        default=DEFAULT_ENV_CONFIG.rear_lidar_observation_mode,
        help="Use normal rear LiDAR observations or zero only the rear LiDAR observation slice",
    )
    parser.add_argument(
        "--disable-action-mask-execution",
        action="store_true",
        help=(
            "Unsafe diagnostic: bypass safe-speed-ratio execution clipping while "
            "leaving observation and collision truth unchanged"
        ),
    )
    parser.add_argument(
        "--enable-mask-floor-fallback",
        action="store_true",
        help="Enable degenerate action-mask floor fallback",
    )
    parser.add_argument(
        "--disable-mask-floor-fallback",
        action="store_true",
        help="Keep degenerate action-mask floor fallback disabled",
    )
    parser.add_argument(
        "--mask-degenerate-eps",
        type=float,
        default=DEFAULT_ENV_CONFIG.mask_degenerate_eps,
        help="Treat action masks with max ratio below this value as degenerate",
    )
    parser.add_argument(
        "--mask-floor-value",
        type=float,
        default=DEFAULT_ENV_CONFIG.mask_floor_value,
        help="Uniform positive safe-speed ratio used for degenerate masks",
    )
    parser.add_argument(
        "--apply-floor-only-when-all-zero",
        action="store_true",
        help="Only apply mask floor fallback when every mask entry is zero",
    )
    _add_config_bool_argument(
        parser,
        "enable-dwa-recovery",
        DEFAULT_ENV_CONFIG.enable_dwa_recovery,
        "Enable strict-mask DWA recovery diagnostics",
        "Disable strict-mask DWA recovery diagnostics",
    )
    _add_config_bool_argument(
        parser,
        "dwa-override-policy-action",
        DEFAULT_ENV_CONFIG.dwa_override_policy_action,
        "Allow DWA recovery to replace the policy action when it finds a candidate",
        "Keep DWA diagnostic-only and do not replace policy actions",
    )
    parser.add_argument(
        "--dwa-recovery-mode",
        choices=("teacher_override", "policy_with_recovery_mask", "recovery_mask_only"),
        default=DEFAULT_ENV_CONFIG.dwa_recovery_mode,
        help="DWA recovery integration mode for degenerate action masks",
    )
    parser.add_argument(
        "--dwa-override-policy-loss-weight",
        type=float,
        default=DEFAULT_ENV_CONFIG.dwa_override_policy_loss_weight,
        help="PPO policy-loss multiplier for transitions executed by DWA override",
    )
    _add_config_bool_argument(
        parser,
        "dwa-deadlock-termination",
        DEFAULT_ENV_CONFIG.dwa_enable_deadlock_termination,
        "Terminate episodes after repeated DWA no-candidate deadlocks",
        "Disable DWA deadlock termination",
    )
    parser.add_argument(
        "--dwa-all-zero-eps",
        type=float,
        default=DEFAULT_ENV_CONFIG.dwa_all_zero_eps,
    )
    parser.add_argument(
        "--dwa-low-safe-ratio",
        type=float,
        default=DEFAULT_ENV_CONFIG.dwa_low_safe_ratio,
    )
    parser.add_argument(
        "--dwa-unlock-safe-ratio",
        type=float,
        default=DEFAULT_ENV_CONFIG.dwa_unlock_safe_ratio,
    )
    parser.add_argument(
        "--dwa-unlock-min-safe-ratio-improvement",
        type=float,
        default=DEFAULT_ENV_CONFIG.dwa_unlock_min_safe_ratio_improvement,
    )
    parser.add_argument(
        "--dwa-forced-stop-patience",
        type=int,
        default=DEFAULT_ENV_CONFIG.dwa_forced_stop_patience,
    )
    parser.add_argument(
        "--dwa-no-progress-patience",
        type=int,
        default=DEFAULT_ENV_CONFIG.dwa_no_progress_patience,
    )
    parser.add_argument(
        "--dwa-deadlock-patience",
        type=int,
        default=DEFAULT_ENV_CONFIG.dwa_deadlock_patience,
    )
    parser.add_argument(
        "--dwa-horizon-steps",
        type=int,
        default=DEFAULT_ENV_CONFIG.dwa_horizon_steps,
    )
    parser.add_argument(
        "--dwa-speed-ratios",
        default=",".join(str(item) for item in DEFAULT_ENV_CONFIG.dwa_speed_ratios),
        help="Comma-separated local DWA speed ratios, each inside (0, 1]",
    )
    parser.add_argument(
        "--dwa-unlock-speed-ratios",
        default=",".join(
            str(item) for item in DEFAULT_ENV_CONFIG.dwa_unlock_speed_ratios
        ),
        help="Comma-separated small physical speed ratios for DWA unlock candidates",
    )
    parser.add_argument(
        "--dwa-recovery-max-speed-ratio",
        type=float,
        default=DEFAULT_ENV_CONFIG.dwa_recovery_max_speed_ratio,
    )
    parser.add_argument(
        "--dwa-recovery-phi-bin-radius",
        type=int,
        default=DEFAULT_ENV_CONFIG.dwa_recovery_phi_bin_radius,
        help="Number of neighboring phi-dot bins enabled around each recovery candidate",
    )
    parser.add_argument(
        "--enable-hope-teacher",
        action="store_true",
        help="Enable training-only HOPE teacher planning and diagnostics",
    )
    parser.add_argument(
        "--hope-code-dir",
        default=DEFAULT_ENV_CONFIG.hope_code_dir,
        help="Path to the HOPE repository directory",
    )
    parser.add_argument(
        "--hope-weight-path",
        default=DEFAULT_ENV_CONFIG.hope_weight_path,
        help="Path to the HOPE checkpoint used to verify teacher availability",
    )
    parser.add_argument(
        "--hope-cache-dir",
        default=DEFAULT_ENV_CONFIG.hope_cache_dir,
        help="Directory for per-episode HOPE teacher cache records",
    )
    parser.add_argument(
        "--use-teacher-reward",
        action="store_true",
        help="Add annealed HOPE guidance reward during training",
    )
    parser.add_argument(
        "--guide-weight-initial",
        type=float,
        default=DEFAULT_ENV_CONFIG.guide_weight_initial,
    )
    parser.add_argument(
        "--guide-weight-final",
        type=float,
        default=DEFAULT_ENV_CONFIG.guide_weight_final,
    )
    parser.add_argument(
        "--guide-anneal-start-episode",
        type=int,
        default=DEFAULT_ENV_CONFIG.guide_anneal_start_episode,
    )
    parser.add_argument(
        "--guide-anneal-end-episode",
        type=int,
        default=DEFAULT_ENV_CONFIG.guide_anneal_end_episode,
    )
    parser.add_argument(
        "--guide-dropout-initial",
        type=float,
        default=DEFAULT_ENV_CONFIG.guide_dropout_initial,
    )
    parser.add_argument(
        "--guide-dropout-final",
        type=float,
        default=DEFAULT_ENV_CONFIG.guide_dropout_final,
    )
    parser.add_argument(
        "--teacher-corridor-width",
        type=float,
        default=DEFAULT_ENV_CONFIG.teacher_corridor_width,
    )
    parser.add_argument(
        "--teacher-anchor-weight",
        type=float,
        default=DEFAULT_ENV_CONFIG.teacher_anchor_weight,
    )
    parser.add_argument(
        "--teacher-heading-weight",
        type=float,
        default=DEFAULT_ENV_CONFIG.teacher_heading_weight,
    )
    parser.add_argument(
        "--teacher-progress-weight",
        type=float,
        default=DEFAULT_ENV_CONFIG.teacher_progress_weight,
    )
    parser.add_argument(
        "--teacher-gear-weight",
        type=float,
        default=DEFAULT_ENV_CONFIG.teacher_gear_weight,
    )
    parser.add_argument(
        "--teacher-reward-clip",
        type=float,
        default=DEFAULT_ENV_CONFIG.teacher_reward_clip,
    )
    parser.add_argument(
        "--enable-offpath-reset",
        action="store_true",
        help="Sample some resets from perturbed states near the HOPE corridor",
    )
    parser.add_argument(
        "--enable-failure-aggregation",
        action="store_true",
        help="Record failed rollout states for later teacher/failure curriculum use",
    )
    parser.add_argument(
        "--no-guide-eval-interval",
        type=int,
        default=DEFAULT_ENV_CONFIG.no_guide_eval_interval,
        help="Optional extra no-guide evaluation interval; 0 disables the extra trigger",
    )
    parser.add_argument(
        "--disable-hard-case-replay",
        action="store_true",
        help="Disable automatic replay resets from no-RS collision/timeout failures",
    )
    parser.add_argument(
        "--hard-case-replay-ratio",
        type=float,
        default=DEFAULT_ENV_CONFIG.hard_case_replay_ratio,
        help="Probability of sampling a reset from the hard-case replay buffer",
    )
    parser.add_argument(
        "--hard-case-replay-capacity",
        type=int,
        default=DEFAULT_ENV_CONFIG.hard_case_replay_capacity,
    )
    parser.add_argument(
        "--hard-case-replay-tail-steps",
        type=int,
        default=DEFAULT_ENV_CONFIG.hard_case_replay_tail_steps,
        help="Number of terminal tail states recorded from each eligible failure",
    )
    parser.add_argument(
        "--hard-case-replay-attempts",
        type=int,
        default=DEFAULT_ENV_CONFIG.hard_case_replay_attempts,
    )
    parser.add_argument(
        "--hard-case-replay-xy-std",
        type=float,
        default=DEFAULT_ENV_CONFIG.hard_case_replay_xy_std,
    )
    parser.add_argument(
        "--hard-case-replay-heading-std-deg",
        type=float,
        default=DEFAULT_ENV_CONFIG.hard_case_replay_heading_std_deg,
    )
    parser.add_argument(
        "--hard-case-replay-phi-std-deg",
        type=float,
        default=DEFAULT_ENV_CONFIG.hard_case_replay_phi_std_deg,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Run fixed-family deterministic and stochastic evaluation every N episodes",
    )
    parser.add_argument(
        "--eval-episodes-per-family",
        type=int,
        default=MIN_EVAL_EPISODES_PER_FAMILY,
        help="Evaluation episodes for each task family and policy mode; minimum 20",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Default: <repo>/runs/local_parking_<timestamp>_seedN",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable multi-stage curriculum training (auto-selects stages 1-4)",
    )
    parser.add_argument(
        "--curriculum-mode",
        choices=("adaptive", "uniform", "fixed"),
        default="adaptive",
        help=(
            "adaptive keeps the performance-weighted curriculum, uniform samples "
            "stages 1-4 uniformly when --curriculum is set, fixed trains only --stage"
        ),
    )
    parser.add_argument(
        "--curriculum-target-success",
        type=float,
        default=0.75,
        help="Target success rate for curriculum worst-performance selection",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
