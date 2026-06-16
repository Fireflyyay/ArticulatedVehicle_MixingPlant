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
    TASK_FAMILIES as SCENE_TASK_FAMILIES,
    normalize_family_schedule,
)
from model.continuous_ppo import ContinuousPPOAgent, RolloutBuffer
from planning.passenger_hybrid_astar import PassengerHybridAStar
from train.curriculum import CurriculumStageSelector, MultiStageScenePool

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK_FAMILIES = SCENE_TASK_FAMILIES


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _task_family(reset_info, scene_metadata):
    task_family = str(reset_info.get("task_family", ""))
    if task_family in TASK_FAMILIES:
        return task_family
    mode = str(reset_info.get("goal_orientation_mode", ""))
    if mode == "head_in":
        return "head_in"
    if mode != "parallel":
        raise ValueError("unsupported goal orientation mode: {}".format(mode))
    if bool(scene_metadata.get("parallel_reverse", False)):
        return "parallel_rev"
    return "parallel_fwd"


def _parse_family_schedule(value):
    return normalize_family_schedule(value)


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


def _write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _rs_metrics_by_mode(infos):
    grouped = {}
    for mode in ("head_in", "parallel"):
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


def _resolve_output_dir(output_dir, seed, timestamp=None):
    if output_dir:
        return os.path.abspath(output_dir)
    run_time = timestamp or datetime.now()
    run_name = "local_parking_{}_seed{}".format(
        run_time.strftime("%Y%m%d_%H%M%S"),
        int(seed),
    )
    return os.path.join(REPO_ROOT, "runs", run_name)


def _write_config_snapshot(path, args, env_config, ppo_config):
    sections = (
        ("training_arguments", vars(args)),
        ("vehicle", asdict(DEFAULT_VEHICLE_PARAMS)),
        ("action_mask", asdict(DEFAULT_MASK_CONFIG)),
        ("scene", asdict(DEFAULT_SCENE_CONFIG)),
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
    return {
        "global_step": global_step,
        "episode": episode_index,
        "update": update_index,
        "rollout_size": len(buffer),
        "steps_per_second": global_step / max(time.perf_counter() - start_time, 1e-6),
        **update_stats,
        "success/head_in": family_success["head_in"],
        "success/parallel_fwd": family_success["parallel_fwd"],
        "success/parallel_rev": family_success["parallel_rev"],
        "episodes/head_in": family_success["head_in_count"],
        "episodes/parallel_fwd": family_success["parallel_fwd_count"],
        "episodes/parallel_rev": family_success["parallel_rev_count"],
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
        "collision_rate": _safe_mean(
            [float(item["collision"]) for item in completed]
        ),
        "timeout_rate": _safe_mean([float(item["timeout"]) for item in completed]),
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
        "parallel_fwd": float(ppo_config.checkpoint_score_weight_parallel_fwd),
        "parallel_rev": float(ppo_config.checkpoint_score_weight_parallel_rev),
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


def _evaluate_policy_by_family(
    agent,
    env_config,
    stage,
    seed,
    episodes_per_family,
):
    episodes_per_family = int(episodes_per_family)
    if episodes_per_family <= 0:
        raise ValueError("episodes_per_family must be positive")
    eval_config = replace(
        env_config,
        curriculum_stage=int(stage),
        use_hybrid_astar=False,
        rs_potential_enabled=False,
        scene_family_schedule=TASK_FAMILIES,
        parallel_rev_curriculum_episodes=0,
    )
    base_seed = ((int(seed) + 100_000) // 16) * 16
    results = {}
    cuda_devices = []
    if agent.device.type == "cuda":
        cuda_devices = [
            agent.device.index
            if agent.device.index is not None
            else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(base_seed + 7)
        if agent.device.type == "cuda":
            torch.cuda.manual_seed_all(base_seed + 7)
        for deterministic in (True, False):
            env = LocalParkingEnv(
                config=eval_config,
                seed=base_seed,
            )
            successes = dict((family, []) for family in TASK_FAMILIES)
            while min(len(values) for values in successes.values()) < episodes_per_family:
                observation, reset_info = env.reset()
                family = _task_family(reset_info, env.scene.metadata)
                if len(successes[family]) >= episodes_per_family:
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
                successes[family].append(float(final_info["success"]))
            mode = "deterministic" if deterministic else "stochastic"
            results[mode] = dict(
                (family, _safe_mean(successes[family]))
                for family in TASK_FAMILIES
            )
    results["weighted_score"] = _weighted_checkpoint_score(
        results["deterministic"],
        agent.config,
    )
    results["stage"] = int(stage)
    results["episodes_per_family"] = episodes_per_family
    return results


def _save_best_checkpoints(
    agent,
    output_dir,
    evaluation,
    best_scores,
    checkpoint_extra,
):
    metrics = dict(evaluation["deterministic"])
    metrics["weighted_score"] = float(evaluation["weighted_score"])
    for name, score in metrics.items():
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
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    if args.eval_episodes_per_family <= 0:
        raise ValueError("--eval-episodes-per-family must be positive")
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
        args.policy_loss_weight_parallel_fwd,
        args.policy_loss_weight_parallel_rev,
    )
    if any(float(weight) < 0.0 for weight in policy_weights):
        raise ValueError("policy loss weights must be non-negative")
    family_schedule = _parse_family_schedule(args.scene_family_schedule)
    if args.parallel_rev_curriculum_episodes < 0:
        raise ValueError("--parallel-rev-curriculum-episodes must be non-negative")
    if args.parallel_rev_warmup_distance_min <= 0.0:
        raise ValueError("--parallel-rev-warmup-distance-min must be positive")
    if args.parallel_rev_warmup_distance_max < args.parallel_rev_warmup_distance_min:
        raise ValueError(
            "--parallel-rev-warmup-distance-max must be >= --parallel-rev-warmup-distance-min"
        )
    if args.parallel_rev_warmup_lateral < 0.0:
        raise ValueError("--parallel-rev-warmup-lateral must be non-negative")
    if args.parallel_rev_warmup_heading_deg < 0.0:
        raise ValueError("--parallel-rev-warmup-heading-deg must be non-negative")
    if args.parallel_rev_warmup_phi_deg < 0.0:
        raise ValueError("--parallel-rev-warmup-phi-deg must be non-negative")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=args.stage,
        scene_family_schedule=family_schedule,
        use_hybrid_astar=bool(args.use_hybrid_astar),
        rs_potential_enabled=not bool(
            getattr(args, "disable_rs_potential", False)
        ),
        parallel_rev_curriculum_episodes=args.parallel_rev_curriculum_episodes,
        parallel_rev_warmup_distance_range=(
            args.parallel_rev_warmup_distance_min,
            args.parallel_rev_warmup_distance_max,
        ),
        parallel_rev_warmup_lateral_range=args.parallel_rev_warmup_lateral,
        parallel_rev_warmup_heading_range_deg=args.parallel_rev_warmup_heading_deg,
        parallel_rev_warmup_phi_range_deg=args.parallel_rev_warmup_phi_deg,
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
        policy_loss_weight_parallel_fwd=args.policy_loss_weight_parallel_fwd,
        policy_loss_weight_parallel_rev=args.policy_loss_weight_parallel_rev,
        checkpoint_score_weight_head_in=args.checkpoint_score_weight_head_in,
        checkpoint_score_weight_parallel_fwd=args.checkpoint_score_weight_parallel_fwd,
        checkpoint_score_weight_parallel_rev=args.checkpoint_score_weight_parallel_rev,
    )
    output_dir = _resolve_output_dir(args.output_dir, args.seed)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    _write_config_snapshot(
        os.path.join(output_dir, "config.txt"),
        args,
        env_config,
        ppo_config,
    )

    curriculum = bool(args.curriculum)
    stage_selector = None
    multi_pool = None
    if curriculum:
        multi_pool = MultiStageScenePool(
            pool_size=env_config.scene_pool_size,
            base_seed=args.seed,
            scene_config=DEFAULT_SCENE_CONFIG,
            family_schedule=env_config.scene_family_schedule,
        )
        stage_selector = CurriculumStageSelector(
            target_success_rate=float(args.curriculum_target_success),
        )
        actual_stage = stage_selector.select_stage(0)
    else:
        actual_stage = args.stage

    planner = PassengerHybridAStar() if args.use_hybrid_astar else None
    env = LocalParkingEnv(
        config=env_config,
        hybrid_planner=planner,
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

        episode_reward = 0.0
        episode_steps = 0
        episode_gear_switch_count = 0
        previous_motion_gear = None
        final_info = None
        done = False
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
            )
            info["task_family"] = task_family
            rollout_infos.append(info)
            observation = next_observation
            global_step += 1
            episode_steps += 1
            episode_reward += float(reward)
            final_info = info
            last_done = done

        episode_index += 1
        episode_rewards.append(episode_reward)
        final_info["task_family"] = task_family
        final_info["gear_switch_count"] = int(episode_gear_switch_count)
        rollout_completed.append(final_info)
        episode_record = {
            "episode": episode_index,
            "global_step": global_step,
            "episode_steps": episode_steps,
            "episode_reward": episode_reward,
            "success": bool(final_info["success"]),
            "collision": bool(final_info["collision"]),
            "timeout": bool(final_info["timeout"]),
            "front_overlap": float(final_info["front_overlap"]),
            "best_front_overlap": float(final_info["best_front_overlap"]),
            "heading_error_deg": float(final_info["heading_error_deg"]),
            "distance_to_goal": float(final_info["distance_to_goal"]),
            "scenario_type": reset_info.get("scenario_type", ""),
            "scene_seed": int(reset_info.get("scene_seed", -1)),
            "goal_orientation_mode": str(reset_info.get("goal_orientation_mode", "")),
            "task_family": task_family,
            "clearance_bucket": str(reset_info.get("clearance_bucket", "")),
            "approach_side_bucket": str(reset_info.get("approach_side_bucket", "")),
            "reverse_required_bucket": str(
                reset_info.get("reverse_required_bucket", "")
            ),
            "difficulty_label": str(reset_info.get("difficulty_label", "")),
            "nominal_target_collision": bool(
                reset_info.get("nominal_target_collision", False)
            ),
            "nominal_target_clearance_m": float(
                reset_info.get("nominal_target_clearance_m", 0.0)
            ),
            "success_neighborhood_feasible_count": int(
                reset_info.get("success_neighborhood_feasible_count", 0)
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
            "parallel_rev_curriculum_progress": float(
                reset_info.get("parallel_rev_curriculum_progress", 1.0)
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
            "mask_cost": float(final_info.get("mask_cost", 0.0)),
            "forced_stop": int(final_info.get("forced_stop", False)),
            "raw_safe_ratio_final": float(final_info.get("raw_safe_ratio", 0.0)),
            "mask_coef": float(current_mask_coef),
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
            "success={} collision={} timeout={}".format(
                episode_index,
                args.total_episodes,
                episode_reward,
                episode_steps,
                global_step,
                int(final_info["success"]),
                int(final_info["collision"]),
                int(final_info["timeout"]),
            )
        )

        checkpoint_due = episode_index % args.checkpoint_interval == 0
        evaluation_due = (
            episode_index % args.eval_interval == 0
            or episode_index == args.total_episodes
        )
        should_update = (
            len(buffer) >= ppo_config.rollout_steps
            or episode_index == args.total_episodes
        )
        update_stats = None
        if should_update:
            update_stats = agent.update(buffer, observation, last_done, mask_coef=current_mask_coef)
            update_index += 1

        if evaluation_due:
            latest_evaluation = _evaluate_policy_by_family(
                agent=agent,
                env_config=env_config,
                stage=args.stage,
                seed=args.seed,
                episodes_per_family=args.eval_episodes_per_family,
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
                "weighted_score={score:.3f}".format(
                    episode=episode_index,
                    det=latest_evaluation["deterministic"],
                    stochastic=latest_evaluation["stochastic"],
                    score=latest_evaluation["weighted_score"],
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
                },
            )
        if curriculum:
            stage_selector.record(actual_stage, bool(final_info["success"]))
            actual_stage = stage_selector.select_stage(episode_index)
            env.set_active_stage(actual_stage)

        if episode_index < args.total_episodes:
            observation, reset_info = env.reset()

    agent.save(
        os.path.join(output_dir, "checkpoint_final.pt"),
        extra={
            "global_step": global_step,
            "episode": episode_index,
            "stage": args.stage,
            "latest_evaluation": dict(latest_evaluation),
            "best_scores": dict(best_scores),
        },
    )
    if writer is not None:
        writer.close()
    print("training artifacts: {}".format(output_dir))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train continuous PPO for local parking.")
    parser.add_argument("--total-episodes", type=int, default=20_000)
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
        "--policy-loss-weight-parallel-fwd",
        type=float,
        default=DEFAULT_PPO_CONFIG.policy_loss_weight_parallel_fwd,
    )
    parser.add_argument(
        "--policy-loss-weight-parallel-rev",
        type=float,
        default=DEFAULT_PPO_CONFIG.policy_loss_weight_parallel_rev,
    )
    parser.add_argument(
        "--checkpoint-score-weight-head-in",
        type=float,
        default=DEFAULT_PPO_CONFIG.checkpoint_score_weight_head_in,
    )
    parser.add_argument(
        "--checkpoint-score-weight-parallel-fwd",
        type=float,
        default=DEFAULT_PPO_CONFIG.checkpoint_score_weight_parallel_fwd,
    )
    parser.add_argument(
        "--checkpoint-score-weight-parallel-rev",
        type=float,
        default=DEFAULT_PPO_CONFIG.checkpoint_score_weight_parallel_rev,
    )
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument(
        "--scene-family-schedule",
        default=",".join(DEFAULT_ENV_CONFIG.scene_family_schedule),
        help=(
            "Comma-separated family schedule for cached scenes; repeat a family "
            "to oversample it, e.g. head_in,parallel_fwd,parallel_rev,parallel_rev"
        ),
    )
    parser.add_argument(
        "--parallel-rev-curriculum-episodes",
        type=int,
        default=DEFAULT_ENV_CONFIG.parallel_rev_curriculum_episodes,
        help="Episodes over which stage-1 parallel_rev starts expand to full ranges",
    )
    parser.add_argument(
        "--parallel-rev-warmup-distance-min",
        type=float,
        default=DEFAULT_ENV_CONFIG.parallel_rev_warmup_distance_range[0],
    )
    parser.add_argument(
        "--parallel-rev-warmup-distance-max",
        type=float,
        default=DEFAULT_ENV_CONFIG.parallel_rev_warmup_distance_range[1],
    )
    parser.add_argument(
        "--parallel-rev-warmup-lateral",
        type=float,
        default=DEFAULT_ENV_CONFIG.parallel_rev_warmup_lateral_range,
    )
    parser.add_argument(
        "--parallel-rev-warmup-heading-deg",
        type=float,
        default=DEFAULT_ENV_CONFIG.parallel_rev_warmup_heading_range_deg,
    )
    parser.add_argument(
        "--parallel-rev-warmup-phi-deg",
        type=float,
        default=DEFAULT_ENV_CONFIG.parallel_rev_warmup_phi_range_deg,
    )
    parser.add_argument("--use-hybrid-astar", action="store_true")
    parser.add_argument(
        "--disable-rs-potential",
        action="store_true",
        help="Disable near-goal Reeds-Shepp potential shaping",
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
        default=4,
        help="Evaluation episodes for each task family and policy mode",
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
        "--curriculum-target-success",
        type=float,
        default=0.75,
        help="Target success rate for curriculum worst-performance selection",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
