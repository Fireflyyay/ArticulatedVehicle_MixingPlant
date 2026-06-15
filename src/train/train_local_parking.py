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
from model.continuous_ppo import ContinuousPPOAgent, RolloutBuffer
from planning.passenger_hybrid_astar import PassengerHybridAStar
from train.curriculum import CurriculumStageSelector, MultiStageScenePool

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


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
):
    raw_actions = np.asarray(buffer.raw_actions)
    executed_actions = np.asarray(buffer.executed_actions)
    return {
        "global_step": global_step,
        "episode": episode_index,
        "update": update_index,
        "rollout_size": len(buffer),
        "steps_per_second": global_step / max(time.perf_counter() - start_time, 1e-6),
        **update_stats,
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
        "gear_switch_rate": _safe_mean([
            float(i > 0 and infos[i].get("gear", 0) != infos[i - 1].get("gear", 0))
            for i in range(1, len(infos))
        ]) if len(infos) > 1 else 0.0,
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


def train(args):
    if args.total_episodes <= 0:
        raise ValueError("--total-episodes must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_config = replace(
        DEFAULT_ENV_CONFIG,
        curriculum_stage=args.stage,
        use_hybrid_astar=bool(args.use_hybrid_astar),
        rs_potential_enabled=not bool(
            getattr(args, "disable_rs_potential", False)
        ),
    )
    ppo_config = replace(
        DEFAULT_PPO_CONFIG,
        rollout_steps=args.rollout_steps,
        batch_size=min(args.batch_size, args.rollout_steps),
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
        final_info = None
        done = False
        while not done:
            raw_action, log_prob, value = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(raw_action)
            done = terminated or truncated
            buffer.add(
                observation=observation,
                raw_action=raw_action,
                executed_action=info["executed_action"],
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                mask_cost=float(info.get("mask_cost", 0.0)),
            )
            rollout_infos.append(info)
            observation = next_observation
            global_step += 1
            episode_steps += 1
            episode_reward += float(reward)
            final_info = info
            last_done = done

        episode_index += 1
        episode_rewards.append(episode_reward)
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
        should_update = (
            len(buffer) >= ppo_config.rollout_steps
            or episode_index == args.total_episodes
        )
        if should_update:
            update_stats = agent.update(buffer, observation, last_done, mask_coef=current_mask_coef)
            update_index += 1
            record = _build_update_record(
                buffer=buffer,
                infos=rollout_infos,
                completed=rollout_completed,
                update_stats=update_stats,
                global_step=global_step,
                episode_index=episode_index,
                update_index=update_index,
                start_time=start_time,
            )
            _write_jsonl(update_jsonl_path, record)
            _write_tensorboard_update(writer, record)
            print(
                "ppo_update={update} episode={episode} rollout={rollout} "
                "success_rate={success:.3f} kl={kl:.5f}".format(
                    update=update_index,
                    episode=episode_index,
                    rollout=record["rollout_size"],
                    success=record["success_rate"],
                    kl=record["approx_kl"],
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
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=1)
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
