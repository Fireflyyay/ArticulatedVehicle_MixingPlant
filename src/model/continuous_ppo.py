from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from config import DEFAULT_PPO_CONFIG


def _orthogonal_init(module, gain=np.sqrt(2.0)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class ObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_vehicle_encoder = nn.Sequential(
            nn.Linear(19, 64),
            nn.Tanh(),
        )
        self.lidar_encoder = nn.Sequential(
            nn.Linear(108, 128),
            nn.Tanh(),
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(22, 64),
            nn.Tanh(),
        )
        self.apply(_orthogonal_init)

    def forward(self, obs):
        slot_vehicle = self.slot_vehicle_encoder(obs[..., :19])
        lidar = self.lidar_encoder(obs[..., 19:127])
        mask = self.mask_encoder(obs[..., 127:149])
        return torch.cat([slot_vehicle, lidar, mask], dim=-1)


class ContinuousActorCritic(nn.Module):
    """Separate actor/critic encoders keep their optimizers independent."""

    action_dim = 2

    def __init__(
        self,
        log_std_init=-0.7,
        log_std_min=-2.5,
        log_std_max=-0.3,
    ):
        super().__init__()
        log_std_min = float(log_std_min)
        log_std_max = float(log_std_max)
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if log_std_max > 0.0:
            raise ValueError("log_std_max must not exceed 0.0")
        if not log_std_min <= float(log_std_init) <= log_std_max:
            raise ValueError("log_std_init must be inside the configured range")
        self.register_buffer(
            "actor_log_std_min",
            torch.tensor(log_std_min, dtype=torch.float32),
        )
        self.register_buffer(
            "actor_log_std_max",
            torch.tensor(log_std_max, dtype=torch.float32),
        )

        self.actor_encoder = ObservationEncoder()
        self.actor_trunk = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, self.action_dim)
        self.actor_log_std = nn.Parameter(
            torch.full((self.action_dim,), float(log_std_init))
        )

        self.critic_encoder = ObservationEncoder()
        self.critic_trunk = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(256, 1)

        self.actor_trunk.apply(_orthogonal_init)
        self.critic_trunk.apply(_orthogonal_init)
        _orthogonal_init(self.actor_mean, gain=0.01)
        _orthogonal_init(self.critic_head, gain=1.0)

    def actor_parameters(self):
        return list(self.actor_encoder.parameters()) + list(self.actor_trunk.parameters()) + list(
            self.actor_mean.parameters()
        ) + [self.actor_log_std]

    def critic_parameters(self):
        return list(self.critic_encoder.parameters()) + list(self.critic_trunk.parameters()) + list(
            self.critic_head.parameters()
        )

    def distribution(self, obs):
        features = self.actor_trunk(self.actor_encoder(obs))
        mean = self.actor_mean(features)
        log_std = self.global_log_std()
        return Normal(mean, log_std.exp().expand_as(mean))

    def global_log_std(self):
        return torch.clamp(
            self.actor_log_std,
            min=self.actor_log_std_min,
            max=self.actor_log_std_max,
        )

    @torch.no_grad()
    def project_log_std(self):
        self.actor_log_std.clamp_(
            min=float(self.actor_log_std_min.item()),
            max=float(self.actor_log_std_max.item()),
        )

    def value(self, obs):
        features = self.critic_trunk(self.critic_encoder(obs))
        return self.critic_head(features).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(distribution, pre_tanh, action):
        base_log_prob = distribution.log_prob(pre_tanh).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return base_log_prob - correction

    def sample_action(self, obs, deterministic=False):
        distribution = self.distribution(obs)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        raw_action = torch.tanh(pre_tanh)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, raw_action)
        return pre_tanh, raw_action, log_prob, self.value(obs)

    def evaluate_actions(self, obs, pre_tanh_actions, raw_actions):
        distribution = self.distribution(obs)
        log_prob = self._squashed_log_prob(
            distribution,
            pre_tanh_actions,
            raw_actions,
        )
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(obs)


class RolloutBuffer:
    """Stores policy samples separately from environment-executed actions."""

    log_prob_action_source = "pre_tanh_action_and_raw_action"

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.pre_tanh_actions = []
        self.raw_actions = []
        self.executed_actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.mask_costs = []
        self.task_families = []
        self.dwa_raw_actions = []
        self.dwa_teacher_action_valid = []
        self.dwa_used = []
        self.dwa_policy_loss_weights = []
        self.recovery_mask_applied = []
        self.recovery_mask_nonzero_counts = []
        self.recovery_mask_maxes = []

    def add(
        self,
        observation,
        raw_action,
        executed_action,
        log_prob,
        reward,
        done,
        value,
        *,
        pre_tanh_action,
        mask_cost=0.0,
        task_family="head_in",
        dwa_raw_action=None,
        dwa_teacher_action_valid=False,
        dwa_used=False,
        dwa_policy_loss_weight=1.0,
        recovery_mask_applied=False,
        recovery_mask_nonzero_count=0,
        recovery_mask_max=0.0,
    ):
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.pre_tanh_actions.append(
            np.asarray(pre_tanh_action, dtype=np.float32)
        )
        self.raw_actions.append(np.asarray(raw_action, dtype=np.float32))
        self.executed_actions.append(np.asarray(executed_action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.mask_costs.append(float(mask_cost))
        self.task_families.append(str(task_family))
        dwa_raw = (
            np.zeros(2, dtype=np.float32)
            if dwa_raw_action is None
            else np.asarray(dwa_raw_action, dtype=np.float32)
        )
        self.dwa_raw_actions.append(np.clip(dwa_raw, -1.0, 1.0))
        self.dwa_teacher_action_valid.append(bool(dwa_teacher_action_valid))
        self.dwa_used.append(bool(dwa_used))
        self.dwa_policy_loss_weights.append(
            float(np.clip(float(dwa_policy_loss_weight), 0.0, 1.0))
        )
        self.recovery_mask_applied.append(bool(recovery_mask_applied))
        self.recovery_mask_nonzero_counts.append(int(recovery_mask_nonzero_count))
        self.recovery_mask_maxes.append(float(recovery_mask_max))

    def __len__(self):
        return len(self.rewards)


class ContinuousPPOAgent:
    def __init__(self, config=DEFAULT_PPO_CONFIG, device=None):
        self.config = config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if float(config.target_kl) <= 0.0:
            raise ValueError("target_kl must be positive")
        if float(config.kl_early_stop_multiplier) <= 0.0:
            raise ValueError("kl_early_stop_multiplier must be positive")
        self.network = ContinuousActorCritic(
            log_std_init=config.log_std_init,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.network.actor_parameters(),
            lr=config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.network.critic_parameters(),
            lr=config.critic_lr,
        )

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        raw_action, _, log_prob, value = self.act_with_pre_tanh(
            observation,
            deterministic=deterministic,
        )
        return raw_action, log_prob, value

    @torch.no_grad()
    def act_with_pre_tanh(self, observation, deterministic=False):
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)
        pre_tanh, raw_action, log_prob, value = self.network.sample_action(
            obs,
            deterministic=deterministic,
        )
        return (
            raw_action.squeeze(0).cpu().numpy(),
            pre_tanh.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def global_log_std(self):
        return self.network.global_log_std().cpu().numpy().copy()

    @torch.no_grad()
    def value(self, observation):
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)
        return float(self.network.value(obs).item())

    def _advantages_and_returns(self, buffer, last_value, last_done):
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for index in reversed(range(len(rewards))):
            if index == len(rewards) - 1:
                next_value = float(last_value)
                next_nonterminal = 1.0 - float(last_done)
            else:
                next_value = values[index + 1]
                next_nonterminal = 1.0 - dones[index]
            delta = (
                rewards[index]
                + self.config.gamma * next_value * next_nonterminal
                - values[index]
            )
            last_gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_nonterminal
                * last_gae
            )
            advantages[index] = last_gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(
        self,
        buffer,
        last_observation,
        last_done,
        mask_coef=0.0,
        dwa_bc_coef=0.0,
    ):
        if len(buffer) == 0:
            raise ValueError("cannot update PPO from an empty rollout")
        last_value = 0.0 if last_done else self.value(last_observation)
        advantages, returns = self._advantages_and_returns(
            buffer,
            last_value,
            last_done,
        )
        observations = torch.as_tensor(
            np.asarray(buffer.observations),
            dtype=torch.float32,
            device=self.device,
        )
        pre_tanh_actions = torch.as_tensor(
            np.asarray(buffer.pre_tanh_actions),
            dtype=torch.float32,
            device=self.device,
        )
        raw_actions = torch.as_tensor(
            np.asarray(buffer.raw_actions),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            np.asarray(buffer.log_probs),
            dtype=torch.float32,
            device=self.device,
        )
        advantages_t = torch.as_tensor(advantages, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        mask_costs_t = torch.as_tensor(
            np.asarray(buffer.mask_costs, dtype=np.float32),
            device=self.device,
        )
        family_weights_t = torch.as_tensor(
            np.asarray(
                [
                    self._policy_loss_weight(family)
                    for family in buffer.task_families
                ],
                dtype=np.float32,
            ),
            device=self.device,
        )
        dwa_raw_actions_t = torch.as_tensor(
            np.asarray(buffer.dwa_raw_actions, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dwa_teacher_valid_t = torch.as_tensor(
            np.asarray(buffer.dwa_teacher_action_valid, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dwa_policy_weights_t = torch.as_tensor(
            np.asarray(buffer.dwa_policy_loss_weights, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "aux_mask_loss": [],
            "dwa_bc_loss": [],
            "dwa_policy_loss_weight": [],
        }
        epoch_approx_kls = []
        epoch_clip_fractions = []
        rollout_size = len(buffer)
        batch_size = min(int(self.config.batch_size), rollout_size)
        clip_range = float(self.config.clip_range)
        early_stop_threshold = (
            float(self.config.target_kl)
            * float(self.config.kl_early_stop_multiplier)
        )
        early_stopped = False
        for _ in range(int(self.config.ppo_epochs)):
            permutation = np.random.permutation(rollout_size)
            for start in range(0, rollout_size, batch_size):
                indices = torch.as_tensor(
                    permutation[start : start + batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                distribution = self.network.distribution(observations[indices])
                new_log_prob = self.network._squashed_log_prob(
                    distribution,
                    pre_tanh_actions[indices],
                    raw_actions[indices],
                )
                entropy = distribution.entropy().sum(dim=-1)
                log_ratio = new_log_prob - old_log_probs[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages_t[indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - clip_range,
                        1.0 + clip_range,
                    )
                    * advantages_t[indices]
                )
                batch_policy_weights = (
                    family_weights_t[indices] * dwa_policy_weights_t[indices]
                )
                weight_sum = torch.clamp(batch_policy_weights.sum(), min=1e-8)
                policy_loss = -(
                    torch.min(unclipped, clipped) * batch_policy_weights
                ).sum() / weight_sum

                aux_mask_loss = torch.tensor(0.0, device=self.device)
                if mask_coef > 0.0 and len(buffer.mask_costs) == rollout_size:
                    ratio_clip = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    weighted_mask_loss = torch.max(
                        ratio * mask_costs_t[indices],
                        ratio_clip * mask_costs_t[indices],
                    ) * batch_policy_weights
                    aux_mask_loss = weighted_mask_loss.sum() / weight_sum

                dwa_bc_loss = torch.tensor(0.0, device=self.device)
                batch_teacher_valid = dwa_teacher_valid_t[indices] > 0.5
                if float(dwa_bc_coef) > 0.0 and bool(torch.any(batch_teacher_valid)):
                    actor_mean_action = torch.tanh(distribution.mean)
                    bc_error = (
                        actor_mean_action - dwa_raw_actions_t[indices]
                    ).pow(2).mean(dim=-1)
                    dwa_bc_loss = bc_error[batch_teacher_valid].mean()

                actor_loss = (
                    policy_loss
                    - self.config.entropy_coef * entropy.mean()
                    + mask_coef * aux_mask_loss
                    + float(dwa_bc_coef) * dwa_bc_loss
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.actor_parameters(),
                    self.config.max_grad_norm,
                )
                self.actor_optimizer.step()
                self.network.project_log_std()

                values = self.network.value(observations[indices])
                value_loss = 0.5 * (returns_t[indices] - values).pow(2).mean()
                self.critic_optimizer.zero_grad()
                (self.config.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.network.critic_parameters(),
                    self.config.max_grad_norm,
                )
                self.critic_optimizer.step()

                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.mean().item()))
                stats["aux_mask_loss"].append(float(aux_mask_loss.item()))
                stats["dwa_bc_loss"].append(float(dwa_bc_loss.item()))
                stats["dwa_policy_loss_weight"].append(
                    float(dwa_policy_weights_t[indices].mean().item())
                )

            with torch.no_grad():
                new_log_prob, _, _ = self.network.evaluate_actions(
                    observations,
                    pre_tanh_actions,
                    raw_actions,
                )
                log_ratio = new_log_prob - old_log_probs
                ratio = log_ratio.exp()
                epoch_approx_kl = ((ratio - 1.0) - log_ratio).mean()
                epoch_clip_fraction = (
                    (torch.abs(ratio - 1.0) > clip_range).float().mean()
                )
            epoch_approx_kls.append(float(epoch_approx_kl.item()))
            epoch_clip_fractions.append(float(epoch_clip_fraction.item()))
            if epoch_approx_kls[-1] > early_stop_threshold:
                early_stopped = True
                break

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        result = {
            key: float(np.mean(values))
            for key, values in stats.items()
        }
        result.update(
            {
                "approx_kl": float(np.mean(epoch_approx_kls)),
                "approx_kl_mean": float(np.mean(epoch_approx_kls)),
                "approx_kl_max": float(np.max(epoch_approx_kls)),
                "clip_fraction": float(np.mean(epoch_clip_fractions)),
                "ppo_epochs_completed": int(len(epoch_approx_kls)),
                "kl_early_stopped": bool(early_stopped),
                "dwa_bc_coef": float(dwa_bc_coef),
                "dwa_teacher_fraction": float(
                    np.mean(np.asarray(buffer.dwa_teacher_action_valid, dtype=np.float32))
                ),
                "dwa_override_policy_weight_mean": float(
                    np.mean(np.asarray(buffer.dwa_policy_loss_weights, dtype=np.float32))
                ),
            }
        )
        return result

    def _policy_loss_weight(self, task_family):
        weights = {
            "head_in": self.config.policy_loss_weight_head_in,
        }
        if task_family not in weights:
            raise ValueError("unknown task family: {}".format(task_family))
        weight = float(weights[task_family])
        if weight < 0.0:
            raise ValueError("policy loss weights must be non-negative")
        return weight

    def save(self, path, extra=None):
        payload = {
            "network": self.network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "ppo_config": asdict(self.config),
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
