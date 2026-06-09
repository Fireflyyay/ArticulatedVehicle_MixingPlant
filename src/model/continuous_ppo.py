from dataclasses import dataclass
from typing import Dict

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
            nn.Linear(18, 64),
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
        slot_vehicle = self.slot_vehicle_encoder(obs[..., :18])
        lidar = self.lidar_encoder(obs[..., 18:126])
        mask = self.mask_encoder(obs[..., 126:148])
        return torch.cat([slot_vehicle, lidar, mask], dim=-1)


class ContinuousActorCritic(nn.Module):
    """Separate actor/critic encoders keep their optimizers independent."""

    action_dim = 2

    def __init__(self):
        super().__init__()
        self.actor_encoder = ObservationEncoder()
        self.actor_trunk = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, self.action_dim)
        self.actor_log_std = nn.Linear(256, self.action_dim)

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
        _orthogonal_init(self.actor_log_std, gain=0.01)
        _orthogonal_init(self.critic_head, gain=1.0)

    def actor_parameters(self):
        return list(self.actor_encoder.parameters()) + list(self.actor_trunk.parameters()) + list(
            self.actor_mean.parameters()
        ) + list(self.actor_log_std.parameters())

    def critic_parameters(self):
        return list(self.critic_encoder.parameters()) + list(self.critic_trunk.parameters()) + list(
            self.critic_head.parameters()
        )

    def distribution(self, obs):
        features = self.actor_trunk(self.actor_encoder(obs))
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_log_std(features), -5.0, 1.0)
        return Normal(mean, log_std.exp())

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
        return raw_action, log_prob, self.value(obs)

    def evaluate_raw_actions(self, obs, raw_actions):
        clipped = torch.clamp(raw_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(clipped)
        distribution = self.distribution(obs)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, clipped)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(obs)


class RolloutBuffer:
    """Stores both actions, while PPO likelihoods are tied to raw_action."""

    log_prob_action_source = "raw_action"

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.raw_actions = []
        self.executed_actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(
        self,
        observation,
        raw_action,
        executed_action,
        log_prob,
        reward,
        done,
        value,
    ):
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.raw_actions.append(np.asarray(raw_action, dtype=np.float32))
        self.executed_actions.append(np.asarray(executed_action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def __len__(self):
        return len(self.rewards)


class ContinuousPPOAgent:
    def __init__(self, config=DEFAULT_PPO_CONFIG, device=None):
        self.config = config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.network = ContinuousActorCritic().to(self.device)
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
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)
        raw_action, log_prob, value = self.network.sample_action(
            obs,
            deterministic=deterministic,
        )
        return (
            raw_action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

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

    def update(self, buffer, last_observation, last_done):
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

        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        rollout_size = len(buffer)
        batch_size = min(int(self.config.batch_size), rollout_size)
        for _ in range(int(self.config.ppo_epochs)):
            permutation = np.random.permutation(rollout_size)
            for start in range(0, rollout_size, batch_size):
                indices = torch.as_tensor(
                    permutation[start : start + batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                new_log_prob, entropy, _ = self.network.evaluate_raw_actions(
                    observations[indices],
                    raw_actions[indices],
                )
                log_ratio = new_log_prob - old_log_probs[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages_t[indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range,
                    )
                    * advantages_t[indices]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.actor_parameters(),
                    self.config.max_grad_norm,
                )
                self.actor_optimizer.step()

                values = self.network.value(observations[indices])
                value_loss = 0.5 * (returns_t[indices] - values).pow(2).mean()
                self.critic_optimizer.zero_grad()
                (self.config.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.network.critic_parameters(),
                    self.config.max_grad_norm,
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_range)
                        .float()
                        .mean()
                    )
                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.mean().item()))
                stats["approx_kl"].append(float(approx_kl.item()))
                stats["clip_fraction"].append(float(clip_fraction.item()))

        return {key: float(np.mean(values)) for key, values in stats.items()}

    def save(self, path, extra=None):
        payload = {
            "network": self.network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
