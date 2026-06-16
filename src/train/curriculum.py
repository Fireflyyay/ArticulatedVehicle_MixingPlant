import numpy as np

from env.mixing_plant_scene import CachedScenePool


class MultiStageScenePool:
    def __init__(
        self,
        pool_size=16,
        base_seed=0,
        scene_config=None,
        family_schedule=None,
    ):
        self._pools = {}
        self._pool_size = int(pool_size)
        self._base_seed = int(base_seed)
        self._scene_config = scene_config
        self._family_schedule = family_schedule

    def pool_for(self, stage):
        stage = int(stage)
        if stage not in self._pools:
            self._pools[stage] = CachedScenePool(
                stage=stage,
                pool_size=self._pool_size,
                base_seed=self._base_seed,
                scene_config=self._scene_config,
                family_schedule=self._family_schedule,
            )
        return self._pools[stage]


class CurriculumStageSelector:
    def __init__(
        self,
        target_success_rate=0.75,
        history_window=2000,
        warmup_episodes=200,
    ):
        self.target_success_rate = float(target_success_rate)
        self.history_window = int(max(100, history_window))
        self.warmup_episodes = int(max(0, warmup_episodes))
        self._results = {stage: [] for stage in (1, 2, 3, 4)}
        self._rng = np.random.default_rng()

    def select_stage(self, episode_index):
        if episode_index < self.warmup_episodes:
            return int(self._rng.integers(1, 5))

        if episode_index < self.history_window:
            return int(self._rng.integers(1, 5))

        if self._rng.random() < 0.5:
            return int(self._rng.integers(1, 5))

        success_rates = []
        for stage in (1, 2, 3, 4):
            recent = self._results[stage][-self.history_window:]
            success_rates.append(sum(recent) / max(len(recent), 1))

        fail_rates = [
            max(self.target_success_rate - sr, 0.01)
            for sr in success_rates
        ]
        total = sum(fail_rates)
        probs = [fr / total for fr in fail_rates]
        return int(self._rng.choice([1, 2, 3, 4], p=probs))

    def record(self, stage, success):
        stage = int(stage)
        if stage in self._results:
            self._results[stage].append(1.0 if success else 0.0)
