import numpy as np

from env.mixing_plant_scene import CachedScenePool, TASK_FAMILIES


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
        self._family_results = {
            (stage, family): []
            for stage in (1, 2, 3, 4)
            for family in TASK_FAMILIES
        }
        self._rng = np.random.default_rng()

    def _recent_rate(self, values):
        recent = values[-self.history_window:]
        return sum(recent) / max(len(recent), 1)

    def _stage_success_rate(self, stage):
        family_rates = []
        for family in TASK_FAMILIES:
            recent = self._family_results[(stage, family)][-self.history_window:]
            if recent:
                family_rates.append(sum(recent) / len(recent))
        if family_rates:
            return min(family_rates)
        return self._recent_rate(self._results[stage])

    def select_stage(self, episode_index):
        if episode_index < self.warmup_episodes:
            return int(self._rng.integers(1, 5))

        if episode_index < self.history_window:
            return int(self._rng.integers(1, 5))

        if self._rng.random() < 0.5:
            return int(self._rng.integers(1, 5))

        success_rates = []
        for stage in (1, 2, 3, 4):
            success_rates.append(self._stage_success_rate(stage))

        fail_rates = [
            max(self.target_success_rate - sr, 0.01)
            for sr in success_rates
        ]
        total = sum(fail_rates)
        probs = [fr / total for fr in fail_rates]
        return int(self._rng.choice([1, 2, 3, 4], p=probs))

    def record(self, stage, success, task_family=None):
        stage = int(stage)
        if stage in self._results:
            value = 1.0 if success else 0.0
            self._results[stage].append(value)
            family = str(task_family or "")
            if (stage, family) in self._family_results:
                self._family_results[(stage, family)].append(value)
