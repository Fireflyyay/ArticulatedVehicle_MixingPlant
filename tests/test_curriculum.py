from env.mixing_plant_scene import SUPPORTED_SCENE_TYPES
from train.curriculum import CurriculumStageSelector, MultiStageScenePool


def test_curriculum_stage_score_uses_head_in_recent_family():
    selector = CurriculumStageSelector(
        target_success_rate=0.75,
        history_window=100,
        warmup_episodes=0,
    )

    for _ in range(120):
        selector.record(1, True, task_family="head_in")

    assert selector._stage_success_rate(1) == 1.0


def test_curriculum_record_keeps_stage_only_compatibility():
    selector = CurriculumStageSelector(history_window=100, warmup_episodes=0)
    selector.record(2, True)

    assert selector._stage_success_rate(2) == 1.0


def test_curriculum_pool_spans_all_supported_scene_types():
    pool = MultiStageScenePool(
        pool_size=1,
        base_seed=17,
        scene_type_schedule=SUPPORTED_SCENE_TYPES,
    ).pool_for(1)

    requested_scene_types = [
        pool.get(index).metadata["requested_scene_type"]
        for index in range(pool.pool_size)
    ]
    assert pool.pool_size == len(SUPPORTED_SCENE_TYPES)
    assert set(requested_scene_types) == set(SUPPORTED_SCENE_TYPES)


def test_curriculum_scene_replacement_preserves_scene_type_slot():
    pool = MultiStageScenePool(
        pool_size=1,
        base_seed=19,
        scene_type_schedule=SUPPORTED_SCENE_TYPES,
    ).pool_for(1)
    slot = len(SUPPORTED_SCENE_TYPES) - 1
    before = pool.get(slot).metadata["requested_scene_type"]

    pool.replace(slot)

    assert pool.get(slot).metadata["requested_scene_type"] == before
