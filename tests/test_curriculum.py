from train.curriculum import CurriculumStageSelector


def test_curriculum_stage_score_uses_worst_recent_family():
    selector = CurriculumStageSelector(
        target_success_rate=0.75,
        history_window=100,
        warmup_episodes=0,
    )

    for _ in range(120):
        selector.record(1, True, task_family="head_in")
        selector.record(1, True, task_family="parallel_fwd")
        selector.record(1, False, task_family="parallel_rev")

    assert selector._stage_success_rate(1) == 0.0


def test_curriculum_record_keeps_stage_only_compatibility():
    selector = CurriculumStageSelector(history_window=100, warmup_episodes=0)
    selector.record(2, True)

    assert selector._stage_success_rate(2) == 1.0
