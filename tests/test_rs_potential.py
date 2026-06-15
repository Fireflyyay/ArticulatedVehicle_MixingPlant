from dataclasses import replace
import math

import numpy as np
import pytest

from config import DEFAULT_ENV_CONFIG
from env.geometry import DirectedParkingSlot, wrap_to_pi
from env.local_parking_env import LocalParkingEnv
from env.rs_potential import RSPlanResult, RSPotentialOracle, RSPotentialPlanner
from env.vehicle import ArticulatedState
from planning.passenger_hybrid_astar import PlannerResult
from planning.reeds_shepp import generate_reeds_shepp_paths


class RecordingRSPlanner:
    def __init__(self, valid=True, reason="success"):
        self.valid = bool(valid)
        self.reason = str(reason)
        self.calls = 0

    def plan(self, scene, state, slot):
        self.calls += 1
        if not self.valid:
            return RSPlanResult(
                valid=False,
                path=None,
                reason=self.reason,
                total_time_ms=0.1,
            )
        path = np.asarray(
            [
                [state.x_front, state.y_front, state.theta_front],
                [
                    0.5 * (state.x_front + slot.x_goal),
                    0.5 * (state.y_front + slot.y_goal),
                    slot.theta_goal,
                ],
                [slot.x_goal, slot.y_goal, slot.theta_goal],
            ],
            dtype=np.float64,
        )
        return RSPlanResult(
            valid=True,
            path=path,
            reason="success",
            total_length=float(
                math.hypot(state.x_front - slot.x_goal, state.y_front - slot.y_goal)
            ),
            candidate_count=2,
            checked_candidates=1,
            collision_checks=3,
            sample_count=3,
            total_time_ms=0.1,
        )


class StraightHybridPlanner:
    def __init__(self):
        self.calls = 0

    def plan_with_cost(self, scene, state, slot):
        self.calls += 1
        path = np.asarray(
            [
                [state.x_front, state.y_front, state.theta_front],
                [slot.x_goal, slot.y_goal, slot.theta_goal],
            ],
            dtype=np.float32,
        )
        return PlannerResult(
            valid=True,
            path=path,
            cost=float(
                math.hypot(state.x_front - slot.x_goal, state.y_front - slot.y_goal)
            ),
            goal_cost=0.0,
            reason="success",
            expanded_nodes=2,
        )


def _state(x, y=0.0, theta=0.0):
    return ArticulatedState(
        x_front=float(x),
        y_front=float(y),
        theta_front=float(theta),
        theta_rear=float(theta),
    )


def _slot():
    return DirectedParkingSlot(
        x_goal=0.0,
        y_goal=0.0,
        theta_goal=0.0,
        front_body_length=4.45,
        front_body_width=3.016,
    )


def test_rs_disabled_preserves_hybrid_reward_path(synthetic_action_mask):
    hybrid = StraightHybridPlanner()
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=False,
            use_hybrid_astar=True,
        ),
        action_mask=synthetic_action_mask,
        hybrid_planner=hybrid,
        seed=2,
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["planner_source"] == "hybrid_astar"
    assert info["rs_latched"] is False
    assert info["reward_components"]["rs_potential"] == 0.0
    assert info["reward_components"]["planner"] == pytest.approx(
        info["reward_components"]["hybrid_astar"]
    )


def test_outside_d_rs_does_not_call_planner():
    planner = RecordingRSPlanner()
    oracle = RSPotentialOracle(planner=planner, d_rs=10.0)
    reward, info = oracle.step(None, _state(12.5), _state(12.0), _slot())

    assert planner.calls == 0
    assert reward == 0.0
    assert info["rs_fail_reason"] == "outside_d_rs"
    assert info["rs_attempt_count"] == 0


def test_collision_free_rs_path_latches_and_planner_is_not_called_again():
    planner = RecordingRSPlanner()
    oracle = RSPotentialOracle(planner=planner, d_rs=10.0)

    _, first_info = oracle.step(None, _state(9.0), _state(8.0), _slot())
    _, second_info = oracle.step(None, _state(8.0), _state(7.0), _slot())

    assert first_info["rs_latched"] is True
    assert first_info["rs_success_count"] == 1
    assert second_info["rs_planner_disabled_after_latch"] is True
    assert planner.calls == 1


def test_rs_latch_suppresses_hybrid_reward_on_same_transition(
    synthetic_action_mask,
):
    hybrid = StraightHybridPlanner()
    rs_planner = RecordingRSPlanner()
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=True,
            rs_potential_d_rs=10.0,
            use_hybrid_astar=True,
        ),
        action_mask=synthetic_action_mask,
        hybrid_planner=hybrid,
        rs_planner=rs_planner,
        seed=4,
    )
    env.reset()
    axis = np.asarray(
        [math.cos(env.slot.theta_goal), math.sin(env.slot.theta_goal)]
    )
    center = np.asarray(env.slot.center) - 5.0 * axis
    env.state = ArticulatedState(
        x_front=float(center[0]),
        y_front=float(center[1]),
        theta_front=float(env.slot.theta_goal),
        theta_rear=float(env.slot.theta_goal),
    )
    env._update_sensors_and_mask()

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["rs_latched"] is True
    assert info["planner_source"] == "rs"
    assert info["hybrid_astar_suppressed_by_rs"] is True
    assert info["reward_components"]["hybrid_astar"] == 0.0
    assert info["reward_components"]["planner"] == pytest.approx(info["rs_reward"])
    assert hybrid.calls == 1


def test_colliding_rs_candidates_do_not_latch_or_reward():
    class AlwaysOccupied:
        def _is_rectangle_occupied(self, scene, x, y, theta):
            return True

    planner = RSPotentialPlanner(
        collision_checker=AlwaysOccupied(),
        turning_radius=6.4,
        candidate_limit=2,
        sample_step=0.3,
    )
    result = planner.plan(None, _state(5.0), _slot())
    oracle = RSPotentialOracle(planner=RecordingRSPlanner(False, "collision"))
    reward, info = oracle.step(None, _state(6.0), _state(5.0), _slot())

    assert result.valid is False
    assert result.reason == "collision"
    assert info["rs_latched"] is False
    assert info["rs_fail_reason"] == "collision"
    assert reward == 0.0


def test_rs_failure_has_no_fallback_planner_reward_without_hybrid(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=True,
            rs_potential_d_rs=10.0,
            use_hybrid_astar=False,
        ),
        action_mask=synthetic_action_mask,
        rs_planner=RecordingRSPlanner(False, "collision"),
        seed=4,
    )
    env.reset()
    axis = np.asarray(
        [math.cos(env.slot.theta_goal), math.sin(env.slot.theta_goal)]
    )
    center = np.asarray(env.slot.center) - 5.0 * axis
    env.state = ArticulatedState(
        x_front=float(center[0]),
        y_front=float(center[1]),
        theta_front=float(env.slot.theta_goal),
        theta_rear=float(env.slot.theta_goal),
    )
    env._update_sensors_and_mask()
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["rs_fail_reason"] == "collision"
    assert info["planner_source"] == "none"
    assert info["reward_components"]["planner"] == 0.0
    assert info["planner_potential_reward"] == 0.0


def test_success_rejects_reverse_and_heading_over_15_degrees(
    synthetic_action_mask,
):
    env = LocalParkingEnv(
        config=replace(
            DEFAULT_ENV_CONFIG,
            rs_potential_enabled=False,
            use_hybrid_astar=False,
            success_overlap=0.0,
        ),
        action_mask=synthetic_action_mask,
        seed=6,
    )
    env.reset()

    for heading_error in (math.pi, math.radians(15.1)):
        env.state = ArticulatedState(
            env.slot.x_goal,
            env.slot.y_goal,
            wrap_to_pi(env.slot.theta_goal + heading_error),
            env.slot.theta_goal,
        )
        env._update_sensors_and_mask()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["success"] is False


def test_rs_reward_is_transition_level_potential_difference():
    planner = RecordingRSPlanner()
    oracle = RSPotentialOracle(
        planner=planner,
        d_rs=10.0,
        gamma=0.98,
        cost_scale=10.0,
        potential_coef=1.0,
        potential_clip=10.0,
        lateral_weight=0.0,
        heading_weight=0.0,
    )
    oracle.step(None, _state(9.0), _state(8.0), _slot())
    previous_cost = float(oracle.rs_last_cost)
    reward, info = oracle.step(None, _state(8.0), _state(7.0), _slot())
    current_cost = float(info["rs_cost"])
    expected = 0.98 * (-current_cost / 10.0) - (-previous_cost / 10.0)

    assert reward == pytest.approx(expected)
    assert reward > 0.0
    assert info["rs_remaining_length"] < previous_cost


def test_rs_candidates_include_extended_scs_family_and_directed_endpoint():
    goal = (-8.0, -8.0, -7.0 * math.pi / 8.0)
    paths = generate_reeds_shepp_paths(
        start=(0.0, 0.0, 0.0),
        goal=goal,
        turning_radius=6.4,
        sample_step=0.3,
    )
    scs_paths = [
        path
        for path in paths
        if path.segment_types in (["S", "L", "S"], ["S", "R", "S"])
    ]

    assert scs_paths
    endpoint = scs_paths[0].poses[-1]
    assert endpoint[:2] == pytest.approx(goal[:2], abs=1e-6)
    assert abs(wrap_to_pi(endpoint[2] - goal[2])) <= 1e-6


def test_rs_candidate_sampling_limits_reject_singular_scs_paths():
    goal = (0.0, 10.0, 1e-4)
    paths = generate_reeds_shepp_paths(
        start=(0.0, 0.0, 0.0),
        goal=goal,
        turning_radius=6.4,
        sample_step=1.0 / 3.0,
    )

    assert paths
    assert max(path.total_length for path in paths) <= 256.0 + 1e-6
    assert max(len(path.poses) for path in paths) <= 8192
    assert all(
        path.segment_types not in (["S", "L", "S"], ["S", "R", "S"])
        for path in paths
    )


# ---------------------------------------------------------------------------
# Helpers for articulated collision-check tests
# ---------------------------------------------------------------------------

from config import ZL50GNVehicleParams
from planning.passenger_hybrid_astar import PassengerHybridAStar
from planning.reeds_shepp import ReedsSheppPath

_VP = ZL50GNVehicleParams()
_LF = float(_VP.front_center_to_hinge)    # 2.225
_LR = float(_VP.rear_center_to_hinge)    # 1.8575
_PHI_MAX = float(_VP.phi_max)            # 0.6109 rad


class _ClearScene:
    """Scene that reports every point as free."""
    world_bounds = (-40, -40, 40, 40)
    resolution = 0.1
    @staticmethod
    def is_occupied_world(x, y):
        return False


class _BoxScene:
    """Scene that returns occupied=True inside a given world-box."""
    world_bounds = (-40, -40, 40, 40)
    resolution = 0.1
    def __init__(self, xmin, ymin, xmax, ymax):
        self._xmin = float(xmin)
        self._ymin = float(ymin)
        self._xmax = float(xmax)
        self._ymax = float(ymax)
    def is_occupied_world(self, x, y):
        return self._xmin <= x <= self._xmax and self._ymin <= y <= self._ymax


class _RecordingCollisionChecker:
    intermediate_checks = 2

    def __init__(self):
        self.recorded = []

    def _is_rectangle_occupied(self, scene, x, y, theta):
        return False

    def _check_two_bodies(self, scene, x_f, y_f, theta_f, x_r, y_r, theta_r):
        self.recorded.append(
            (
                float(x_f),
                float(y_f),
                float(theta_f),
                float(x_r),
                float(y_r),
                float(theta_r),
            )
        )
        return False


def _articulated_checker():
    return PassengerHybridAStar(
        front_half_length=0.5 * _VP.front_body_length,
        front_half_width=0.5 * _VP.front_body_width,
        rear_half_length=0.5 * _VP.rear_body_length,
        rear_half_width=0.5 * _VP.rear_body_width,
        front_center_to_hinge=_LF,
        rear_center_to_hinge=_LR,
        footprint_samples=3,
        intermediate_checks=2,
    )


def _make_rs_path(lengths, seg_types, poses):
    total = sum(abs(float(v)) for v in lengths)
    n = len(poses)
    static_dir = 1 if lengths[0] >= 0 else -1
    return ReedsSheppPath(
        lengths=[float(v) for v in lengths],
        segment_types=list(seg_types),
        total_length=float(total),
        poses=np.asarray(poses, dtype=np.float64),
        directions=np.full(n, static_dir, dtype=np.int8),
    )


def _make_planner(checker=None, vp=_VP):
    if checker is None:
        checker = _articulated_checker()
    return RSPotentialPlanner(
        collision_checker=checker,
        turning_radius=float(_VP.minimum_turning_radius),
        candidate_limit=2,
        sample_step=0.3,
        vehicle_params=vp,
    )


# ---------------------------------------------------------------------------
# Articulated collision-check tests
# ---------------------------------------------------------------------------

def test_no_rear_params_preserves_old_behavior():
    """vehicle_params=None 时退化为单矩形旧行为."""
    class FrontOnly:
        def _is_rectangle_occupied(self, scene, x, y, theta):
            return True
    planner = RSPotentialPlanner(
        collision_checker=FrontOnly(),
        turning_radius=6.4,
        candidate_limit=2,
        sample_step=0.3,
        vehicle_params=None,
    )
    path = _make_rs_path([5.0], ["S"], [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    free, checks = planner._collision_free(None, path)
    assert free is False
    assert checks > 0


def test_front_body_collision_rejects():
    """前车体矩形占据 -> 拒绝."""
    scene = _BoxScene(xmin=-1.0, ymin=-1.0, xmax=1.0, ymax=1.0)
    # 起点 (0,0,0) 处前车体矩形覆盖 [-2.225, 2.225]x[-1.508,1.508] → 包含障碍 (0,0)
    path = _make_rs_path([5.0], ["S"], [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    planner = _make_planner()
    free, checks = planner._collision_free(scene, path, initial_phi=0.0)
    assert free is False
    assert checks > 0


def test_rear_body_collision_rejects():
    """前车体 clear 但后车体占据 -> 拒绝."""
    scene = _BoxScene(xmin=-5.0, ymin=-1.0, xmax=-2.5, ymax=1.0)
    # 起点 (0,0,0) phi=0: 后车体矩形 [-5.94, -2.225]x[-1.508,1.508] → 包含 (-4,0)
    path = _make_rs_path([5.0], ["S"], [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    planner = _make_planner()
    free, checks = planner._collision_free(scene, path, initial_phi=0.0)
    assert free is False
    assert checks > 0


def test_intermediate_sweep_catches_tunneling():
    """扫掠中间点（非原始 RS 采样点）检测障碍."""
    # 钝角障碍仅被 sweep_ds=0.1m 采样点覆盖，不会被 0.3m 原始采样跳过
    scene = _BoxScene(xmin=0.05, ymin=-1.0, xmax=0.25, ymax=1.0)
    # 前向直行: 0→5m, 障碍在 x=0.05~0.25 区间
    # sweep_ds ≈ 0.3/(2+1) = 0.1, 障碍宽度 0.2m = 2 个 sweep 步 → 可命中
    path = _make_rs_path([5.0], ["S"], [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    planner = _make_planner()
    free, checks = planner._collision_free(scene, path, initial_phi=0.0)
    assert free is False
    assert checks > 0


def test_phi_exceeds_max_articulation_rejects():
    """phi 积分超过 phi_max → 拒绝.

    倒车直行使用负 ds，使 phi 沿倒车距离增长；在 phi_max 处每倒车
    1m 增长约 0.31rad，不到一个 sweep 步就超限."""
    scene = _ClearScene()
    path = _make_rs_path([-0.5], ["S"], [[0.0, 0.0, 0.0]])
    planner = _make_planner()
    free, checks = planner._collision_free(scene, path, initial_phi=_PHI_MAX)
    assert free is False
    assert checks > 0


def test_forward_straight_phi_converges_to_zero():
    """前进直行: 非零 initial_phi → phi 衰减到 0, 不超限."""
    scene = _ClearScene()
    path = _make_rs_path([5.0], ["S"], [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    planner = _make_planner()
    free, checks = planner._collision_free(scene, path, initial_phi=0.5)
    assert free is True
    assert checks > 0


def test_reverse_straight_phi_diverges():
    """倒车直行: 非零 initial_phi → phi 发散, 超过 phi_max 后拒绝."""
    scene = _ClearScene()
    # 倒车直行 3m: dphi/ds=-sin(phi)/lr 且 ds<0，因此 phi 增长。
    # 初始 0.3 → 约 0.8m 后超 phi_max
    path = _make_rs_path([-3.0], ["S"], [[0.0, 0.0, 0.0]])
    planner = _make_planner()
    free, checks = planner._collision_free(scene, path, initial_phi=0.3)
    assert free is False
    assert checks > 0


def test_phi_continuous_at_direction_reversal():
    """换向段: phi 不重置, 连续递推."""
    scene = _ClearScene()
    # 两段: 前进直行 1m, 倒车直行 1m. gear 从 +1 变 -1.
    # phi 始终连续, 不重置为 0.
    path = _make_rs_path([1.0, -1.0], ["S", "S"],
                         [[0.0, 0.0, 0.0]])
    planner = _make_planner()
    # initial_phi=0.3 → 前进 1m phi 略降, 倒车 1m phi 又上升
    # 整体应在 phi_max 内
    free, checks = planner._collision_free(scene, path, initial_phi=0.3)
    # 前进 1m: phi ≈ 0.3 - roughly 0.145 = 0.155
    # 倒车 1m: phi ≈ 0.155 + roughly 0.155 = 0.31 (sin(0.155)≈0.154)
    # 远小于 phi_max=0.611 → pass
    assert free is True
    assert checks > 0


def test_segment_curvature_used_for_L_R_S():
    """段级曲率 (L→+1/R, S→0) 作为主曲率来源, 影响 phi 递推.

    通过 RecordingCollisionChecker 记录每次 _check_two_bodies 的 (theta_f, theta_r),
    反推 phi = theta_f - theta_r. 验证 L 段后 phi > 0, S 段后 phi ≈ 0."""

    class RecordingCollisionChecker:
        def __init__(self):
            self.inner = _articulated_checker()
            self.recorded = []   # (theta_f, theta_r, phi)
        def _is_rectangle_occupied(self, *a):
            return self.inner._is_rectangle_occupied(*a)
        def _check_two_bodies(self, scene, x_f, y_f, theta_f, x_r, y_r, theta_r):
            phi = wrap_to_pi(theta_f - theta_r)
            self.recorded.append((theta_f, theta_r, phi))
            return self.inner._check_two_bodies(
                scene, x_f, y_f, theta_f, x_r, y_r, theta_r
            )

    scene = _ClearScene()
    path_L = _make_rs_path([2.0], ["L"], [[0.0, 0.0, 0.0]])
    path_S = _make_rs_path([2.0], ["S"], [[0.0, 0.0, 0.0]])

    rec_L = RecordingCollisionChecker()
    planner_L = _make_planner(checker=rec_L)
    planner_L._collision_free(scene, path_L, initial_phi=0.0)

    rec_S = RecordingCollisionChecker()
    planner_S = _make_planner(checker=rec_S)
    planner_S._collision_free(scene, path_S, initial_phi=0.0)

    _, _, phi_L = rec_L.recorded[-1] if rec_L.recorded else (0.0, 0.0, 0.0)
    _, _, phi_S = rec_S.recorded[-1] if rec_S.recorded else (0.0, 0.0, 0.0)
    assert phi_L > 0.1, f"L 段应产生正 phi, 实际 {phi_L:.4f}"
    assert abs(phi_S) < 0.01, f"S 段应保持 phi≈0, 实际 {phi_S:.4f}"


def test_reverse_direction_changes_phi_dynamics():
    """同一有符号 ODE 下，倒车直行 phi 发散而前进直行收敛."""
    scene = _ClearScene()
    # 前进 3m, gear=+1, initial_phi=0.3 → phi 衰减 → 通过
    path_fwd = _make_rs_path([3.0], ["S"], [[0.0, 0.0, 0.0]])
    # 倒车 3m, gear=-1, initial_phi=0.3 → phi 增长 → 超限拒绝
    path_rev = _make_rs_path([-3.0], ["S"], [[0.0, 0.0, 0.0]])
    planner = _make_planner()
    free_fwd, _ = planner._collision_free(scene, path_fwd, initial_phi=0.3)
    free_rev, _ = planner._collision_free(scene, path_rev, initial_phi=0.3)
    assert free_fwd is True
    assert free_rev is False


@pytest.mark.parametrize("seg_type", ["S", "L", "R"])
def test_reverse_segment_reconstruction_matches_rs_endpoint(seg_type):
    """负长度直线和圆弧必须按倒车方向重建前车终点."""
    radius = float(_VP.minimum_turning_radius)
    seg_len = -2.0
    if seg_type == "S":
        expected = (-2.0, 0.0, 0.0)
    else:
        kappa = (1.0 if seg_type == "L" else -1.0) / radius
        dtheta = kappa * seg_len
        expected = (
            math.sin(dtheta) / kappa,
            (1.0 - math.cos(dtheta)) / kappa,
            dtheta,
        )

    path = _make_rs_path(
        [seg_len],
        [seg_type],
        [[0.0, 0.0, 0.0], list(expected)],
    )
    checker = _RecordingCollisionChecker()
    planner = _make_planner(
        checker=checker,
        vp=replace(_VP, phi_max=100.0),
    )

    free, _ = planner._collision_free(_ClearScene(), path, initial_phi=0.0)

    assert free is True
    assert checker.recorded
    x_f, y_f, theta_f = checker.recorded[-1][:3]
    assert (x_f, y_f) == pytest.approx(expected[:2], abs=1e-9)
    assert wrap_to_pi(theta_f - expected[2]) == pytest.approx(0.0, abs=1e-9)


def test_generated_reverse_candidates_reconstruct_their_sampled_endpoints():
    """真实 RS 候选的碰撞扫掠终点必须与候选采样终点一致."""
    paths = generate_reeds_shepp_paths(
        start=(0.0, 0.0, 0.0),
        goal=(4.0, -3.0, 1.1),
        turning_radius=float(_VP.minimum_turning_radius),
        sample_step=0.3,
    )
    reverse_paths = [
        path for path in paths if any(length < 0.0 for length in path.lengths)
    ][:5]
    assert len(reverse_paths) == 5

    unconstrained_vp = replace(_VP, phi_max=100.0)
    for path in reverse_paths:
        checker = _RecordingCollisionChecker()
        planner = _make_planner(checker=checker, vp=unconstrained_vp)
        free, _ = planner._collision_free(
            _ClearScene(), path, initial_phi=0.0
        )

        assert free is True
        endpoint = path.poses[-1]
        x_f, y_f, theta_f = checker.recorded[-1][:3]
        assert (x_f, y_f) == pytest.approx(endpoint[:2], abs=1e-8)
        assert wrap_to_pi(theta_f - endpoint[2]) == pytest.approx(
            0.0, abs=1e-8
        )
