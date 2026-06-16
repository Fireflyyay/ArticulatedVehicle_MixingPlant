# AGENTS.md

## Scope

This repository implements only near-goal articulated-loader parking and local
recovery. Do not add global navigation, global topology decisions, or a global
SMDP here.

## Runtime

- Preferred environment: `conda run -n HOPE ...`
- Python: 3.8 compatible syntax is required.
- Direct module runs use `PYTHONPATH=src`.
- Run tests with `conda run -n HOPE pytest -q`.
- Local training is episode-count based via `--total-episodes`. Default run
  artifacts belong under the repository `runs/` directory.
- Every training run must snapshot effective configuration to `config.txt` and
  update `reward_curve.png` every 10 completed episodes.

## Core Contracts

- Vehicle parameters have one source of truth:
  `src/config.py::ZL50GNVehicleParams`.
- State reference is the front-body geometric center.
- `phi = wrap(theta_front - theta_rear)` is always wrapped to `[-pi, pi]`.
- Policy action is normalized continuous `[v_cmd_norm, phi_dot_cmd_norm]`.
- Map `phi_dot_cmd_norm` linearly into the current executable `phi_dot`
  interval given `phi`, `phi_max`, `phi_dot_max`, and `dt`; do not scale to
  the symmetric rate limit and then clip against the articulation-angle limit.
- PPO log probability is computed for the raw squashed policy action.
- The continuous actor outputs only the pre-tanh Gaussian mean. Exploration
  uses one global learnable `log_std` vector, bounded by `PPOConfig`.
- Rollouts store both `pre_tanh_action` and `raw_action`; PPO likelihood
  recomputation must use those saved policy samples, never `executed_action`.
- PPO updates use epoch-level target-KL early stopping and task-family actor
  loss weights. Family best checkpoints use fixed deterministic evaluation.
- `LocalParkingEnv.step()` applies the action mask and advances the model with
  `executed_action`; both actions must remain in diagnostics and rollouts.
- Observation order and slices are defined by
  `LocalParkingEnv.OBS_SLICES`; current dimension is 148.
- LiDAR is exactly 54 front-body-center beams plus 54 rear-body-center beams.
  Online LiDAR uses vectorized line-segment intersection and must not include
  either vehicle body as an obstacle.
- Action mask is 2 x 11 max-safe-speed ratios. It is a policy input and a hard
  execution-layer speed constraint.
- Online mask computation may only compare current front/rear LiDAR vectors
  with the offline sweep table. Do not add per-candidate online rollouts.
- Regenerate `assets/action_mask/zl50gn_articulated_mask.npz` after changing
  vehicle geometry, dynamics, LiDAR beams, mask bins, or mask horizon:
  `conda run -n HOPE python scripts/generate_articulated_action_mask_table.py`.
- Parking success uses front-body target overlap >= 0.80 and wrapped front
  heading error <= 15 degrees. Rear overlap and rear heading are diagnostics,
  not success conditions.
- Collision termination uses real front/rear polygons against scene obstacles.
  LiDAR and the action mask are not collision truth.
- Hybrid A* is optional auxiliary shaping. Planner failure returns zero hybrid
  reward and must not terminate the episode or remove base reward terms.
- RS potential is near-goal reward guidance only. It may latch one
  collision-free directed RS path per episode; after latch it replaces the
  Hybrid A* contribution and never controls the continuous action.

## Scene Policy

- Mixing-plant scenes start blocked and constructively carve known-free
  corridors, bays, and maneuver areas.
- Scene bounds are `[-40, 40] x [-40, 40]` meters. Target poses must remain
  fully inside the target Bay.
- Supported target orientation modes are `head_in` (front points from the Bay
  mouth toward the far wall) and `parallel` (vehicle axis parallel to the
  attached corridor).
- Every scene seed maps to a reproducible parameterized layout. Seed controls
  corridor orientation/origin, Bay mode/side/position, branch positions, and
  parallel goal direction. Do not collapse seeds to a small template modulo.
- Scene variants are cached. Do not replace constructive scene generation with
  random geometry followed by repeated rule rejection.
- Initial-state ranges are configured in `LocalParkingEnvConfig`; keep
  near-goal, poor-terminal-pose, and recovery distributions diverse and
  collision-free.
- Keep obstacle edges cached for LiDAR. Avoid per-beam or per-obstacle Shapely
  calls in training steps.

## Reward Policy

Keep the reward intentionally small:

- terminal success/failure
- history-best front overlap improvement
- HOPE Appendix B initial-distance-relative proximity:
  `-(D(t) - D(0)) / max(D(0), distance_d_min)`
- history-best heading-score improvement
- light time penalty
- optional Hybrid A* progress/lateral term

Do not add dense steering smoothness, gear-change, strong path-tracking, or
continuous obstacle-distance penalties without explicit evidence and tests.

## Validation

Before handing off changes:

1. Regenerate the action-mask asset when its compatibility inputs change.
2. Run `conda run -n HOPE pytest -q`.
3. Run a short training smoke test.
4. Render at least one scene when scene or geometry code changes:
   `conda run -n HOPE python scripts/visualize_local_parking_scene.py --stage 3`.
