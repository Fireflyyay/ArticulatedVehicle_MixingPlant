# Articulated Vehicle Local Parking RL

Near-goal parking and local recovery for a ZL50GN-style articulated loader.
The package intentionally excludes global navigation and global SMDP logic.

Cached mixing-plant scenes cover `[-40, 40] x [-40, 40]` meters and are built
constructively from wide main corridors, branches, and attached parking bays.
Target poses are always inside a bay and alternate between head-in and
corridor-parallel parking orientations.

## Quick start

```bash
conda run -n HOPE python scripts/generate_articulated_action_mask_table.py
conda run -n HOPE pytest -q
conda run -n HOPE python scripts/visualize_local_parking_scene.py --stage 3
conda run -n HOPE python scripts/train_local_parking.py --total-episodes 1000
```

Direct module commands require the repository `src` directory on `PYTHONPATH`:

```bash
PYTHONPATH=src conda run -n HOPE python -m train.train_local_parking
```

```bash
conda run -n HOPE python scripts/visualize_local_parking_paths.py \
  --stage 3 \
  --checkpoint  /home/cyberbus/Public/ArticulatedVehicle_MixingPlant/runs/local_parking_20260610_105416_seed0/checkpoint_episode_007800.pt \
  --num-paths 8 \
  --seed 0 \
  --output outputs/paths/stage3_paths.png
```

Training is terminated by completed episode count. By default, each run is
written to `runs/local_parking_<timestamp>_seedN/`. The directory contains
`config.txt`, per-episode and PPO-update JSONL metrics, TensorBoard events,
checkpoints, and `reward_curve.png`. The reward figure is refreshed every 10
episodes with Episode on the x-axis and Reward on the y-axis.

The environment observation has 149 values:

- directed slot features: 13
- vehicle features: 6
- front/rear LiDAR: 108
- continuous action-mask features: 22

The policy emits the raw normalized continuous action
`[v_cmd_norm, phi_dot_cmd_norm]`. PPO log probabilities are computed for this
raw action. The environment then filters and clips it with the offline
articulated sweep table before advancing the vehicle model.

## ICRA Ablation Suite

The main paper method is `full`: articulated local parking with dual-body
LiDAR, geometry-aware safe action-mask execution, PPO, adaptive curriculum,
and optional near-goal RS potential shaping. RS is a
potential-shaping signal only; it is not a teacher or expert action source.
HOPE teacher code remains diagnostic and is not part of the main ablation
table.

Launch the complete suite from the repository root:

```bash
TOTAL_EPISODES=20000 \
SEEDS="0 1 2" \
DEVICE=cuda \
CONDA_ENV=HOPE \
USE_CONDA=1 \
nohup bash scripts/launch_ablation_suite.sh > runs/ablation_suite/nohup.out 2>&1 &
```

Each run writes `command.txt`, `train.log`, `config.txt`, metrics JSONL files,
checkpoints, and reward plots under
`runs/ablation_suite/<timestamp>/<experiment>/seed_<seed>/`. With `DEVICE=cuda`
the launcher forces `MAX_PARALLEL=1` to avoid multiple processes contending for
one GPU. With `DEVICE=cpu`, it can launch multiple independent training
processes; set `MAX_PARALLEL` to control the queue width. If `MAX_PARALLEL` is
unset on CPU, the launcher chooses a conservative value from the available CPU
count. It also exports per-run BLAS/OpenMP thread limits and waits when
available memory drops below `MIN_AVAILABLE_MEM_GB_PER_JOB`.

CPU parallel example:

```bash
TOTAL_EPISODES=20000 \
SEEDS="0 1 2" \
DEVICE=cpu \
MAX_PARALLEL=4 \
CONDA_ENV=HOPE \
USE_CONDA=1 \
nohup bash scripts/launch_ablation_suite.sh > runs/ablation_suite/nohup.out 2>&1 &
```

Ablation semantics:

- `no_rs_potential` disables only near-goal RS potential shaping, not a teacher.
- `no_mask_observation` zeroes the mask observation slice but keeps execution
  safety masking enabled.
- `no_mask_cost` removes the PPO mask penalty but keeps execution safety
  masking enabled.
- `front_lidar_only` zeroes only rear LiDAR observations; rear-body collision
  checking and action-mask truth still use real dual-body geometry.
- `dwa_assisted` is an execution-time fallback diagnostic, not the default
  learning method.
- `unsafe_no_action_mask_execution` is an unsafe diagnostic and should be kept
  out of fair main-table comparisons.

### Individual ablation nohup commands

Launch each ablation from the repository root. All commands share the base
flags `--log-std-init 0.0 --log-std-max 0.0 --total-episodes 50000` and the
`runs/ablation_suite/<name>/` output directory.

```bash
# A0 — full (main method)
mkdir -p runs/ablation_suite/full && nohup conda run -n HOPE python scripts/train_local_parking.py \
  --log-std-init 0.0 --log-std-max 0.0 \
  --curriculum --curriculum-mode adaptive \
  --disable-dwa-recovery \
  --mask-cost-coef-final 0.8 \
  --total-episodes 50000 \
  > runs/ablation_suite/full/nohup.out 2>&1 &

# A1 — no_rs_potential
mkdir -p runs/ablation_suite/no_rs_potential && nohup conda run -n HOPE python scripts/train_local_parking.py \
  --log-std-init 0.0 --log-std-max 0.0 \
  --curriculum --curriculum-mode adaptive \
  --disable-dwa-recovery \
  --disable-rs-potential \
  --mask-cost-coef-final 0.8 \
  --total-episodes 50000 \
  > runs/ablation_suite/no_rs_potential/nohup.out 2>&1 &

# A2 — uniform_curriculum
mkdir -p runs/ablation_suite/uniform_curriculum && nohup conda run -n HOPE python scripts/train_local_parking.py \
  --log-std-init 0.0 --log-std-max 0.0 \
  --curriculum --curriculum-mode uniform \
  --disable-dwa-recovery \
  --mask-cost-coef-final 0.8 \
  --total-episodes 50000 \
  > runs/ablation_suite/uniform_curriculum/nohup.out 2>&1 &

# A3 — fixed_stage4_only
mkdir -p runs/ablation_suite/fixed_stage4_only && nohup conda run -n HOPE python scripts/train_local_parking.py \
  --log-std-init 0.0 --log-std-max 0.0 \
  --stage 4 --curriculum-mode fixed \
  --disable-dwa-recovery \
  --mask-cost-coef-final 0.8 \
  --total-episodes 50000 \
  > runs/ablation_suite/fixed_stage4_only/nohup.out 2>&1 &

# A4 — no_mask_cost
mkdir -p runs/ablation_suite/no_mask_cost && nohup conda run -n HOPE python scripts/train_local_parking.py \
  --log-std-init 0.0 --log-std-max 0.0 \
  --curriculum --curriculum-mode adaptive \
  --disable-dwa-recovery \
  --mask-cost-coef-final 0.0 \
  --total-episodes 50000 \
  > runs/ablation_suite/no_mask_cost/nohup.out 2>&1 &

# A5 — no_mask_observation
mkdir -p runs/ablation_suite/no_mask_observation && nohup conda run -n HOPE python scripts/train_local_parking.py \
  --log-std-init 0.0 --log-std-max 0.0 \
  --curriculum --curriculum-mode adaptive \
  --disable-dwa-recovery \
  --mask-cost-coef-final 0.8 \
  --disable-mask-observation \
  --total-episodes 50000 \
  > runs/ablation_suite/no_mask_observation/nohup.out 2>&1 &

# A6 — front_lidar_only
mkdir -p runs/ablation_suite/front_lidar_only && nohup conda run -n HOPE python scripts/train_local_parking.py \
  --log-std-init 0.0 --log-std-max 0.0 \
  --curriculum --curriculum-mode adaptive \
  --disable-dwa-recovery \
  --mask-cost-coef-final 0.8 \
  --rear-lidar-observation-mode zero \
  --total-episodes 50000 \
  > runs/ablation_suite/front_lidar_only/nohup.out 2>&1 &
```

For a quick parameter and directory sanity check:

```bash
TOTAL_EPISODES=5 \
EVAL_INTERVAL=5 \
EVAL_EPISODES_PER_FAMILY=20 \
MAX_STEPS=60 \
SCENE_POOL_SIZE=1 \
SEEDS="0" \
DEVICE=cuda \
CONDA_ENV=HOPE \
USE_CONDA=1 \
bash scripts/sanity_check_ablation_flags.sh
```
