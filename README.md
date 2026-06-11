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

The environment observation has 148 values:

- directed slot features: 13
- vehicle features: 5
- front/rear LiDAR: 108
- continuous action-mask features: 22

The policy emits the raw normalized continuous action
`[v_cmd_norm, phi_dot_cmd_norm]`. PPO log probabilities are computed for this
raw action. The environment then filters and clips it with the offline
articulated sweep table before advancing the vehicle model.
