#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOTAL_EPISODES="${TOTAL_EPISODES:-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
export EVAL_EPISODES_PER_FAMILY="${EVAL_EPISODES_PER_FAMILY:-20}"
export MAX_STEPS="${MAX_STEPS:-60}"
export SEEDS="${SEEDS:-0}"
export SCENE_POOL_SIZE="${SCENE_POOL_SIZE:-1}"

bash "${SCRIPT_DIR}/launch_ablation_suite.sh"
