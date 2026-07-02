#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TOTAL_EPISODES="${TOTAL_EPISODES:-20000}"
UNSAFE_TOTAL_EPISODES="${UNSAFE_TOTAL_EPISODES:-$((TOTAL_EPISODES / 4))}"
if [[ "${UNSAFE_TOTAL_EPISODES}" -lt 1 ]]; then
  UNSAFE_TOTAL_EPISODES=1
fi
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_EPISODES_PER_FAMILY="${EVAL_EPISODES_PER_FAMILY:-20}"
MAX_STEPS="${MAX_STEPS:-600}"
SEEDS="${SEEDS:-0 1 2}"
CONDA_ENV="${CONDA_ENV:-HOPE}"
USE_CONDA="${USE_CONDA:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
SCENE_POOL_SIZE="${SCENE_POOL_SIZE:-18}"
DRY_RUN="${DRY_RUN:-0}"
MIN_AVAILABLE_MEM_GB_PER_JOB="${MIN_AVAILABLE_MEM_GB_PER_JOB:-4}"

if [[ -z "${DEVICE:-}" ]]; then
  DEVICE_CHECK='import torch; print("cuda" if torch.cuda.is_available() else "cpu")'
  if [[ "${USE_CONDA}" == "1" ]] && command -v conda >/dev/null 2>&1; then
    DEVICE="$(conda run -n "${CONDA_ENV}" python -c "${DEVICE_CHECK}" 2>/dev/null || echo cpu)"
  else
    DEVICE="$(python -c "${DEVICE_CHECK}" 2>/dev/null || echo cpu)"
  fi
fi

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
if [[ -z "${MAX_PARALLEL:-}" ]]; then
  if [[ "${DEVICE}" == "cpu" ]]; then
    MAX_PARALLEL=$((CPU_COUNT / 8))
    if [[ "${MAX_PARALLEL}" -lt 2 ]]; then
      MAX_PARALLEL=2
    fi
    if [[ "${MAX_PARALLEL}" -gt 4 ]]; then
      MAX_PARALLEL=4
    fi
  else
    MAX_PARALLEL=1
  fi
fi
if [[ "${DEVICE}" != "cpu" && "${MAX_PARALLEL}" != "1" ]]; then
  echo "DEVICE=${DEVICE} requested; forcing MAX_PARALLEL=1 to avoid GPU contention."
  MAX_PARALLEL=1
fi
if [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "MAX_PARALLEL must be >= 1" >&2
  exit 1
fi
RUN_THREADS="${RUN_THREADS:-$((CPU_COUNT / MAX_PARALLEL))}"
if [[ "${RUN_THREADS}" -lt 1 ]]; then
  RUN_THREADS=1
fi
if [[ "${DEVICE}" == "cpu" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${RUN_THREADS}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${RUN_THREADS}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${RUN_THREADS}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${RUN_THREADS}}"
fi

SUITE_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${REPO_ROOT}/runs/ablation_suite"
SUITE_DIR="${SUITE_ROOT}/${SUITE_TIMESTAMP}"
mkdir -p "${SUITE_DIR}"
FAILED_RUNS="${SUITE_DIR}/failed_runs.txt"

EXPERIMENTS=(
  full
  no_rs_potential
  uniform_curriculum
  fixed_stage4_only
  no_mask_cost
  no_mask_observation
  front_lidar_only
  dwa_assisted
  mask_floor_fallback
  unsafe_no_action_mask_execution
)

write_suite_config() {
  {
    echo "suite_dir = ${SUITE_DIR}"
    echo "git_commit = $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status_short = $(git status --short 2>/dev/null | tr '\n' ';' || true)"
    echo "hostname = $(hostname 2>/dev/null || echo unknown)"
    echo "date = $(date -Is)"
    echo "total_episodes = ${TOTAL_EPISODES}"
    echo "unsafe_total_episodes = ${UNSAFE_TOTAL_EPISODES}"
    echo "eval_interval = ${EVAL_INTERVAL}"
    echo "eval_episodes_per_family = ${EVAL_EPISODES_PER_FAMILY}"
    echo "max_steps = ${MAX_STEPS}"
    echo "checkpoint_interval = ${CHECKPOINT_INTERVAL}"
    echo "scene_pool_size = ${SCENE_POOL_SIZE}"
    echo "seeds = ${SEEDS}"
    echo "device = ${DEVICE}"
    echo "conda_env = ${CONDA_ENV}"
    echo "use_conda = ${USE_CONDA}"
    echo "max_parallel = ${MAX_PARALLEL}"
    echo "run_threads = ${RUN_THREADS}"
    echo "omp_num_threads = ${OMP_NUM_THREADS:-}"
    echo "mkl_num_threads = ${MKL_NUM_THREADS:-}"
    echo "openblas_num_threads = ${OPENBLAS_NUM_THREADS:-}"
    echo "numexpr_num_threads = ${NUMEXPR_NUM_THREADS:-}"
    echo "cpu_count = ${CPU_COUNT}"
    echo "min_available_mem_gb_per_job = ${MIN_AVAILABLE_MEM_GB_PER_JOB}"
    echo "dry_run = ${DRY_RUN}"
    echo "experiments = ${EXPERIMENTS[*]}"
    echo
    echo "Notes:"
    echo "- full is the main method."
    echo "- no_rs_potential disables near-goal RS potential shaping, not a teacher."
    echo "- no_mask_observation and no_mask_cost keep safe action-mask execution enabled."
    echo "- front_lidar_only zeroes only rear LiDAR observations; collision and mask truth still use both bodies."
    echo "- dwa_assisted is an execution-time diagnostic, not the default learning method."
    echo "- unsafe_no_action_mask_execution is an unsafe diagnostic and is excluded from fair main-table comparisons."
  } > "${SUITE_DIR}/suite_config.txt"
}

available_mem_gb() {
  awk '/MemAvailable:/ { printf "%.1f\n", $2 / 1024 / 1024 }' /proc/meminfo 2>/dev/null || echo 0
}

wait_for_memory() {
  local required_gb="$1"
  local available_gb
  while true; do
    available_gb="$(available_mem_gb)"
    if awk -v available="${available_gb}" -v required="${required_gb}" 'BEGIN { exit !(available >= required) }'; then
      return 0
    fi
    echo "[$(date -Is)] Waiting for memory: available=${available_gb}GiB required=${required_gb}GiB"
    sleep 30
  done
}

active_jobs() {
  jobs -pr | wc -l
}

wait_for_slot() {
  while [[ "$(active_jobs)" -ge "${MAX_PARALLEL}" ]]; do
    wait -n || true
  done
}

runner_prefix() {
  if [[ "${USE_CONDA}" == "1" ]]; then
    printf '%s\n' conda run -n "${CONDA_ENV}"
  fi
}

print_command() {
  printf 'PYTHONPATH=src '
  printf '%q ' "$@"
  printf '\n'
}

experiment_args() {
  local experiment="$1"
  case "${experiment}" in
    full)
      echo "--enable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.8 --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    no_rs_potential)
      echo "--disable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.8 --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    uniform_curriculum)
      echo "--enable-rs-potential --curriculum --curriculum-mode uniform --mask-cost-coef-final 0.8 --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    fixed_stage4_only)
      echo "--enable-rs-potential --stage 4 --curriculum-mode fixed --mask-cost-coef-final 0.8 --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    no_mask_cost)
      echo "--enable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.0 --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    no_mask_observation)
      echo "--enable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.8 --disable-mask-observation --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    front_lidar_only)
      echo "--enable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.8 --rear-lidar-observation-mode zero --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    dwa_assisted)
      echo "--enable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.8 --enable-dwa-recovery --enable-dwa-override-policy-action --enable-dwa-deadlock-termination"
      ;;
    mask_floor_fallback)
      echo "--enable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.8 --enable-mask-floor-fallback --apply-floor-only-when-all-zero --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    unsafe_no_action_mask_execution)
      echo "--enable-rs-potential --curriculum --curriculum-mode adaptive --mask-cost-coef-final 0.8 --disable-action-mask-execution --disable-dwa-recovery --disable-dwa-override-policy-action --disable-dwa-deadlock-termination"
      ;;
    *)
      echo "unknown experiment: ${experiment}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local experiment="$1"
  local seed="$2"
  local episodes="${TOTAL_EPISODES}"
  if [[ "${experiment}" == "unsafe_no_action_mask_execution" ]]; then
    episodes="${UNSAFE_TOTAL_EPISODES}"
  fi

  local run_dir="${SUITE_DIR}/${experiment}/seed_${seed}"
  mkdir -p "${run_dir}"
  local log_path="${run_dir}/train.log"
  local command_path="${run_dir}/command.txt"
  local status_path="${run_dir}/status.txt"

  local -a runner=()
  while IFS= read -r item; do
    [[ -n "${item}" ]] && runner+=("${item}")
  done < <(runner_prefix)

  local -a extra=()
  read -r -a extra <<< "$(experiment_args "${experiment}")"

  local -a command=(
    "${runner[@]}"
    python src/train/train_local_parking.py
    --total-episodes "${episodes}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-episodes-per-family "${EVAL_EPISODES_PER_FAMILY}"
    --max-steps "${MAX_STEPS}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --scene-pool-size "${SCENE_POOL_SIZE}"
    --seed "${seed}"
    --device "${DEVICE}"
    --scene-family-schedule head_in
    --output-dir "${run_dir}"
    "${extra[@]}"
  )

  print_command "${command[@]}" > "${command_path}"
  echo "[$(date -Is)] START experiment=${experiment} seed=${seed} episodes=${episodes}"
  echo "command: $(cat "${command_path}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry_run" > "${status_path}"
    return 0
  fi

  set +e
  PYTHONPATH=src "${command[@]}" > "${log_path}" 2>&1
  local exit_code=$?
  set -e
  if [[ "${exit_code}" -eq 0 ]]; then
    echo "success" > "${status_path}"
    echo "[$(date -Is)] DONE experiment=${experiment} seed=${seed}"
  else
    echo "failed exit_code=${exit_code}" > "${status_path}"
    echo "${experiment},seed_${seed},exit_code=${exit_code},${run_dir}" >> "${FAILED_RUNS}"
    echo "[$(date -Is)] FAILED experiment=${experiment} seed=${seed} exit_code=${exit_code}"
  fi
}

write_suite_config
echo "Ablation suite directory: ${SUITE_DIR}"
echo "Device: ${DEVICE}; MAX_PARALLEL=${MAX_PARALLEL}; RUN_THREADS=${RUN_THREADS}; CPU_COUNT=${CPU_COUNT}; MemAvailable=$(available_mem_gb)GiB"

for experiment in "${EXPERIMENTS[@]}"; do
  for seed in ${SEEDS}; do
    if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
      wait_for_slot
      wait_for_memory "${MIN_AVAILABLE_MEM_GB_PER_JOB}"
      run_one "${experiment}" "${seed}" &
    else
      wait_for_memory "${MIN_AVAILABLE_MEM_GB_PER_JOB}"
      run_one "${experiment}" "${seed}"
    fi
  done
done

if [[ "${MAX_PARALLEL}" -gt 1 ]]; then
  wait
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete; commands are under ${SUITE_DIR}."
  exit 0
fi

echo "[$(date -Is)] Summarizing suite ${SUITE_DIR}"
if PYTHONPATH=src python scripts/summarize_ablation_suite.py --suite-dir "${SUITE_DIR}"; then
  echo "summary.csv: ${SUITE_DIR}/summary.csv"
  echo "summary.md: ${SUITE_DIR}/summary.md"
  echo "best_checkpoints.md: ${SUITE_DIR}/best_checkpoints.md"
  if [[ -f "${FAILED_RUNS}" ]]; then
    echo "failed_runs.txt: ${FAILED_RUNS}"
  fi
else
  echo "Summary failed for ${SUITE_DIR}" >&2
  exit 1
fi
