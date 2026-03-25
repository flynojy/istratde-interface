#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f ".venv-brax/bin/activate" ]]; then
  echo "Missing .venv-brax. Run: bash ./setup_brax_ubuntu.sh"
  exit 1
fi

source .venv-brax/bin/activate

export BRAX_ENVS="${BRAX_ENVS:-swimmer,hopper,reacher}"
export BRAX_TIME_BUDGET_MINUTES="${BRAX_TIME_BUDGET_MINUTES:-60}"
export BRAX_POP_SIZE="${BRAX_POP_SIZE:-10000}"
export BRAX_MAX_EPISODE_LENGTH="${BRAX_MAX_EPISODE_LENGTH:-500}"
export BRAX_NUM_EPISODES="${BRAX_NUM_EPISODES:-1}"
export BRAX_HIDDEN_DIMS="${BRAX_HIDDEN_DIMS:-32,32}"
export BRAX_LOG_INTERVAL_SECONDS="${BRAX_LOG_INTERVAL_SECONDS:-60}"
export BRAX_SEED="${BRAX_SEED:-42}"
export BRAX_SAVE_HTML="${BRAX_SAVE_HTML:-0}"

python brax_paper_benchmark.py
