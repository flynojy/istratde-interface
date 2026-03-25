#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN"
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if not ((major, minor) >= (3, 10) and (major, minor) <= (3, 12)):
    raise SystemExit(
        f"Python {major}.{minor} is not recommended. Please use Python 3.10-3.12 for Brax/JAX."
    )
PY

"$PYTHON_BIN" -m venv .venv-brax
source .venv-brax/bin/activate

python -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA wheels on Linux
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

# Common project dependencies
python -m pip install -r requirements.txt

# JAX CUDA wheels on Linux, following the official JAX installation style.
python -m pip install --upgrade "jax[cuda12]==0.4.33" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Brax-side runtime dependencies
python -m pip install -r requirements-brax.txt

python - <<'PY'
import brax
import jax
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("jax:", jax.__version__)
print("jax backend:", jax.default_backend())
print("brax:", brax.__version__)
PY

echo
echo "Brax Ubuntu environment is ready in: $ROOT/.venv-brax"
echo "Run: bash ./run_brax_paper_ubuntu.sh"
