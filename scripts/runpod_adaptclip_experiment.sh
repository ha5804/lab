#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a RunPod RTX 4090 pod.
# This script only prepares the environment, then passes every argument to main.py.
# Examples:
#   bash scripts/runpod_adaptclip_experiment.sh --dataset visa --model winclip --topk-heatmaps 8 --output-dir results_visa_winclip_split
#   bash scripts/runpod_adaptclip_experiment.sh --dataset visa --model adaptclip --adaptclip-checkpoint visa --device cuda --topk-heatmaps 8 --output-dir results_visa_adaptclip_split
#   bash scripts/runpod_adaptclip_experiment.sh --dataset mvtec --model patchcore --device cuda --k 10000 --topk-heatmaps 8 --output-dir results_mvtec_patchcore

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-$PWD/.cache/open_clip}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# RTX 4090 works well with CUDA 12.1 wheels on RunPod.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
python -m pip install open_clip_torch huggingface_hub Pillow numpy pandas matplotlib scikit-learn jupyter

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python main.py "$@"
