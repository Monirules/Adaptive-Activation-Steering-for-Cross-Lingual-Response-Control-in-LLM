#!/usr/bin/env bash
#SBATCH --job-name=clas-steer
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# Runs DiffMean steering generation. All arguments go to scripts/run_steering.py.
# Usage (from ~/clas_project):  sbatch slurm/steer.sbatch --model M --split dev ...
set -euo pipefail
source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== $(date) host=$(hostname)"
echo "args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$CLAS_ROOT"
"$CLAS_PY" scripts/run_steering.py "$@"
echo "=== done $(date)"
