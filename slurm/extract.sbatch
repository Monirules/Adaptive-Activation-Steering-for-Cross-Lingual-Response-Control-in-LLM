#!/usr/bin/env bash
#SBATCH --job-name=clas-extract
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
MODEL="${1:?model}"
source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "=== $(date) host=$(hostname) model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$CLAS_ROOT"
"$CLAS_PY" scripts/extract_means.py --model "$MODEL"
echo "=== done $(date)"
