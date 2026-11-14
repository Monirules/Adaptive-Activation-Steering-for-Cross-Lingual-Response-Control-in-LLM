#!/usr/bin/env bash
#SBATCH --job-name=clas-agreement
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$CLAS_ROOT"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
"$CLAS_PY" scripts/judge_agreement.py --target "${1:-hi}"
echo "=== done $(date)"
