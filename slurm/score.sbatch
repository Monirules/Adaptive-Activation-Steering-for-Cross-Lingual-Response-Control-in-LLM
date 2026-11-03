#!/usr/bin/env bash
#SBATCH --job-name=clas-score
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# Scores every unscored generation folder under ROOT (fastText + Qwen3-8B judge, loaded once).
# Usage (from ~/clas_project):  sbatch slurm/score.sbatch ROOT
set -euo pipefail
ROOT="${1:?root folder}"
source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "=== $(date) host=$(hostname) root=$ROOT"
cd "$CLAS_ROOT"
"$CLAS_PY" scripts/score_many.py --root "$ROOT"
echo "=== done $(date)"
