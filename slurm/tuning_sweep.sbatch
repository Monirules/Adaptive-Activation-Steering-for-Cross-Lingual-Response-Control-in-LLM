#!/usr/bin/env bash
#SBATCH --job-name=clas-tuning
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
MODEL="${1:?model}"
T1_LAYERS="${2:?t1 layers}"
T1_STRENGTHS="${3:?t1 strengths}"
T2_LAYERS="${4:?t2 layers}"
T2_STRENGTHS="${5:?t2 strengths}"
SHORT="${MODEL##*/}"

source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CLAS_ROOT"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} model=$MODEL"

"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split dev --track t1 --pairs sweep \
    --vectors source --layers $T1_LAYERS --strengths $T1_STRENGTHS --out-root "$CLAS_GEN/sweep"
"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split dev --track t2 --pairs sweep \
    --vectors source --layers $T2_LAYERS --strengths $T2_STRENGTHS --out-root "$CLAS_GEN/sweep"
"$CLAS_PY" scripts/score_many.py --root "$CLAS_GEN/sweep/$SHORT"
"$CLAS_PY" scripts/pick_per_target.py --model "$SHORT"

echo "=== done $(date)"
