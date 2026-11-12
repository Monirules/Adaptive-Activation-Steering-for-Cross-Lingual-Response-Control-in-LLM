#!/usr/bin/env bash
#SBATCH --job-name=clas-sweep-test
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
MODEL="${1:?model}"
LAYERS="${2:?layers}"
STRENGTHS="${3:?strengths}"
T3_TARGET="${4:-de}"
TRACKS="${5:-t1 t2 t3}"
SHORT="${MODEL##*/}"

source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CLAS_ROOT"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "model=$MODEL layers=$LAYERS strengths=$STRENGTHS"

"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split dev --pairs sweep \
    --vectors source --layers $LAYERS --strengths $STRENGTHS --out-root "$CLAS_GEN/sweep"
"$CLAS_PY" scripts/score_many.py --root "$CLAS_GEN/sweep/$SHORT"
"$CLAS_PY" scripts/pick_best.py --root "$CLAS_GEN/sweep"

for TRACK in $TRACKS; do
  bash slurm/track_test.sbatch "$MODEL" "$TRACK" "$T3_TARGET"
done

echo "=== done $(date)"
