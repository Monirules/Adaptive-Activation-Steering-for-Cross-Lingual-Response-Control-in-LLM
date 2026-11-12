#!/usr/bin/env bash
#SBATCH --job-name=clas-tuned
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
MODEL="${1:?model}"
TRACK="${2:?track}"
T3_TARGET="${3:-de}"
SHORT="${MODEL##*/}"
CONFIG="results/tuned_config_${SHORT}.json"

source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CLAS_ROOT"

if [ "$TRACK" = "t3" ]; then
  TRACK_DIR="t3/$T3_TARGET"
  LABEL="t3-$T3_TARGET"
else
  TRACK_DIR="$TRACK"
  LABEL="$TRACK"
fi
GEN="$CLAS_GEN/test/$SHORT/$TRACK_DIR/tuned"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "model=$MODEL track=$TRACK_DIR config=$CONFIG"
echo "output=$GEN"

"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split test --track "$TRACK" \
    --t3-target "$T3_TARGET" --pairs all --config-file "$CONFIG" --method-name tuned \
    --max-new-tokens 1024
"$CLAS_PY" scripts/score_many.py --root "$GEN"
"$CLAS_PY" scripts/inspect_outputs.py --gen-dir "$GEN" > "results/inspect_${SHORT}_${LABEL}_tuned.txt"

echo "=== done $(date)"
