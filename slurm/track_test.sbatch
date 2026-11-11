#!/usr/bin/env bash
#SBATCH --job-name=clas-track
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

source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CLAS_ROOT"

read -r VEC LAYER C TAG < <("$CLAS_PY" -c "import json,sys; b=json.load(open('results/best_config.json'))[sys.argv[1]]; c=float(b['strength']); print(b['vector'], b['layer'], c, 'diffmean-%s-L%d-c%s' % (b['vector'], b['layer'], ('%g' % c).replace('.', 'p')))" "$SHORT")

if [ "$TRACK" = "t3" ]; then
  TRACK_DIR="t3/$T3_TARGET"
  LABEL="t3-$T3_TARGET"
else
  TRACK_DIR="$TRACK"
  LABEL="$TRACK"
fi
GEN="$CLAS_GEN/test/$SHORT/$TRACK_DIR/$TAG"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "model=$MODEL track=$TRACK_DIR vector=$VEC layer=$LAYER strength=$C"
echo "output=$GEN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split test --track "$TRACK" \
    --t3-target "$T3_TARGET" --pairs all --vectors "$VEC" --layers "$LAYER" \
    --strengths "$C" --max-new-tokens 1024
"$CLAS_PY" scripts/score_many.py --root "$GEN"
"$CLAS_PY" scripts/inspect_outputs.py --gen-dir "$GEN" > "results/inspect_${SHORT}_${LABEL}_diffmean.txt"

echo "=== done $(date)"
