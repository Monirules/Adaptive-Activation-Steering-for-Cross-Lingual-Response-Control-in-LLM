#!/usr/bin/env bash
#SBATCH --job-name=clas-adaptive
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
VARIANT="${4:-adaptive}"
SHORT="${MODEL##*/}"

source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CLAS_ROOT"

read -r MODE W TAU LATE EMA VEC LAYER C TAG < <("$CLAS_PY" -c "import json,sys; a=json.load(open('results/adaptive_config_%s.json' % sys.argv[1])); f=lambda x: ('%g' % x).replace('.', 'p'); print(a['mode'], a['warmup'], a['tau'], a['late'], a['ema'], a['vector'], a['layer'], a['strength'], 'adaptive-%s-w%d-t%s-l%s-%s-L%d-c%s' % (a['mode'], a['warmup'], f(a['tau']), f(a['late']), a['vector'], a['layer'], f(a['strength'])))" "$SHORT")

if [ "$TRACK" = "t3" ]; then
  TRACK_DIR="t3/$T3_TARGET"
  LABEL="t3-$T3_TARGET"
else
  TRACK_DIR="$TRACK"
  LABEL="$TRACK"
fi

ADAPT=(--adaptive "$MODE" --warmup "$W" --tau "$TAU" --late "$LATE" --ema "$EMA")
if [ "$VARIANT" = "tuned-adaptive" ]; then
  GEN="$CLAS_GEN/test/$SHORT/$TRACK_DIR/tuned-adaptive"
  STEER=(--config-file "results/tuned_config_${SHORT}.json" --method-name tuned-adaptive)
else
  GEN="$CLAS_GEN/test/$SHORT/$TRACK_DIR/$TAG"
  STEER=(--vectors "$VEC" --layers "$LAYER" --strengths "$C")
fi

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "model=$MODEL track=$TRACK_DIR variant=$VARIANT mode=$MODE warmup=$W tau=$TAU late=$LATE"
echo "output=$GEN"

"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split test --track "$TRACK" \
    --t3-target "$T3_TARGET" --pairs all "${STEER[@]}" "${ADAPT[@]}" --max-new-tokens 1024
"$CLAS_PY" scripts/score_many.py --root "$GEN"
"$CLAS_PY" scripts/inspect_outputs.py --gen-dir "$GEN" > "results/inspect_${SHORT}_${LABEL}_${VARIANT}.txt"

echo "=== done $(date)"
