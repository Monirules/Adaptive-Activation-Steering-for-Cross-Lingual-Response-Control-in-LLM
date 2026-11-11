#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.clas_env"
for MODEL in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct; do
  echo "== $MODEL"
  hf download "$MODEL"
done
echo "ALL DOWNLOADS DONE"
