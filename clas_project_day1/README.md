# Adaptive Activation Steering for Cross-Lingual Response Control in LLM

Team Neural-Ninja, SemEval 2027 Task 7 (CLaS), CS 5134/6034 Fall 2026.

## Layout

| Path | What it holds | In git? |
|---|---|---|
| `scripts/` | our Python and shell code | yes |
| `slurm/` | ARCC2 job scripts | yes |
| `splits/split.json` | the fixed 20 dev / 50 test question split | yes |
| `results/` | score tables copied from the cluster | yes |
| `official/` | clone of github.com/d-gurgurov/CLaS-SemEval-2027 | no |
| `/N/lustre/scratch/mahmudmi/clas-data/` | data, model cache, generations, vectors | no |

## Data split

The 70 CLaS-Bench questions are split once, by question index, into 20 development
and 50 course-test questions (`scripts/make_splits.py`, seed 2027). The same indices
are used in every language, so no question appears in both sets in any language.
Development questions are used for all choices (layers, strengths, methods). Test
questions are used only to report results.
