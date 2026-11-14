# Paired bootstrap over language pairs (Llama-3.1-8B-Instruct, 10000 resamples, 95% CI)

| track | method | baseline | LSS | baseline LSS | delta LSS | 95% CI | delta LFS | delta OR | p | significant |
|---|---|---|---|---|---|---|---|---|---|---|
| Track 1 | DiffMean (L20, c=1) | Prompting | 0.931 | 0.692 | +0.239 | [+0.186, +0.296] | +0.426 | -0.084 | 0.0000 | yes |
| Track 1 | DiffMean (L16, c=0.75) | DiffMean (L20, c=1) | 0.964 | 0.931 | +0.033 | [+0.020, +0.048] | +0.004 | +0.060 | 0.0000 | yes |
| Track 1 | Tuned | DiffMean (L20, c=1) | 0.952 | 0.931 | +0.021 | [-0.000, +0.040] | -0.020 | +0.060 | 0.0518 | no |
| Track 1 | Tuned | DiffMean (L16, c=0.75) | 0.952 | 0.964 | -0.012 | [-0.029, -0.000] | -0.024 | -0.000 | 0.0446 | yes |
| Track 1 | Adaptive | DiffMean (L16, c=0.75) | 0.968 | 0.964 | +0.004 | [-0.000, +0.008] | -0.002 | +0.010 | 0.0624 | no |
| Track 1 | Adaptive | Tuned | 0.968 | 0.952 | +0.016 | [+0.004, +0.033] | +0.022 | +0.010 | 0.0020 | yes |
| Track 2 | DiffMean (L20, c=1) | Prompting | 0.778 | 0.841 | -0.063 | [-0.121, -0.006] | +0.101 | -0.227 | 0.0264 | yes |
| Track 2 | DiffMean (L16, c=0.75) | Prompting | 0.842 | 0.841 | +0.000 | [-0.043, +0.044] | +0.052 | -0.066 | 0.9848 | no |
| Track 2 | Tuned | DiffMean (L20, c=1) | 0.864 | 0.778 | +0.086 | [+0.063, +0.112] | -0.003 | +0.159 | 0.0000 | yes |
| Track 2 | Tuned | DiffMean (L16, c=0.75) | 0.864 | 0.842 | +0.022 | [+0.011, +0.035] | +0.045 | -0.001 | 0.0000 | yes |
| Track 2 | Adaptive | DiffMean (L16, c=0.75) | 0.851 | 0.842 | +0.009 | [+0.000, +0.020] | -0.012 | +0.035 | 0.0374 | yes |
| Track 2 | Tuned + Adaptive | Tuned | 0.876 | 0.864 | +0.012 | [+0.005, +0.019] | -0.003 | +0.026 | 0.0004 | yes |
| Track 2 | Tuned + Adaptive | Prompting | 0.876 | 0.841 | +0.034 | [-0.005, +0.074] | +0.095 | -0.041 | 0.0824 | no |
| Track 3 | DiffMean (L20, c=1) | Prompting | 0.964 | 0.707 | +0.258 | [+0.170, +0.357] | +0.420 | -0.026 | 0.0000 | yes |
| Track 3 | Tuned | DiffMean (L20, c=1) | 0.975 | 0.964 | +0.011 | [-0.004, +0.025] | +0.012 | +0.009 | 0.1550 | no |
| Track 3 | Adaptive | DiffMean (L16, c=0.75) | 0.975 | 0.974 | +0.001 | [-0.002, +0.004] | +0.001 | +0.002 | 0.4038 | no |
