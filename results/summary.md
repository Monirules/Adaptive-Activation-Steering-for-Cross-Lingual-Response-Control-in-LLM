# Results

## dev

| model | track | method | LFS | OR | LSS | pairs | answers | judge unparsed |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | t1 | english | 0.493 | 0.981 | **0.656** | 90 | 1800 | 0 |
| Llama-3.1-8B-Instruct | t1 | none | 0.000 | 0.993 | **0.000** | 90 | 1800 | 0 |
| tiny-aya-global | t1 | english | 0.959 | 0.988 | **0.973** | 90 | 1800 | 0 |
| tiny-aya-global | t1 | none | 0.000 | 0.995 | **0.000** | 90 | 1800 | 0 |

## sweep

| model | track | method | LFS | OR | LSS | pairs | answers | judge unparsed |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L12-c0p5 | 0.008 | 0.978 | **0.017** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L12-c1 | 0.400 | 0.961 | **0.565** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L12-c2 | 0.835 | 0.819 | **0.827** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L12-c4 | 0.620 | 0.374 | **0.467** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L16-c0p5 | 0.055 | 0.980 | **0.104** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L16-c1 | 0.643 | 0.953 | **0.768** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L16-c2 | 0.815 | 0.673 | **0.737** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L16-c4 | 0.488 | 0.045 | **0.083** | 30 | 600 | 4 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L20-c0p5 | 0.047 | 0.973 | **0.089** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L20-c1 | 0.748 | 0.928 | **0.829** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L20-c2 | 0.830 | 0.592 | **0.691** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L20-c4 | 0.287 | 0.010 | **0.019** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L24-c0p5 | 0.032 | 0.973 | **0.061** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L24-c1 | 0.598 | 0.937 | **0.730** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L24-c2 | 0.725 | 0.661 | **0.692** | 30 | 600 | 2 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L24-c4 | 0.353 | 0.057 | **0.098** | 30 | 600 | 4 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L28-c0p5 | 0.012 | 0.972 | **0.023** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L28-c1 | 0.412 | 0.880 | **0.561** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L28-c2 | 0.673 | 0.453 | **0.542** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-others-L28-c4 | 0.380 | 0.020 | **0.038** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L12-c0p5 | 0.103 | 0.968 | **0.187** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L12-c1 | 0.853 | 0.950 | **0.899** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L12-c2 | 0.840 | 0.729 | **0.780** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L12-c4 | 0.357 | 0.198 | **0.255** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L16-c0p5 | 0.495 | 0.959 | **0.653** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L16-c1 | 0.920 | 0.930 | **0.925** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L16-c2 | 0.782 | 0.487 | **0.601** | 30 | 600 | 2 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L16-c4 | 0.335 | 0.031 | **0.057** | 30 | 600 | 7 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L20-c0p5 | 0.677 | 0.948 | **0.790** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L20-c1 | 0.963 | 0.894 | **0.927** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L20-c2 | 0.733 | 0.443 | **0.553** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L20-c4 | 0.312 | 0.018 | **0.033** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L24-c0p5 | 0.623 | 0.927 | **0.745** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L24-c1 | 0.912 | 0.873 | **0.892** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L24-c2 | 0.623 | 0.485 | **0.545** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L24-c4 | 0.217 | 0.021 | **0.038** | 30 | 600 | 1 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L28-c0p5 | 0.303 | 0.892 | **0.453** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L28-c1 | 0.883 | 0.810 | **0.845** | 30 | 600 | 0 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L28-c2 | 0.452 | 0.224 | **0.299** | 30 | 600 | 2 |
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L28-c4 | 0.148 | 0.002 | **0.003** | 30 | 600 | 1 |
| tiny-aya-global | t1 | diffmean-others-L14-c0p5 | 0.058 | 0.994 | **0.110** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L14-c1 | 0.715 | 0.968 | **0.822** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L14-c2 | 0.850 | 0.948 | **0.896** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L14-c4 | 0.522 | 0.578 | **0.548** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L18-c0p5 | 0.033 | 0.992 | **0.065** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L18-c1 | 0.703 | 0.978 | **0.818** | 30 | 600 | 1 |
| tiny-aya-global | t1 | diffmean-others-L18-c2 | 0.712 | 0.876 | **0.785** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L18-c4 | 0.558 | 0.471 | **0.511** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L22-c0p5 | 0.047 | 0.991 | **0.089** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L22-c1 | 0.762 | 0.963 | **0.851** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L22-c2 | 0.630 | 0.732 | **0.677** | 30 | 600 | 1 |
| tiny-aya-global | t1 | diffmean-others-L22-c4 | 0.230 | 0.234 | **0.232** | 30 | 600 | 2 |
| tiny-aya-global | t1 | diffmean-others-L27-c0p5 | 0.022 | 0.989 | **0.042** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L27-c1 | 0.520 | 0.948 | **0.672** | 30 | 600 | 1 |
| tiny-aya-global | t1 | diffmean-others-L27-c2 | 0.595 | 0.826 | **0.692** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L27-c4 | 0.323 | 0.235 | **0.272** | 30 | 600 | 3 |
| tiny-aya-global | t1 | diffmean-others-L32-c0p5 | 0.007 | 0.989 | **0.013** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-others-L32-c1 | 0.245 | 0.927 | **0.388** | 30 | 600 | 1 |
| tiny-aya-global | t1 | diffmean-others-L32-c2 | 0.575 | 0.429 | **0.491** | 30 | 600 | 3 |
| tiny-aya-global | t1 | diffmean-others-L32-c4 | 0.342 | 0.004 | **0.008** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L14-c0p5 | 0.420 | 0.981 | **0.588** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L14-c1 | 0.958 | 0.976 | **0.967** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L14-c2 | 0.835 | 0.925 | **0.878** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L14-c4 | 0.517 | 0.390 | **0.445** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L18-c0p5 | 0.563 | 0.973 | **0.714** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L18-c1 | 0.918 | 0.978 | **0.947** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L18-c2 | 0.540 | 0.788 | **0.641** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L18-c4 | 0.198 | 0.271 | **0.229** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L22-c0p5 | 0.825 | 0.972 | **0.892** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L22-c1 | 0.840 | 0.941 | **0.888** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L22-c2 | 0.372 | 0.552 | **0.444** | 30 | 600 | 1 |
| tiny-aya-global | t1 | diffmean-source-L22-c4 | 0.127 | 0.040 | **0.061** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L27-c0p5 | 0.413 | 0.964 | **0.579** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L27-c1 | 0.870 | 0.964 | **0.915** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L27-c2 | 0.538 | 0.685 | **0.603** | 30 | 600 | 1 |
| tiny-aya-global | t1 | diffmean-source-L27-c4 | 0.223 | 0.087 | **0.126** | 30 | 600 | 3 |
| tiny-aya-global | t1 | diffmean-source-L32-c0p5 | 0.110 | 0.953 | **0.197** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L32-c1 | 0.723 | 0.870 | **0.790** | 30 | 600 | 0 |
| tiny-aya-global | t1 | diffmean-source-L32-c2 | 0.333 | 0.197 | **0.248** | 30 | 600 | 5 |
| tiny-aya-global | t1 | diffmean-source-L32-c4 | 0.107 | 0.001 | **0.002** | 30 | 600 | 1 |

## test

| model | track | method | LFS | OR | LSS | pairs | answers | judge unparsed |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | t1 | diffmean-source-L20-c1 | 0.958 | 0.905 | **0.931** | 90 | 4500 | 0 |
| Llama-3.1-8B-Instruct | t1 | english | 0.532 | 0.989 | **0.692** | 90 | 4500 | 0 |
| Llama-3.1-8B-Instruct | t1 | none | 0.000 | 0.994 | **0.000** | 90 | 4500 | 0 |
| Llama-3.1-8B-Instruct | t2 | diffmean-source-L20-c1 | 0.868 | 0.705 | **0.778** | 100 | 5000 | 7 |
| Llama-3.1-8B-Instruct | t2 | english | 0.767 | 0.932 | **0.841** | 100 | 5000 | 11 |
| Llama-3.1-8B-Instruct | t3/de | diffmean-source-L20-c1 | 0.972 | 0.957 | **0.964** | 19 | 950 | 0 |
| Llama-3.1-8B-Instruct | t3/de | english | 0.552 | 0.983 | **0.707** | 19 | 950 | 0 |
| tiny-aya-global | t1 | diffmean-source-L14-c1 | 0.962 | 0.989 | **0.975** | 90 | 4500 | 0 |
| tiny-aya-global | t1 | english | 0.963 | 0.996 | **0.979** | 90 | 4500 | 0 |
| tiny-aya-global | t1 | none | 0.000 | 0.998 | **0.000** | 90 | 4500 | 2 |
| tiny-aya-global | t2 | english | 0.959 | 0.991 | **0.975** | 100 | 5000 | 15 |
| tiny-aya-global | t3/de | english | 0.993 | 0.996 | **0.995** | 19 | 950 | 1 |
