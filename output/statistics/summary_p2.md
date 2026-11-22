# P2: sensitivity, neighbour selection, efficiency, learned kernels

## Sensitivity of DLT-dyn (test MSE, one factor at a time; default L = 48, R = 4, k = 8 from the P0 grid)

| L | ETTh1-96 | Weather-96 | PEMS08-96 |
|---|---|---|---|
| 12 | 0.4067 ± 0.0080 | 0.1646 ± 0.0004 | 0.5375 ± 0.0214 |
| 24 | 0.4095 ± 0.0030 | 0.1637 ± 0.0005 | 0.5282 ± 0.0119 |
| 48 | 0.4102 ± 0.0055 (default) | 0.1637 ± 0.0006 (default) | 0.5229 ± 0.0087 (default) |
| 96 | 0.4072 ± 0.0042 | 0.1628 ± 0.0003 | 0.5248 ± 0.0063 |

| R | ETTh1-96 | Weather-96 | PEMS08-96 |
|---|---|---|---|
| 2 | 0.4048 ± 0.0046 | 0.1659 ± 0.0004 | 0.5814 ± 0.0259 |
| 4 | 0.4102 ± 0.0055 (default) | 0.1637 ± 0.0006 (default) | 0.5229 ± 0.0087 (default) |
| 8 | 0.4085 ± 0.0025 | 0.1616 ± 0.0005 | 0.4974 ± 0.0161 |

| k | ETTh1-96 | Weather-96 | PEMS08-96 |
|---|---|---|---|
| 4 | 0.4008 ± 0.0037 | 0.1685 ± 0.0018 | 0.5834 ± 0.0052 |
| 8 | 0.4102 ± 0.0055 (default) | 0.1637 ± 0.0006 (default) | 0.5229 ± 0.0087 (default) |
| 16 | 0.4102 ± 0.0055 | 0.1598 ± 0.0034 | 0.4998 ± 0.0199 |

ETTh1 has 7 variables, so k is capped at 6 (k = 8 and k = 16 coincide).

## Neighbour selection (test MSE)

| variant | neighbours | Weather-96 | PEMS08-96 |
|---|---|---|---|
| DLT | CCF top-8 (default) | 0.1672 ± 0.0005 | 0.8583 ± 0.0047 |
| DLT | 8 random | 0.1711 ± 0.0013 | 0.7634 ± 0.0235 |
| DLT | all other variables | 0.1641 ± 0.0005 | 0.5108 ± 0.0073 |
| DLT-dyn | CCF top-8 (default) | 0.1637 ± 0.0006 | 0.5229 ± 0.0087 |
| DLT-dyn | 8 random | 0.1651 ± 0.0009 | 0.5093 ± 0.0053 |
| DLT-dyn | all other variables | 0.1616 ± 0.0028 | 0.4211 ± 0.0127 |

CI reference (no neighbours): Weather-96 0.1942 ± 0.0006, PEMS08-96 1.1681 ± 0.0024

## Efficiency on cpu, 4 threads: parameters / training ms per batch / inference ms per batch / peak memory MB

Batch 32, input 96, horizon 96, random data; baselines with their official hyper-parameters for each dataset. First block: DLT-dyn and the eight representative baselines; second block: kernel variants of the controlled study, the plug-in and further configurations.

| model | Weather | PEMS08 | ECL | Traffic |
|---|---|---|---|---|
| DLT-dyn | 22K / 14 / 2 / 123 | 27K / 84 / 26 / 308 | 32K / 171 / 64 / 446 | 49K / 493 / 159 / 994 |
| Informer | 11378K / 1142 / 205 / 1034 | 11912K / 1111 / 297 / 1035 | 12454K / 1014 / 222 / 1042 | 14393K / 1080 / 250 / 1121 |
| TimesNet | 1194K / 1069 / 250 / 213 | 150150K / 75610 / 17816 / 3295 | 150305K / 111583 / 12574 / 3300 | 301693K / 122965 / 26956 / 5868 |
| DLinear | 19K / 2 / 1 / 88 | 19K / 10 / 5 / 109 | 19K / 13 / 3 / 137 | 19K / 40 / 23 / 202 |
| PatchTST | 6904K / 1454 / 250 / 1298 | 415K / 1731 / 314 / 2163 | – | – |
| iTransformer | 4834K / 132 / 23 / 269 | 3256K / 583 / 109 / 897 | 4834K / 2253 / 378 / 2567 | 6412K / 18361 / 3952 / 12401 |
| TimeMixer | 104K / 200 / 45 / 434 | 894K / 156 / 43 / 341 | 107K / 3275 / 770 / 4920 | – |
| TimeXer | 998K / 120 / 16 / 273 | 11009K / 9427 / 2127 / 5713 | 11086K / 19993 / 4787 / 10943 | – |

| model | Weather | PEMS08 | ECL | Traffic |
|---|---|---|---|---|
| CI (RLinear-style) | 9K / 1 / 0 / 86 | 9K / 10 / 3 / 109 | 9K / 12 / 4 / 137 | 9K / 36 / 13 / 190 |
| scalar | 19K / 3 / 2 / 95 | 20K / 22 / 12 / 146 | 21K / 33 / 19 / 199 | 26K / 121 / 83 / 381 |
| full kernel | 27K / 10 / 4 / 109 | 85K / 52 / 23 / 196 | 144K / 108 / 44 / 280 | 357K / 330 / 133 / 521 |
| top-5 lags | 19K / 7 / 3 / 106 | 25K / 72 / 34 / 236 | 31K / 99 / 71 / 361 | 53K / 368 / 263 / 834 |
| MillGNN-dyn kernel | 21K / 11 / 9 / 108 | 25K / 54 / 38 / 251 | 28K / 107 / 80 / 396 | 41K / 425 / 268 / 887 |
| DLT | 19K / 11 / 4 / 108 | 24K / 56 / 26 / 206 | 29K / 107 / 46 / 254 | 46K / 329 / 143 / 514 |
| DLT-dyn (kernel path) | 22K / 9 / 2 / 118 | 27K / 86 / 30 / 236 | 32K / 146 / 44 / 350 | 49K / 406 / 149 / 717 |
| iTransformer + DLT-dyn | 4847K / 155 / 26 / 298 | 3274K / 693 / 120 / 988 | 4856K / 2463 / 363 / 2635 | 6452K / 19787 / 4033 / 12885 |
| PatchTST (paper config) | 547K / 266 / 54 / 472 | 547K / 2450 / 446 / 2896 | 547K / 6291 / 1822 / 4294 | – |
| LagTS | 5642K / 310 / 89 / 489 | – | – | – |


MillGNN (reported, A100, Traffic): 9.7 GB GPU memory; iTransformer 7.5 GB in the same figure.

## Efficiency in the S-Mamba setting on cpu, 4 threads: parameters / training ms per batch / inference ms per batch / peak memory MB

Batch 16, input 96; ETTm1 horizon 96, ECL horizon 96, Traffic horizon 96, PEMS07 horizon 12 (the efficiency experiment of the S-Mamba paper, Fig. 5); random data; baselines with their official hyper-parameters for each dataset. DLT-dyn (tuned): the recipe selected on the validation loss in the eff_tune rounds (output/efficiency/eff_tune.md), on the basis path as trained and on the equivalent kernel path.

| model | ETTm1 | ECL | Traffic | PEMS07 |
|---|---|---|---|---|
| DLT-dyn | 22K / 4 / 8 / 102 | 32K / 82 / 21 / 308 | 49K / 266 / 80 / 565 | 34K / 237 / 74 / 566 |
| DLT-dyn (tuned) | 22K / 4 / 1 / 102 | 63K / 305 / 71 / 696 | 77K / 444 / 132 / 943 | 62K / 460 / 132 / 926 |
| DLT-dyn (tuned, kernel path) | 22K / 4 / 1 / 102 | 63K / 136 / 54 / 334 | 77K / 387 / 149 / 694 | 62K / 380 / 154 / 705 |
| iTransformer | 224K / 16 / 3 / 103 | 4834K / 1226 / 162 / 1320 | 6412K / 8630 / 1187 / 7162 | 3213K / 4499 / 716 / 4025 |
| PatchTST | 3752K / 137 / 20 / 296 | 6904K / 13924 / 3197 / 7065 | 3755K / 20900 / 5008 / 10643 | – |
| RLinear | 9K / 1 / 0 / 84 | 10K / 11 / 4 / 115 | 11K / 21 / 8 / 174 | 3K / 11 / 8 / 124 |
| DLinear | 19K / 1 / 0 / 88 | 19K / 5 / 1 / 108 | 19K / 13 / 4 / 150 | 2K / 4 / 3 / 112 |
| TiDE | 1185K / 52 / 10 / 166 | 1185K / 1763 / 442 / 2030 | 1185K / 4735 / 1196 / 5294 | 744K / 3259 / 866 / 3075 |

Not measured: PatchTST on PEMS07: killed (SIGKILL, out of memory).

## Learned kernels on PEMS08 vs the road network (mean over seeds)

| quantity | DLT | DLT-dyn |
|---|---|---|
| selected neighbours 1 hop away | 0.123 ± 0.000 | 0.123 ± 0.000 |
| all pairs 1 hop away | 0.019 ± 0.000 | 0.019 ± 0.000 |
| selected neighbours 2 hops away | 0.115 ± 0.000 | 0.115 ± 0.000 |
| all pairs 2 hops away | 0.037 ± 0.000 | 0.037 ± 0.000 |
| selected neighbours upstream of the target | 0.593 ± 0.000 | 0.593 ± 0.000 |
| kernel mass beyond lag 0 | 0.860 ± 0.016 | 0.874 ± 0.068 |
| kernels with >= 2 lags at >= 50 % of the peak | 0.571 ± 0.159 | 0.587 ± 0.320 |
| mean peak lag, upstream-only sources (steps) | 5.619 ± 1.115 | 5.167 ± 2.853 |
| mean peak lag, downstream-only sources (steps) | 5.164 ± 1.673 | 4.763 ± 3.809 |
| Spearman rho(road distance, peak lag) | -0.203 ± 0.046 | -0.045 ± 0.204 |
| Spearman rho(road distance, lag centroid) | -0.317 ± 0.020 | -0.202 ± 0.170 |

Checkpoints: 6 (DLT and DLT-dyn, PEMS08-96, P0 grid). Figure: `figures/fig_kernels_pems08.pdf` in the DLTpaper repository (written by scripts/mechanism/kernels.py).

