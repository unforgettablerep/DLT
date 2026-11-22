# Baseline check: the six baselines against the S-Mamba paper (2026-11-17)

The study now uses six baselines, those of the S-Mamba paper (Wang et al., Neurocomputing 2025): S-Mamba,
iTransformer, PatchTST, RLinear, DLinear and TiDE. The reference is the paper's Tables 2-4 (input 96; 13 datasets,
four horizons each; `assets/reported.json`, block `smamba_paper`). Its baseline columns equal the iTransformer
paper, which reproduced the baselines in Time-Series-Library "built on the configurations provided by each model's
original paper or official code" (its Appendix A.2).

Each baseline was checked in three steps: the implementation against the official code, the settings against the
official scripts, and the forecasts against the published numbers.

## 1. Implementation

| baseline | official code | result |
|---|---|---|
| DLinear, PatchTST, TiDE | Time-Series-Library (latest, `4e938a1`) | `models/*.py` identical |
| iTransformer | github.com/thuml/iTransformer | same model; embedding, encoder layers and attention identical (AST); here a `--use_norm` switch, default on as there |
| S-Mamba | github.com/wzhwzhwzh0921/S-D-Mamba | same forward pass; the official encoder layer also builds two Mamba blocks and an attention layer that never run, left out here (they only change the random initial weights) |
| RLinear | github.com/plumprc/RTSF | ported (`src/model/RLinear.py`): RevIN with learnable affine parameters, dropout, one linear map shared by the variates; a unit test checks it against the written-out formula |

## 2. Settings and protocol

`src/configs/baseline_profiles.py` reads every setting from copies of the official scripts (`scripts/benchmark/official/`). Differences
to the earlier settings that this fixed:
- iTransformer ETTh1 at horizons 336 / 720: d_model / d_ff 512 (one setting of 256 was used for every horizon);
- PatchTST: the Time-Series-Library scripts (the earlier study used the PatchTST paper's settings);
- TiDE: `--c_out 8` is TiDE's decoder output dimension in the library; the earlier runs set it to the number of
  variables;
- protocol of the official loaders, for every baseline: four time features on every dataset (no official script
  sets `--freq`, so the default `h` holds; the earlier runs used five on 10/15-minute data and three on Exchange),
  training and validation loaders that drop their last incomplete batch, test batch 1, and on PEMS and Solar no
  time marks and the split of the iTransformer loader.

Official lines are missing for: PatchTST on Solar and PEMS (the Electricity line is taken), TiDE on every dataset
but ETTh1 (the ETTh1 line is taken), RLinear on Exchange, Traffic, Solar and PEMS (ETTh1's or Electricity's line),
DLinear on most datasets (library defaults; DLinear has no settings of its own besides training).

**Pre-flight.** `scripts/utils/smoke_jobs.py` built every configuration of the check on the CPU (batch 2, the shortest and the longest horizon of each) and trained it for one step: 134 checks passed, 0 failed. S-Mamba ran on ETTh1/h2/m1/m2, PEMS03 and PEMS04-12 only; its PyTorch block needs about 18 minutes for one step on PEMS04 on this CPU, and on the GPU it runs mamba_ssm.

## 3. Forecasts against the published numbers

### On the CPU here, with the protocol of the check (seed 2021)

#### RLinear

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| ETTh1 | 0.383 / 0.386 (-0.8 %) | 0.435 / 0.437 (-0.5 %) | 0.475 / 0.479 (-0.9 %) | 0.476 / 0.481 (-1.1 %) | -0.8 % |
| ETTh2 | 0.290 / 0.288 (+0.5 %) | 0.377 / 0.374 (+0.9 %) | 0.426 / 0.415 (+2.7 %) | 0.446 / 0.420 (+6.1 %)* | +1.8 % |
| ETTm1 | 0.350 / 0.355 (-1.5 %) | 0.388 / 0.391 (-0.8 %) | 0.421 / 0.424 (-0.8 %) | 0.479 / 0.487 (-1.6 %) | -1.9 % |
| ETTm2 | 0.182 / 0.182 (-0.2 %) | 0.247 / 0.246 (+0.3 %) | 0.308 / 0.307 (+0.3 %) | 0.408 / 0.407 (+0.3 %) | +0.3 % |
| Exchange | 0.083 / 0.093 (-10.4 %) | 0.180 / 0.184 (-2.2 %) | 0.343 / 0.351 (-2.3 %) | 0.901 / 0.886 (+1.7 %) | -3.0 % |
| Weather | 0.193 / 0.192 (+0.6 %) | 0.237 / 0.240 (-1.3 %) | 0.288 / 0.292 (-1.4 %) | 0.359 / 0.364 (-1.2 %) | -0.5 % |

24 settings; within 5 %: 23; mean |change| 1.7 %; more than 10 % worse: none.

#### PatchTST

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| ETTh1 | 0.403 / 0.414 (-2.7 %) | 0.425 / 0.460 (-7.7 %) | - / 0.501 | - / 0.500 | -2.5 % |

2 settings; within 5 %: 2; mean |change| 5.2 %; more than 10 % worse: none.

#### TiDE

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| ETTh2 | 0.290 / 0.400 (-27.6 %) | - / 0.528 | - / 0.643 | - / 0.874 | -23.1 % |
| Exchange | 0.084 / 0.094 (-10.3 %) | - / 0.184 | - / 0.349 | - / 0.852 | -6.9 % |

2 settings; within 5 %: 2; mean |change| 18.9 %; more than 10 % worse: none.

#### Summary

| baseline | settings run | within 5 % | mean abs. change | > 10 % worse |
|---|---:|---:|---:|---:|
| RLinear | 24 | 23 | 1.7 % | 0 |
| PatchTST | 2 | 2 | 5.2 % | 0 |
| TiDE | 2 | 2 | 18.9 % | 0 |

### Earlier GPU runs, previous protocol (RTX 4090, seed 2021; `output/plugin/plugin_all.tsv`, backbone alone)

These ran before the protocol was aligned (five time features on 10/15-minute data, no dropped batches; PEMS with
time marks and the TSLib split; PatchTST with its paper's settings). They show where the baselines stood:

#### S-Mamba

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| ETTh1 | 0.389 / 0.386 (+0.7 %) | 0.445 / 0.443 (+0.4 %) | 0.491 / 0.489 (+0.4 %) | 0.511 / 0.502 (+1.8 %) | +0.8 % |
| ETTh2 | 0.296 / 0.296 (+0.0 %) | 0.378 / 0.376 (+0.5 %) | 0.422 / 0.424 (-0.4 %) | 0.428 / 0.426 (+0.5 %) | +0.4 % |
| ETTm1 | 0.331 / 0.333 (-0.5 %) | 0.377 / 0.376 (+0.3 %) | 0.409 / 0.408 (+0.3 %) | 0.473 / 0.475 (-0.3 %) | +0.3 % |
| ETTm2 | 0.179 / 0.179 (+0.1 %) | 0.252 / 0.250 (+0.9 %) | 0.312 / 0.312 (-0.1 %) | 0.413 / 0.411 (+0.4 %) | +0.2 % |
| Exchange | 0.087 / 0.086 (+1.2 %) | 0.181 / 0.182 (-0.5 %) | 0.330 / 0.332 (-0.5 %) | 0.858 / 0.867 (-1.1 %) | -0.2 % |
| Weather | 0.167 / 0.165 (+1.4 %) | 0.216 / 0.214 (+1.0 %) | 0.277 / 0.274 (+1.0 %) | 0.352 / 0.350 (+0.5 %) | +0.7 % |
| PEMS04 | 0.073 / 0.076 (-4.2 %) | 0.084 / 0.084 (-0.1 %) | 0.104 / 0.115 (-10.0 %) | 0.125 / 0.137 (-9.1 %) | -3.0 % |
| PEMS08 | 0.077 / 0.076 (+1.0 %) | 0.106 / 0.104 (+2.3 %) | 0.161 / 0.167 (-3.5 %) | 0.277 / 0.245 (+12.9 %)** | +7.0 % |

32 settings; within 5 %: 31; mean |change| 1.8 %; more than 10 % worse: PEMS08-96 +12.9 %.

#### iTransformer

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| ETTh1 | 0.386 / 0.386 (-0.1 %) | 0.440 / 0.441 (-0.1 %) | 0.487 / 0.487 (+0.1 %) | 0.501 / 0.503 (-0.3 %) | +0.1 % |
| ETTh2 | 0.300 / 0.297 (+1.0 %) | 0.380 / 0.380 (-0.1 %) | 0.422 / 0.428 (-1.3 %) | 0.431 / 0.427 (+0.9 %) | +0.0 % |
| ETTm1 | 0.341 / 0.334 (+2.2 %) | 0.380 / 0.377 (+0.9 %) | 0.424 / 0.426 (-0.5 %) | 0.485 / 0.491 (-1.1 %) | +0.7 % |
| ETTm2 | 0.183 / 0.180 (+1.4 %) | 0.253 / 0.250 (+1.0 %) | 0.315 / 0.311 (+1.2 %) | 0.412 / 0.412 (+0.1 %) | +0.7 % |
| Exchange | 0.087 / 0.086 (+1.4 %) | 0.184 / 0.177 (+3.7 %) | 0.328 / 0.331 (-1.0 %) | 0.861 / 0.847 (+1.6 %) | +1.1 % |
| Weather | 0.175 / 0.174 (+0.6 %) | 0.225 / 0.221 (+1.9 %) | 0.282 / 0.278 (+1.3 %) | 0.360 / 0.358 (+0.4 %) | +0.9 % |
| PEMS04 | 0.081 / 0.078 (+3.5 %) | 0.100 / 0.095 (+4.8 %) | 0.131 / 0.120 (+9.3 %)* | 0.173 / 0.150 (+15.5 %)** | +5.1 % |
| PEMS08 | 0.087 / 0.079 (+9.7 %)* | 0.133 / 0.115 (+15.3 %)** | 0.263 / 0.186 (+41.4 %)** | 0.493 / 0.221 (+123.1 %)** | +26.9 % |

32 settings; within 5 %: 26; mean |change| 7.7 %; more than 10 % worse: PEMS04-96 +15.5 %, PEMS08-24 +15.3 %, PEMS08-48 +41.4 %, PEMS08-96 +123.1 %.

#### DLinear

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| ETTh1 | 0.400 / 0.386 (+3.7 %) | 0.446 / 0.437 (+2.1 %) | 0.487 / 0.481 (+1.3 %) | 0.520 / 0.519 (+0.2 %) | +1.8 % |
| ETTh2 | 0.351 / 0.333 (+5.4 %)* | 0.481 / 0.477 (+0.8 %) | 0.594 / 0.594 (+0.1 %) | 0.819 / 0.831 (-1.5 %) | +1.1 % |
| ETTm1 | 0.345 / 0.345 (+0.1 %) | 0.382 / 0.380 (+0.4 %) | 0.413 / 0.413 (+0.1 %) | 0.474 / 0.474 (-0.0 %) | +0.1 % |
| ETTm2 | 0.197 / 0.193 (+1.9 %) | 0.278 / 0.284 (-2.1 %) | 0.369 / 0.369 (+0.0 %) | 0.541 / 0.554 (-2.3 %) | -1.0 % |
| Exchange | 0.095 / 0.088 (+7.6 %)* | 0.182 / 0.176 (+3.2 %) | 0.327 / 0.313 (+4.5 %) | 0.749 / 0.839 (-10.7 %) | +0.9 % |
| Weather | 0.195 / 0.196 (-0.4 %) | 0.237 / 0.237 (-0.2 %) | 0.282 / 0.283 (-0.5 %) | 0.346 / 0.345 (+0.3 %) | -0.3 % |
| PEMS04 | 0.146 / 0.148 (-1.5 %) | 0.222 / 0.224 (-1.0 %) | 0.351 / 0.355 (-1.1 %) | 0.447 / 0.452 (-1.0 %) | -0.7 % |
| PEMS08 | 0.154 / 0.154 (-0.3 %) | 0.249 / 0.248 (+0.4 %) | 0.446 / 0.440 (+1.5 %) | 0.692 / 0.674 (+2.6 %) | -0.2 % |

32 settings; within 5 %: 30; mean |change| 1.8 %; more than 10 % worse: none.

#### PatchTST

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| ETTh1 | 0.427 / 0.414 (+3.1 %) | 0.472 / 0.460 (+2.7 %) | 0.509 / 0.501 (+1.7 %) | 0.507 / 0.500 (+1.5 %) | +1.9 % |
| ETTh2 | 0.303 / 0.302 (+0.3 %) | 0.389 / 0.388 (+0.3 %) | 0.426 / 0.426 (-0.1 %) | 0.428 / 0.431 (-0.6 %) | +0.1 % |
| ETTm1 | 0.376 / 0.329 (+14.3 %)** | 0.440 / 0.367 (+19.8 %)** | 0.466 / 0.399 (+16.9 %)** | 0.516 / 0.454 (+13.6 %)** | +6.6 % |
| ETTm2 | 0.187 / 0.175 (+6.7 %)* | 0.255 / 0.241 (+5.6 %)* | 0.310 / 0.305 (+1.7 %) | 0.414 / 0.402 (+3.1 %) | +3.3 % |
| Exchange | 0.094 / 0.088 (+6.5 %)* | 0.189 / 0.176 (+7.6 %)* | 0.343 / 0.301 (+13.9 %)** | 0.866 / 0.901 (-3.8 %) | +3.8 % |
| Weather | 0.186 / 0.177 (+5.2 %)* | 0.232 / 0.225 (+3.0 %) | 0.287 / 0.278 (+3.1 %) | 0.361 / 0.354 (+2.1 %) | +2.3 % |
| PEMS04 | 0.141 / 0.105 (+33.9 %)** | 0.221 / 0.153 (+44.7 %)** | 0.420 / 0.229 (+83.5 %)** | 0.795 / 0.291 (+173.2 %)** | +41.7 % |
| PEMS08 | 0.140 / 0.168 (-16.9 %) | 0.211 / 0.224 (-5.7 %) | 0.413 / 0.321 (+28.7 %)** | 0.872 / 0.408 (+113.8 %)** | +31.6 % |

32 settings; within 5 %: 16; mean |change| 19.9 %; more than 10 % worse: ETTm1-96 +14.3 %, ETTm1-192 +19.8 %, ETTm1-336 +16.9 %, ETTm1-720 +13.6 %, Exchange-336 +13.9 %, PEMS04-12 +33.9 %, PEMS04-24 +44.7 %, PEMS04-48 +83.5 %, PEMS04-96 +173.2 %, PEMS08-48 +28.7 %, PEMS08-96 +113.8 %.

#### Summary

| baseline | settings run | within 5 % | mean abs. change | > 10 % worse |
|---|---:|---:|---:|---:|
| S-Mamba | 32 | 31 | 1.8 % | 1 |
| iTransformer | 32 | 26 | 7.7 % | 4 |
| DLinear | 32 | 30 | 1.8 % | 0 |
| PatchTST | 32 | 16 | 19.9 % | 11 |

## 4. TiDE: the published numbers are not those of the official configuration

ETTh1-96, CPU, seed 2021, protocol of the check (published: 0.479 / 0.464):

| configuration | MSE / MAE | against published |
|---|---|---:|
| Time-Series-Library script (`scripts/benchmark/official/TSLib/ETT_script/TiDE_ETTh1.sh`: lr 0.1, batch 512, d_model 256, decoder output 8) | 0.385 / 0.393 | -19.6 % |
| TiDE paper, Table 8 (ETTh1, lookback 720 there: lr 3.82e-5, batch 512, hidden 256, temporal decoder 128) | 0.880 / 0.626 | +83.7 % |
| library defaults (lr 1e-4, batch 32, d_model 512): stopped after 8 of 10 epochs, validation MSE 0.753 against 0.69 with the script | – | – |

The official implementation with the official script is 20 % better than the published number, the TiDE paper's
settings at input 96 are far worse. With the same script TiDE is also better than published on ETTh2-96 (-28 %) and
Exchange-96 (-10 %) (table above). The published TiDE column cannot be reproduced from an official configuration;
the defaults may be closest but were not finished. The comparison should use our TiDE runs (a stronger baseline)
and say so, rather than the published column.

## 5. Verdict per baseline so far

| baseline | implementation | settings | forecasts |
|---|---|---|---|
| S-Mamba | official forward pass | official scripts | long-term sets within 5 % in 31 of 32 settings (earlier protocol); PEMS and the new protocol on the GPU |
| iTransformer | official | official scripts (ETTh1-336/720 corrected) | long-term sets within 5 % in 26 of 32 settings (earlier protocol); PEMS to be repeated (earlier runs had time marks) |
| PatchTST | Time-Series-Library | TSLib scripts (Solar, PEMS: Electricity line) | ETTh1-96 0.403 against 0.414 with the TSLib script (the paper's settings gave 0.427); the rest on the GPU |
| RLinear | port of the official code | RLinear `script.md` | 24 settings on six sets: 23 within 5 %, none worse by more than 10 % (mean abs. change 1.7 %) |
| DLinear | Time-Series-Library | TSLib scripts / defaults | 32 settings: 30 within 5 %, none worse by more than 10 % (earlier protocol; DLinear ignores time marks) |
| TiDE | Time-Series-Library | TSLib script (ETTh1 line everywhere) | at horizon 96 better than published by 20 % (ETTh1), 28 % (ETTh2) and 10 % (Exchange); the published column is not reproducible (section 4) |

Still to run on the GPU (README.md (GPU Runs), "Baseline check"): every baseline on every dataset with the protocol of the
check; then `python scripts/benchmark/baseline_check.py`.
