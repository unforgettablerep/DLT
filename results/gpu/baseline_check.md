# Baseline check on the GPU: Weather and PEMS04 (2026-11-18)

**Source.** [`baselines.jsonl`](baselines.jsonl): 45 runs from the RTX 4090 (PyTorch 2.10.0+cu130), seed 2021, suite
`baselines` of `scripts/exps.py`. Every run uses the settings of the official scripts (`scripts/baselines.py`) and
the protocol of the official loaders: four time features, dropped last training / validation batch, test batch 1;
on PEMS no time marks and the split of the iTransformer loader. S-Mamba ran with `mamba_ssm`. No forecast was NaN.
Coverage: Weather complete (6 baselines × 4 horizons); PEMS04 complete for five baselines, TiDE 1 of 4 horizons.
Tables: `python scripts/p1/baseline_check.py --files results/gpu/baselines.jsonl`.

## Weather: all six baselines reproduce the S-Mamba paper

- **All 24 settings are within 5 % of the published MSE:**
  - S-Mamba +0.3 to +1.3 %;
  - iTransformer +0.7 to +1.5 %;
  - PatchTST +0.3 to +0.8 %;
  - RLinear −1.4 to +0.6 %;
  - DLinear −0.5 to +0.3 %;
  - TiDE −4.7 to +3.8 %.
- **PatchTST matches with the Time-Series-Library script** (3 epochs on Weather). The PatchTST paper's settings,
  used until 2026-11-15, were 2–5 % behind.
- **RLinear on the GPU equals the CPU run to 1e-4** (Weather-96: 0.19308 against 0.19306), even where the best
  epoch differs: the pipeline does not depend on the device.
- **TiDE matches the paper on Weather.** At horizon 96 on ETTh1, ETTh2 and Exchange ours was 10–28 % better
  ([CPU check](../p1/baseline_check_cpu.md)). On Weather its MAE is 9 % better than published.

## PEMS04: four baselines match or beat the paper; iTransformer at 48 / 96 and PatchTST do not

- **The PEMS pipeline is right.** DLinear is within ±0.3 %, RLinear 2–6 % better and S-Mamba 0–11 % better than
  published, all three with the same loader and protocol.
- **TiDE**, horizon 12: 37 % better than published. As on the long-term sets, the published TiDE column is weak.
- **iTransformer: +4 %, +5 %, +13 %, +14 %** at horizons 12 to 96, with the official settings (4 layers, d_model
  1024, learning rate 5e-4, no instance normalisation).
  - At horizons 24 to 96 the best epoch is the last one (10 of 10): the validation loss still fell when the
    schedule ended (the learning rate halves every epoch, to 1e-6 at epoch 10).
  - S-Mamba, with the same loader, beats its published numbers, so the data are not the cause. Left: the seed
    (one run) and the learning rate.
- **PatchTST: +6 %, +18 %, +50 %, +112 %.** There is no official PEMS setting; the Electricity line (2 layers,
  d_model 512, learning rate 1e-4, batch 16) is far off at the long horizons. At horizon 96 it is worse than
  DLinear.

## Next on the GPU

1. PEMS04 TiDE, the remaining three horizons (already queued).
2. Learning-rate check for iTransformer and PatchTST on PEMS04: the same settings at the other learning rates of
   {1e-4, 5e-4, 1e-3}, 16 runs. `baseline_check.py` then takes, per setting, the learning rate with the lowest
   validation loss and names it in the table:
   ```bash
   python scripts/exps.py baselines --device gpu --seeds 2021 --models iTransformer PatchTST --datasets PEMS04 \
       --backbone_lr 0.0001 0.0005 0.001 --list bl_lrcheck
   bash scripts/gpu/run_gpu_queue.sh bl_lrcheck 0 0
   ```
3. A second seed for iTransformer on PEMS04, to see how much of its gap is seed spread (4 runs):
   ```bash
   python scripts/exps.py baselines --device gpu --seeds 2022 --models iTransformer --datasets PEMS04 --list bl_seed
   bash scripts/gpu/run_gpu_queue.sh bl_seed 0
   ```
4. The other datasets of the check, as in GPU_RUNS.md ("Baseline check").

## Tables

Change = ours / published - 1 (MSE per horizon; MAE averaged over the horizons). * more than 5 % worse than published, ** more than 10 %. Seeds averaged; n = runs per setting.

### S-Mamba

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| Weather | 0.167 / 0.165 (+1.1 %) | 0.215 / 0.214 (+0.3 %) | 0.275 / 0.274 (+0.4 %) | 0.354 / 0.350 (+1.3 %) | +0.5 % |
| PEMS04 | 0.073 / 0.076 (-4.0 %) | 0.084 / 0.084 (-0.3 %) | 0.103 / 0.115 (-10.7 %) | 0.123 / 0.137 (-10.2 %) | -3.1 % |

8 settings; within 5 %: 8; mean |change| 3.5 %; more than 10 % worse: none.

### iTransformer

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| Weather | 0.175 / 0.174 (+0.7 %) | 0.224 / 0.221 (+1.5 %) | 0.280 / 0.278 (+0.7 %) | 0.360 / 0.358 (+0.7 %) | +0.8 % |
| PEMS04 | 0.081 / 0.078 (+4.2 %) | 0.099 / 0.095 (+4.5 %) | 0.135 / 0.120 (+12.8 %)** | 0.171 / 0.150 (+13.9 %)** | +5.1 % |

8 settings; within 5 %: 6; mean |change| 4.9 %; more than 10 % worse: PEMS04-48 +12.8 %, PEMS04-96 +13.9 %.

### PatchTST

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| Weather | 0.178 / 0.177 (+0.8 %) | 0.226 / 0.225 (+0.3 %) | 0.280 / 0.278 (+0.6 %) | 0.356 / 0.354 (+0.4 %) | +0.3 % |
| PEMS04 | 0.111 / 0.105 (+6.2 %)* | 0.181 / 0.153 (+18.2 %)** | 0.344 / 0.229 (+50.3 %)** | 0.616 / 0.291 (+111.7 %)** | +20.3 % |

8 settings; within 5 %: 4; mean |change| 23.5 %; more than 10 % worse: PEMS04-24 +18.2 %, PEMS04-48 +50.3 %, PEMS04-96 +111.7 %.

### RLinear

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| Weather | 0.193 / 0.192 (+0.6 %) | 0.237 / 0.240 (-1.3 %) | 0.288 / 0.292 (-1.4 %) | 0.359 / 0.364 (-1.2 %) | -0.5 % |
| PEMS04 | 0.129 / 0.138 (-6.2 %) | 0.245 / 0.258 (-5.1 %) | 0.552 / 0.572 (-3.4 %) | 1.109 / 1.137 (-2.4 %) | -3.6 % |

8 settings; within 5 %: 8; mean |change| 2.7 %; more than 10 % worse: none.

### DLinear

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| Weather | 0.196 / 0.196 (+0.0 %) | 0.237 / 0.237 (-0.1 %) | 0.282 / 0.283 (-0.5 %) | 0.346 / 0.345 (+0.3 %) | -0.1 % |
| PEMS04 | 0.148 / 0.148 (-0.3 %) | 0.225 / 0.224 (+0.3 %) | 0.356 / 0.355 (+0.2 %) | 0.453 / 0.452 (+0.3 %) | +0.1 % |

8 settings; within 5 %: 8; mean |change| 0.3 %; more than 10 % worse: none.

### TiDE

| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |
|---|---|---|---|---|---:|
| Weather | 0.193 / 0.202 (-4.7 %) | 0.240 / 0.242 (-0.8 %) | 0.292 / 0.287 (+1.7 %) | 0.364 / 0.351 (+3.8 %) | -9.3 % |
| PEMS04 | 0.139 / 0.219 (-36.7 %) | - / 0.292 | - / 0.409 | - / 0.492 | -25.9 % |

5 settings; within 5 %: 5; mean |change| 9.5 %; more than 10 % worse: none.

### Summary

| baseline | settings run | within 5 % | mean abs. change | > 10 % worse |
|---|---:|---:|---:|---:|
| S-Mamba | 8 | 8 | 3.5 % | 0 |
| iTransformer | 8 | 6 | 4.9 % | 2 |
| PatchTST | 8 | 4 | 23.5 % | 3 |
| RLinear | 8 | 8 | 2.7 % | 0 |
| DLinear | 8 | 8 | 0.3 % | 0 |
| TiDE | 5 | 5 | 9.5 % | 0 |