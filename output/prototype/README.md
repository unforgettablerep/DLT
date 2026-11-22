# Feasibility prototype

The first study of the idea, before the integration into the forecasting framework: a linear prototype
(`scripts/prototype/proto.py`), 69 runs on the CPU. Records: [`runs.jsonl`](runs.jsonl); table:
[`summary.md`](summary.md) (`python scripts/prototype/agg_md.py`).

All variants share RevIN, the same top-k neighbour sets N(i) (selected offline by |CCF| on the training split), the
same intra-variable head W_self and cross head W_cross. **They differ only in how K_ij(τ) is parameterised:**

| variant | K_ij(τ) | resembles |
|---|---|---|
| `ci` | no cross branch | RLinear |
| `scalar` | a_ij at τ = 0 | iTransformer / VCformer-style mixing |
| `shift` | a_ij at τ = d_ij (CCF peak lag) | LIFT / TLGNN / DTLGSL-style |
| `tap3` | free, τ = 0..2 | CrossLinear-style |
| `dlt` | Σ_r α_ijr φ_r(τ), shared basis | **proposed** |
| `full` | free, τ = 0..L | unconstrained upper bound |
| `scalardyn` | `scalar` with input-adaptive a_ij | control: dynamics without a lag kernel |
| `dltdyn` | `dlt` with input-adaptive α | **proposed (dynamic)** |

Settings: lookback T = 96, horizon H ∈ {24, 96}; k = 8 neighbours, L = 48 lags, R = 4 basis kernels; Adam (lr 3e-3,
×0.5 per epoch from epoch 2), ≤ 6 epochs, early stopping on validation MSE (patience 2); Time-Series-Library splits
(PEMS08 6:2:2), z-score on train; CPU only.

## Results (69 runs)

MSE on the z-scored test set, mean over seeds (¹ = single seed); % = change vs. `ci`; bold = best per column.

| variant | ETTh1 H=96 | Weather H=96 | Electricity H=96 | PEMS08 H=24 | PEMS08 H=96 |
|---|---|---|---|---|---|
| CI (no cross) | **0.3895 (+0.0%)** | 0.1932 (+0.0%) | 0.1975 (+0.0%)¹ | 0.2326 (+0.0%) | 1.1678 (+0.0%) |
| scalar mixing | 0.3900 (+0.1%) | 0.1747 (-9.5%) | 0.1902 (-3.7%)¹ | 0.1904 (-18.1%) | 1.0170 (-12.9%) |
| single-lag shift | 0.3899 (+0.1%) | 0.1750 (-9.4%) | 0.1895 (-4.0%)¹ | 0.1944 (-16.4%) | 1.0375 (-11.2%)¹ |
| 3-tap kernel | 0.3897 (+0.0%) | 0.1730 (-10.5%) | 0.1875 (-5.1%)¹ | 0.1853 (-20.3%) | 0.9295 (-20.4%) |
| DLT (static) | 0.3927 (+0.8%) | 0.1665 (-13.8%) | 0.1866 (-5.5%)¹ | 0.1794 (-22.9%) | 0.8634 (-26.1%) |
| free kernel | 0.3934 (+1.0%) | 0.1694 (-12.3%) | – | 0.1805 (-22.4%) | 0.8568 (-26.6%)¹ |
| scalar + dynamic (control) | 0.3933 (+1.0%) | 0.1717 (-11.1%) | 0.1826 (-7.5%)¹ | 0.1495 (-35.7%) | 0.7205 (-38.3%) |
| DLT-dyn | 0.4083 (+4.8%) | **0.1637 (-15.3%)** | **0.1725 (-12.7%)¹** | **0.1328 (-42.9%)** | **0.5522 (-52.7%)** |

Key observations:
1. **Multi-lag kernels beat single-lag shifts and scalar mixing** on every coupled dataset (static DLT vs. `shift`:
   −4.9 / −1.5 / −7.7 / −16.8 % on Weather / Electricity / PEMS08-24 / PEMS08-96).
2. **The shared basis matches the free kernel** (better on Weather and PEMS08-24, 0.8 % worse on PEMS08-96) with ~12×
   fewer cross-kernel parameters (PEMS08: 5.6 K vs. 66.6 K).
3. **Input-adaptive kernels give the largest gain and need the lag kernel**: DLT-dyn beats the dynamic scalar control
   by 4.7 / 5.5 / 11.2 / 23.4 % and is more stable across seeds (PEMS08-96: 0.555 / 0.549 vs. 0.681 / 0.760).
4. **Weakly coupled data (ETTh1) does not benefit**; DLT-dyn overfits there (+4.8 %).

Premise check (`python scripts/mechanism/pems_lag.py`): on PEMS08 road edges, 52 % of upstream→downstream pairs peak at
a one-step (5 min) lead, and 65 % spread their response over ≥ 2 lags.

Caveats: linear prototype, ≤ 6 epochs, no tuning, 1–2 seeds, lookback 96 only; absolute errors are far from deep
state-of-the-art models — only the relative comparison between variants is meaningful.

## From the prototype to `src/model/DLT.py`

`src/model/DLT.py` is a vectorised re-implementation of the prototype (no Python loop over neighbours).
- **Equivalence:** with identical parameters and neighbours, all 8 prototype variants (and both computation paths)
  give the same outputs (max |Δ| ≤ 1.1e-6) and gradients (≤ 1.2e-7).
- **Same results in the full pipeline** (MSE, H = 96 unless noted; prototype = mean over its seeds, framework = one
  run with its seed and data pipeline, prototype hyper-parameters: lr 3e-3, ≤ 6 epochs, patience 2, basis path):

| setting | ci | scalar | dlt | scalardyn | dltdyn |
|---|---|---|---|---|---|
| PEMS08 – prototype | 1.168 | 1.017 | 0.863 | 0.721 | 0.552 |
| PEMS08 – framework | 1.169 | 1.040 | 0.858 | 0.674 | **0.526** |
| PEMS08 H=24 – prototype / framework | | | 0.179 / 0.180 | | 0.133 / **0.133** |
| Weather – prototype | 0.193 | | 0.167 | 0.172 | 0.164 |
| Weather – framework | 0.193 | | 0.167 | 0.176 | **0.165** |
| ETTh1 – prototype | **0.390** | | 0.393 | | 0.408 |
| ETTh1 – framework | **0.389** | | 0.396 | | 0.405 |

- **Speed / memory** of one training step (CPU, 4 threads, batch 32, T = H = 96, k = 8, L = 48, R = 4):

| variables | dlt (kernel) | dlt (basis) | dltdyn (kernel) | dltdyn (basis) | full | prototype loop: dlt / dltdyn |
|---|---|---|---|---|---|---|
| 862 (Traffic) | 0.77 s, 527 MB | 1.03 s, 904 MB | 0.85 s, 734 MB | 0.89 s, 1028 MB | 0.59 s, 552 MB | 1.79 s / 1.17 s |
| 321 (Electricity) | 0.20 s, 271 MB | 0.36 s, 398 MB | 0.25 s, 360 MB | 0.30 s, 445 MB | 0.28 s, 262 MB | 0.35 s / 0.27 s |
| 170 (PEMS08) | 0.13 s, 203 MB | 0.20 s, 290 MB | 0.19 s, 243 MB | 0.16 s, 310 MB | 0.13 s, 202 MB | – |

Parameters at Traffic scale: dlt 46 K, dltdyn 49 K vs. free kernel 357 K.

## Reproduce

```bash
python scripts/prototype/proto.py --data PEMS08 --pred_len 96 --variant dltdyn --out output/prototype/my_runs.jsonl
bash scripts/prototype/run_grid.sh            # the full grid (CPU; ~1–2 h with 4 workers)
python scripts/prototype/agg_md.py output/prototype/runs_reproduced.jsonl
```
