# DLT: Distributed-Lag Transfer for Multivariate Time Series Forecasting

> **中文简介**：本仓库是 DLT（跨变量分布滞后传递）方向的可行性原型。核心思想是为每对相关变量学习一条**多时滞核** K_ij(τ)，用共享的时滞基降低参数量，并让核系数随输入窗口自适应（DLT-dyn）。
> 仓库包含线性原型代码、69 组对比实验的原始结果和架构图（SVG/PDF/PNG）。这些是可行性验证结果，不是论文最终结果。

![architecture](figures/dlt_architecture.png)

## Idea
Cross-variable effects in multivariate series are often **lagged and spread over several lags**
(e.g. upstream → downstream traffic sensors). Existing channel-dependent models either mix whole
series with one scalar per pair (iTransformer/VCformer-style), shift a source by a single estimated
lag (LIFT / TLGNN / DTLGSL-style), or use very short static kernels (CrossLinear-style).
**DLT** gives every target–source pair its own multi-lag kernel

$$z_i(t)=\sum_{j\in N(i)}\sum_{\tau=0}^{L}K_{ij}(\tau)\,\tilde x_j(t-\tau),\qquad K_{ij}(\tau)=\sum_{r=1}^{R}\alpha_{ijr}\,\phi_r(\tau)$$

with a small bank of shared basis kernels φ_r (cost linear in the number of variables), and
**DLT-dyn** makes α input-adaptive: α_ijr(X) = α_ijr · (1 + tanh(MLP([s_i ‖ s_j]))).

## Feasibility prototype
All variants share RevIN, the same top-k neighbour sets N(i) (selected offline by |CCF| on the
training split), the same intra-variable head W_self and cross head W_cross. **They differ only in
how K_ij(τ) is parameterised:**

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

Settings: lookback T = 96, horizon H ∈ {24, 96}; k = 8 neighbours, L = 48 lags, R = 4 basis kernels;
Adam (lr 3e-3, ×0.5 per epoch from epoch 2), ≤ 6 epochs, early stopping on validation MSE
(patience 2); Time-Series-Library splits (PEMS08 6:2:2), z-score on train; CPU only.

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
1. **Multi-lag kernels beat single-lag shifts and scalar mixing** on every coupled dataset
   (static DLT vs. `shift`: −4.9 / −1.5 / −7.7 / −16.8 % on Weather / Electricity / PEMS08-24 / PEMS08-96).
2. **The shared basis matches the free kernel** (better on Weather and PEMS08-24, 0.8 % worse on PEMS08-96)
   with ~12× fewer cross-kernel parameters (PEMS08: 5.6 K vs. 66.6 K).
3. **Input-adaptive kernels give the largest gain and need the lag kernel**: DLT-dyn beats the
   dynamic scalar control by 4.7 / 5.5 / 11.2 / 23.4 % and is more stable across seeds
   (PEMS08-96: 0.555 / 0.549 vs. 0.681 / 0.760).
4. **Weakly coupled data (ETTh1) does not benefit**; DLT-dyn overfits there (+4.8 %) — a gate /
   regulariser is needed before full benchmarking.

Premise check (`code/pems_lag.py`): on PEMS08 road edges, 52 % of upstream→downstream pairs peak at
a one-step (5 min) lead, and 65 % spread their response over ≥ 2 lags.

Caveats: linear prototype, ≤ 6 epochs, no tuning, 1–2 seeds, lookback 96 only; absolute errors are far
from deep state-of-the-art models — only the relative comparison between variants is meaningful.

## Repository layout
```
code/proto.py        prototype model + training/eval (all 8 variants)
code/pems_lag.py     lag premise check on PEMS08 road edges
code/agg.py          console summary of a runs file
code/agg_md.py       markdown results table (results/summary.md)
code/make_arch.py    generates figures/dlt_architecture.svg
code/render.js       SVG -> PNG/PDF with Playwright Chromium
scripts/download_data.sh   fetch datasets into ./data (not tracked)
scripts/run_grid.sh        reproduce all 69 runs
results/runs.jsonl   one JSON line per run (MSE, MAE, params, time, lag statistics)
results/summary.md   results table
figures/             architecture figure (SVG: editable/Visio-importable; PDF: vector for LaTeX; PNG: 4800×3180)
tslib/               Time-Series-Library snapshot with DLT (models/DLT.py, scripts/long_term_forecast/DLT_script/)
```

## Reproduce
```bash
pip install -r requirements.txt
bash scripts/download_data.sh

# a single run
python code/proto.py --data PEMS08 --pred_len 96 --root data --variant dltdyn --out results/my_runs.jsonl

# the full grid (CPU; ~1–2 h with 4 workers), then the table
bash scripts/run_grid.sh
python code/agg_md.py results/runs_reproduced.jsonl

# lag premise check and the figure
python code/pems_lag.py data/PEMS08.npz data/PEMS08_edges.csv
python code/make_arch.py figures/dlt_architecture.svg
node code/render.js figures/dlt_architecture.svg figures/dlt_architecture.png figures/dlt_architecture.pdf 3
```

## Time-Series-Library port (`tslib/`) — GPU-ready
`tslib/` is a snapshot of [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
(commit `4e938a1`, MIT) with DLT added as `--model DLT`; every change to upstream is listed in
[`tslib/DLT_CHANGES.md`](tslib/DLT_CHANGES.md). The TSLib model is a vectorised re-implementation
of `code/proto.py` (no Python loop over neighbours) and is device-agnostic: CPU, single / multi-GPU
(DataParallel), AMP and Apple MPS.

```bash
cd tslib
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
pip install -r requirements_dlt.txt
# ETT / Weather / ECL / Traffic / Exchange: put the CSVs under ./dataset/... as in the TSLib scripts
# (missing CSVs are downloaded from the TSLib Hugging Face repo automatically).
# PEMS08 is not in that repo:
mkdir -p dataset/PEMS && curl -fL -o dataset/PEMS/PEMS08.npz \
  https://huggingface.co/datasets/jimmygao3218/PEMS08/resolve/main/PEMS08.npz

bash scripts/long_term_forecast/DLT_script/DLT_PEMS08.sh                 # GPU 0, DLT-dyn, H = 12/24/48/96
VARIANT=dlt bash scripts/long_term_forecast/DLT_script/DLT_Weather.sh    # another kernel variant
bash scripts/long_term_forecast/DLT_script/DLT_ablation.sh               # all 8 variants (PEMS08 + Weather)
EXTRA_ARGS="--no_use_gpu --num_workers 0" bash scripts/long_term_forecast/DLT_script/DLT_ETTh1.sh   # CPU
EXTRA_ARGS="--use_multi_gpu --devices 0,1" bash scripts/long_term_forecast/DLT_script/DLT_Traffic.sh # 2 GPUs
EXTRA_ARGS="--use_amp" bash scripts/long_term_forecast/DLT_script/DLT_ECL.sh                        # AMP
```
Scripts exist for ETTh1/2, ETTm1/2, Weather, ECL, Traffic, Exchange and PEMS08. Metrics are appended
to `tslib/result_long_term_forecast.txt`; predictions go to `tslib/results/<setting>/`.
DLT-specific flags: `--dlt_variant {ci,scalar,shift,tap3,dlt,full,scalardyn,dltdyn}`, `--dlt_k`,
`--dlt_L`, `--dlt_R`, `--dlt_dyn_dim`, `--dlt_ccf_len`, `--dlt_impl {kernel,basis}` (two numerically
identical computation paths for `dlt`/`dltdyn`; `kernel` is the default and uses less memory).

**GPU memory.** DLT itself is small (≤ 0.36 M parameters even at Traffic scale) and one training step at
batch 32 needs ≤ ~0.75 GB of activations (table below), so any CUDA GPU with ≥ 4 GB runs it. For the full
benchmark the baselines (e.g. iTransformer on Traffic) dominate; a single 24 GB card (RTX 3090/4090,
A5000, A10) is comfortable.

### Verification of the port
* **Equivalence with `code/proto.py`:** with identical parameters and neighbours, all 8 variants (and both
  computation paths) give the same outputs (max |Δ| ≤ 1.1e-6) and gradients (≤ 1.2e-7).
* **Device-agnostic:** every variant runs forward + backward on PyTorch's `meta` device, i.e. no tensor is
  pinned to the CPU inside the model. *Not yet run on a physical GPU* (the development machine had none).
* **End-to-end in TSLib (CPU):** all variants train and test; without CUDA the run falls back to the CPU;
  test-only mode (`--is_training 0`) reloads the checkpoint (incl. neighbours) and reproduces the metric.
* **Same results as the prototype** (MSE, H = 96 unless noted; prototype = mean over its seeds, TSLib = one
  run with TSLib's seed and data pipeline, prototype hyper-parameters: lr 3e-3, ≤ 6 epochs, patience 2):

| setting | ci | scalar | dlt | scalardyn | dltdyn |
|---|---|---|---|---|---|
| PEMS08 – prototype | 1.168 | 1.017 | 0.863 | 0.721 | 0.552 |
| PEMS08 – TSLib | 1.169 | 1.040 | 0.858 | 0.674 | **0.526** |
| PEMS08 H=24 – prototype / TSLib | | | 0.179 / 0.180 | | 0.133 / **0.133** |
| Weather – prototype | 0.193 | | 0.167 | 0.172 | 0.164 |
| Weather – TSLib | 0.193 | | 0.167 | 0.176 | **0.165** |
| ETTh1 – prototype | **0.390** | | 0.393 | | 0.408 |
| ETTh1 – TSLib | **0.389** | | 0.396 | | 0.405 |

  (These TSLib runs used `--dlt_impl basis`; repeating ETTh1 dlt/dltdyn, Weather dltdyn and PEMS08 H=24
  dltdyn with the default `kernel` path gives the same MSE to four decimals.)

* **Speed / memory** of one training step (CPU, 4 threads, batch 32, T = H = 96, k = 8, L = 48, R = 4):

| variables | dlt (kernel) | dlt (basis) | dltdyn (kernel) | dltdyn (basis) | full | prototype loop: dlt / dltdyn |
|---|---|---|---|---|---|---|
| 862 (Traffic) | 0.77 s, 527 MB | 1.03 s, 904 MB | 0.85 s, 734 MB | 0.89 s, 1028 MB | 0.59 s, 552 MB | 1.79 s / 1.17 s |
| 321 (Electricity) | 0.20 s, 271 MB | 0.36 s, 398 MB | 0.25 s, 360 MB | 0.30 s, 445 MB | 0.28 s, 262 MB | 0.35 s / 0.27 s |
| 170 (PEMS08) | 0.13 s, 203 MB | 0.20 s, 290 MB | 0.19 s, 243 MB | 0.16 s, 310 MB | 0.13 s, 202 MB | – |

  Parameters at Traffic scale: dlt 46 K, dltdyn 49 K vs. free kernel 357 K.

## Next steps
Gate/regulariser for weakly coupled data; a stronger temporal backbone; full GPU benchmark in `tslib/`
(ETT×4, Weather, Electricity, Traffic, Exchange, Solar, PEMS03/04/07/08; 4 horizons; 3 seeds) against
iTransformer, PatchTST, TimeMixer, TimeFilter (already in TSLib) and DUET, CrossLinear, LIFT, VCformer,
LagTS (to be added); visualise learned kernels on PEMS08.
