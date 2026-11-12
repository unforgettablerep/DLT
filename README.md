# DLT: Distributed-Lag Transfer for Multivariate Time Series Forecasting

> **中文简介**：本仓库是 DLT（跨变量分布滞后传递）的代码仓库：一份 Time-Series-Library 快照加上 DLT 模型（`--model DLT`），以及全部实验脚本和结果表。核心思想是为每对相关变量学习一条**多时滞核** K_ij(τ)，用共享的时滞基降低参数量，并让核系数随输入窗口自适应（DLT-dyn）。
> 论文稿、相关论文 PDF 和所有图片在 `DLTpaper` 仓库；本仓库只保留代码、实验记录和结果表。

## Quick start: one GPU, one dataset, stand-alone DLT-dyn

Reference environment: one RTX 4090, host CUDA 13.1, PyTorch 2.5.1+cu124. The cu124 wheel ships its own CUDA 12.4
runtime, so the host only needs a recent NVIDIA driver. Run everything from the repository root.

```bash
# 1. environment
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. data: fixed directory ./dataset (Time-Series-Library layout, e.g. dataset/weather/weather.csv)
bash scripts/download_data.sh             # or copy / link an existing copy; a missing CSV is also downloaded at run time

# 3. the GPU computes what the CPU computes (stand-alone DLT-dyn, under a minute)
python scripts/gpu/check_gpu.py           # must end with "all checks passed"

# 4. one training run: DLT-dyn, Weather, horizon 96, GPU 0, seed 2021
bash scripts/train_dltdyn.sh
```

There is nothing to configure: every script reads the datasets from `dataset/` (or from `../data` next to the
repository when `dataset/` does not exist) and writes job lists, logs, checkpoints and result records to `work/`
([`scripts/paths.py`](scripts/paths.py)). Both are ignored by git; if the data or an earlier work directory is
elsewhere, make `dataset` or `work` a symbolic link to it. S-Mamba finds `mamba_ssm` in the environment or in the
source checkouts `../mamba` and `../causal-conv1d`; `python scripts/gpu/check_mamba.py` checks that.

The script ends with one line, for example (CPU, PyTorch 2.5.1, this exact command):

```
Weather_96_96 seed 2021: MSE 0.1642  MAE 0.2117  | best epoch 7 of 10, 26.2 s per epoch, 22232 parameters
```

On the GPU expect the same metrics to about three decimals: cuDNN and TensorFloat-32 are not bit-identical to the
CPU. Other datasets and horizons: `bash scripts/train_dltdyn.sh PEMS08 12` (dataset, horizon, then optionally the
seed and further `run.py` arguments; see the script header). CPU reference values of the default configuration,
mean ± std over seeds 2021–2023:

| dataset / horizon | DLT-dyn | same model without the cross branch (CI) | CPU s / epoch (1 thread) |
|---|---|---|---:|
| Weather / 96 (MSE, MAE) | 0.1637 ± 0.0006, 0.2101 ± 0.0014 | 0.1942, 0.2335 | 25 |
| ECL / 96 (MSE, MAE) | 0.1713 ± 0.0007, 0.2619 ± 0.0003 | 0.1975, 0.2739 | 201 |
| PEMS08 / 96 (MSE, MAE, scaled) | 0.5229 ± 0.0087, 0.4945 ± 0.0055 | 1.1681, 0.8059 | 56 |
| PEMS08 / 12 (MAE, MAPE, RMSE; TimeMixer protocol) | 17.87 ± 0.14, 10.68 %, 28.30 | 21.84, 12.62 %, 35.01 | 54 |
| PEMS04 / 12 (MAE, MAPE, RMSE; TimeMixer protocol) | 22.31 ± 0.06, 13.85 %, 35.10 | 27.36, 16.34 %, 43.54 | 100 |
| ETTh1 / 96 (MSE, MAE) | 0.4102 ± 0.0055, 0.4140 ± 0.0028 | 0.3887, 0.3957 | 2 |

ETTh1 is listed on purpose: on weakly coupled data the cross-variable branch does not help (see
[`results/README.md`](results/README.md)).

Datasets: `scripts/download_data.sh` fetches the Time-Series-Library CSVs and PEMS08 into `dataset/`. The bundle
is also on [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing),
[Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) and
[Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library). PEMS04 / PEMS08 (`PEMS0x.npz`, flow in
channel 0) are not part of it; they belong in `dataset/PEMS/` (PEMS04 has to be added by hand).

## Idea
Cross-variable effects in multivariate series are often **lagged and spread over several lags**
(e.g. upstream → downstream traffic sensors). Existing channel-dependent models either mix whole
series with one scalar per pair (iTransformer/VCformer-style), shift a source by a single estimated
lag (LIFT / TLGNN / DTLGSL-style), or use very short static kernels (CrossLinear-style).
**DLT** gives every target–source pair its own multi-lag kernel

$$z_i(t)=\sum_{j\in N(i)}\sum_{\tau=0}^{L}K_{ij}(\tau)\,\tilde x_j(t-\tau),\qquad K_{ij}(\tau)=\sum_{r=1}^{R}\alpha_{ijr}\,\phi_r(\tau)$$

with a small bank of shared basis kernels φ_r (cost linear in the number of variables), and
**DLT-dyn** makes α input-adaptive: α_ijr(X) = α_ijr · (1 + tanh(MLP([s_i ‖ s_j]))).


## Repository layout
```
run.py  exp/  data_provider/  layers/  models/  utils/     Time-Series-Library; ours: models/DLT.py, DLTPlug.py,
                                     LagTS.py, SMamba.py (S-Mamba baseline) and layers/MambaPy.py
scripts/paths.py                     the fixed data and work directories (dataset/, work/)
scripts/train_dltdyn.sh              quick start: one GPU, one dataset, stand-alone DLT-dyn
scripts/gpu/                         GPU-vs-CPU check, job lists and queue of the remaining GPU experiments
scripts/long_term_forecast/DLT_script/   DLT per dataset at four horizons; */SMamba*.sh: official S-Mamba settings
                                     (the other scripts of the *_script/ folders are upstream)
scripts/exps.py  scripts/p0 p1 p2/   experiment suites, validation search, aggregation, unit tests
scripts/run_queue.sh  run_missing.py  run_p1p2_cpu.sh     resumable CPU queues
scripts/collect_results.py           merges the records of work/ into results/ and rebuilds the tables
results/                             conclusions (README.md), tables and raw records of P0 / P1 / P2
code/                                first feasibility prototype (proto.py) with results/runs.jsonl, results/summary.md
DLT_CHANGES.md                       every change to the upstream snapshot
GPU_RUNS.md                          experiments that still need a GPU, for one RTX 4090
requirements.txt                     DLT and the classic models; requirements_tslib_full.txt = upstream list
```
Run-time output of `run.py` (`checkpoints/`, `results/<setting>/`, `test_results/`, `result_long_term_forecast.txt`)
is ignored by git; `results/p0`, `p1`, `p2` are the tracked experiment records.

## Experiments and results
* **Conclusions:** [`results/README.md`](results/README.md) (P1 / P2) and [`results/p0/README.md`](results/p0/README.md) (P0).
* P0 kernel / gate study, 195 runs: [`results/p0/summary.md`](results/p0/summary.md)
* P1 protocol comparisons, plug-in, validation search, baselines: [`results/p1/summary.md`](results/p1/summary.md)
* P2 sensitivity, neighbour selection, efficiency, learned kernels vs the PEMS08 road graph: [`results/p2/summary.md`](results/p2/summary.md)
* Still to run on the GPU (Traffic, plug-in on large data, search on ECL / Traffic / PEMS, baseline re-runs,
  GPU efficiency): [`GPU_RUNS.md`](GPU_RUNS.md)

All of these ran on a 4-core CPU with the code of this repository; the unit tests are
`python scripts/p0/test_variants.py` and `python scripts/p0/test_p1p2.py`.

**Baselines.** Every comparison (accuracy and efficiency) uses the same eight representative models, one per
family: Informer, TimesNet, DLinear, PatchTST, iTransformer, TimeMixer, TimeXer and S-Mamba (`BASELINES` in
[`scripts/exps.py`](scripts/exps.py)), each with its official hyper-parameters. S-Mamba is not in
Time-Series-Library; `models/SMamba.py` re-implements it after the official code (`mamba_ssm` block if
installed, a pure-PyTorch block otherwise).

## DLT in Time-Series-Library
The repository root is a snapshot of [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
(commit `4e938a1`, MIT, see `LICENSE`) with DLT added as `--model DLT`; every change to upstream is listed in
[`DLT_CHANGES.md`](DLT_CHANGES.md). The model is a vectorised re-implementation of `code/proto.py` (no Python
loop over neighbours) and is device-agnostic.

```bash
bash scripts/long_term_forecast/DLT_script/DLT_PEMS08.sh                 # GPU 0, DLT-dyn, H = 12/24/48/96
VARIANT=dlt bash scripts/long_term_forecast/DLT_script/DLT_Weather.sh    # another kernel variant
bash scripts/long_term_forecast/DLT_script/DLT_ablation.sh               # all variants (PEMS08 + Weather)
EXTRA_ARGS="--no_use_gpu --num_workers 0" bash scripts/long_term_forecast/DLT_script/DLT_ETTh1.sh   # CPU
```
Main flags (all in `run.py --help`, details in `models/DLT.py`):
* `--dlt_variant` `dltdyn` (default) | `dlt` | `ci` | `scalar` | `shift` | `tap3` | `full` | `scalardyn` | `topk` | `mill` | `milldyn`
* `--dlt_k` neighbours (8), `--dlt_L` maximum lag (48), `--dlt_R` basis kernels (4), `--dlt_dyn_dim` (16)
* `--dlt_impl` `kernel` | `basis` | `dense`: identical results; `basis` is the path of all reported DLT-dyn runs
* `--dlt_neighbours` `ccf` | `random` | `all`, `--dlt_gate` `none` | `moe` | `static`
* `--model DLTPlug --plug_base <model>` adds the DLT branch to another forecaster; `--model LagTS` is our
  re-implementation of LagTS
* `--result_jsonl <file>` appends one JSON record per run; `--no_save_preds` streams the test metrics

**GPU memory.** DLT itself is small (≤ 0.36 M parameters even at Traffic scale; DLT-dyn needs about 1 GB per
training step at batch 32 on Traffic), so the 24 GB of an RTX 4090 are needed only by the baselines (e.g.
iTransformer on Traffic: 12.4 GB at batch 32 on the CPU).

### Verification of the port
* **Equivalence with `code/proto.py`:** with identical parameters and neighbours, all 8 prototype variants (and both
  computation paths) give the same outputs (max |Δ| ≤ 1.1e-6) and gradients (≤ 1.2e-7).
* **Device-agnostic:** every variant runs forward + backward on PyTorch's `meta` device, i.e. no tensor is
  pinned to the CPU inside the model. On a physical GPU (RTX 4090, PyTorch 2.5.1+cu124): see the status note in
  [`GPU_RUNS.md`](GPU_RUNS.md).
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


## Feasibility prototype (`code/`)
The first study of the idea, before the Time-Series-Library integration: a linear prototype, 69 runs.

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


### Results (69 runs)
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


### Reproduce the prototype
```bash
bash scripts/download_data.sh                 # into the fixed dataset/ directory
python code/proto.py --data PEMS08 --pred_len 96 --variant dltdyn --out results/my_runs.jsonl
bash scripts/run_grid.sh                      # the full grid (CPU; ~1–2 h with 4 workers)
python code/agg_md.py results/runs_reproduced.jsonl
python code/pems_lag.py                       # lag premise check
```
The architecture figure and its generator moved to the `DLTpaper` repository (`figures/`, `figures/src/`).

## Next steps
A validation-driven gate or regulariser for weakly coupled data; k and R in the search space; the GPU experiments
of [`GPU_RUNS.md`](GPU_RUNS.md).
