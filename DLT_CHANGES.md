# DLT changes to this Time-Series-Library snapshot

Upstream: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library),
commit `4e938a1767106324dd753b2a44832bf870a0252e` (2026-04-18), MIT License (see `LICENSE`).
Since 2026-11-10 the snapshot is the repository root (it used to live in `tslib/`). Everything not listed
below is unchanged upstream code.

Removed upstream files: `README.md`, `README_zh.md`, `CONTRIBUTING.md`, `pic/` and `tutorial/` (documentation and
images of upstream; see the upstream repository). `requirements.txt` of upstream is kept as
`requirements_tslib_full.txt`, and the `Dockerfile` installs that file.

| file | change |
|---|---|
| `models/DLT.py` | **new** – DLT model (11 kernel variants incl. the MillGNN-style `topk` / `mill` / `milldyn`), neighbour and lag-set selection, `prepare()` hook, optional branch gates (`--dlt_gate none|moe|static`); two identical computation paths for `dlt`/`dltdyn` (`--dlt_impl kernel` = default, less memory; `basis`) |
| `data_provider/data_loader.py` | **new** `Dataset_PEMS` (PEMS03/04/07/08 `.npz`, 6:2:2 split, rebuilt 5-min time marks) |
| `data_provider/data_factory.py` | register `--data PEMS` |
| `exp/exp_long_term_forecasting.py` | call `model.prepare(train_data)` before training if the model defines it (DataParallel-safe); optional `--no_epoch_test`, `--no_save_preds` (also streams the test metrics batch by batch instead of holding the whole test split in memory, which needs tens of GB for Electricity / Traffic at horizon 720; same values up to float32 rounding), `--result_jsonl` (one JSON line per run with metrics incl. TimeMixer's clipped MAPE and the masked MAPE over |true| > 1e-3 — a zero flow is ~1e-5, not 0, after the float32 scale / inverse round trip; runs before commit f694112 (incl. all P0 and the frozen P1/P2 runs at e232fba) used `true != 0` and their masked MAPE is not meaningful on PEMS, epochs, best epoch and timing; since 2026-11-13 also the PyTorch version and the device, `torch` and `device`; since 2026-11-15 the number of NaN / inf forecasts, `test_nonfinite`, with a warning in the log); `--loss MAE` (L1 training loss) and `--vali_inverse` (validation MAE on the original scale) for the TimeMixer / MillGNN PEMS protocol together with `--inverse`; models may add an auxiliary loss via `extra_loss(x, pred, true)`; per-group learning rates for `--dlt_lr` |
| `models/DLTPlug.py` | **new** – the DLT cross-variable branch as a plug-in: `Backbone(X) + sigma * W_c z` for any TSLib forecaster (`--plug_base`); `--dlt_variant ci` = backbone alone through the same code path; `--dlt_lr` gives the branch its own learning rate |
| `models/LagTS.py` | **new** – re-implementation of LagTS (ICASSP 2025; no official code) incl. its lag relation loss (`extra_loss`), `--lagts_lambda`, `--lagts_rel_dim` |
| `scripts/train_dltdyn.sh` | **new** quick start: one GPU, one dataset, stand-alone DLT-dyn |
| `models/SMamba.py`, `layers/MambaPy.py` | **new** – the S-Mamba baseline (Wang et al., Neurocomputing 2025), re-implemented after the official code: bidirectional Mamba over variate tokens; `mamba_ssm.Mamba` if it imports (from the environment, else from the source checkouts `../mamba` and `../causal-conv1d`; the block is imported without `mamba_ssm/__init__.py`, so `transformers` is not needed; a mamba_ssm without its CUDA kernel `selective_scan_cuda`, as a source install of mamba_ssm >= 2.3 without `MAMBA_KEEP_CUDA_BUILD=TRUE`, counts as not importable) and the run is on a GPU, else a pure-PyTorch block with the same parameters, initialisation and outputs (`--mamba_impl auto|ssm|torch`, `--d_state`; `scripts/gpu/check_mamba.py` checks the installation). The unused modules of the official encoder layer are left out |
| `scripts/long_term_forecast/*_script/SMamba*.sh` | **new** – the hyper-parameters of the official S-Mamba scripts for Electricity, Traffic, Weather, Exchange and ETTh1/h2/m1/m2 in the format of this library |
| `run.py` | `--d_state`, `--mamba_impl` (S-Mamba); `--no_time_marks`: the model gets no time-mark features (`x_mark = None`), as the iTransformer / S-Mamba code does on PEMS and Solar (records: `time_marks`); `--test_batch_size` (that code tests with batch 1); `--pems_split itransformer`: PEMS validation / test windows start inside their split, as in its PEMS loader (default `tslib`: unchanged) |
| `models/iTransformer.py` | honours `--use_norm` (default 1: unchanged); the official PEMS settings of iTransformer use `--use_norm 0` |
| `data_provider/data_loader.py`, `data_factory.py` | **new** `Dataset_Solar` (`--data Solar`, `solar_AL.txt`): split 7:1:2 and scaling as in iTransformer / S-Mamba, time marks rebuilt from 2006-01-01 at 10 minutes (as `Dataset_PEMS` does) |
| `scripts/download_data.sh`, `scripts/check_data.py` | all forecasting datasets (ETT, Exchange, Weather, ECL, Traffic, ILI, PEMS03/04/07/08 with road networks, Solar) from their sources, with fallbacks and `--mirror`; check of every file against its published size |
| `scripts/paths.py` | **new** – the fixed locations used by every script: data (`dataset/`, or `../data` next to the repository), work (`work/`), Mamba sources (`../mamba`, `../causal-conv1d`) |
| `exp/exp_basic.py` | fall back to CPU when CUDA / MPS is requested but not available (instead of crashing); keep a `CUDA_VISIBLE_DEVICES` mask set by the caller instead of overwriting it with `--gpu` (one job per GPU in `scripts/gpu/run_gpu_queue.sh`) |
| `utils/timefeatures.py` | pandas ≥ 3 compatibility: map TSLib `--freq` codes (`t`, `h`, `s`, `d`, `b`, `w`, `m`, `q`, `a`) to the new pandas aliases (`min`, `h`, `s`, `D`, `B`, `W`, `ME`, `QE`, `YE`); falls back to the original string on older pandas |
| `run.py` | `--dlt_variant --dlt_k --dlt_L --dlt_R --dlt_dyn_dim --dlt_ccf_len --dlt_impl --dlt_topk --dlt_gate --dlt_gate_init`; `--fix_seed` (default 2021 = upstream behaviour); DLT experiment names get a `_{variant}_k{k}_L{L}_R{R}` suffix (plus `_K{topk}`, `_g{gate}`, `_seed{n}` when used) so configurations do not overwrite each other |
| `scripts/long_term_forecast/DLT_script/` | **new** run scripts (ETTh1/2, ETTm1/2, Weather, ECL, Traffic, Exchange, PEMS08, kernel ablation) |
| `utils/tools.py` | `adjust_learning_rate` keeps per-group learning-rate ratios (`lr_scale`; default 1 = upstream behaviour) |
| `run.py` (structure) | the argument parser is built by `get_parser()` (module level) so that scripts can construct identical configurations |
| `requirements.txt` | **replaced** by the dependency list of DLT and the classic models (upstream list: `requirements_tslib_full.txt`) |

## Device support
* `models/DLT.py` creates no tensors on a fixed device inside `forward`; neighbour indices / lags are
  registered buffers. It therefore runs unchanged on CPU, a single CUDA GPU, several GPUs
  (`--use_multi_gpu --devices 0,1`, DataParallel) and Apple MPS, and supports `--use_amp`.
* Neighbour selection (`select_neighbours`) runs once per training run on the model's device
  (float64; on MPS it runs on the CPU because MPS has no float64).
* GPU is used by default (`--use_gpu` defaults to on in TSLib); pass `--no_use_gpu` to force the CPU.
  Without CUDA the run now falls back to the CPU automatically.

## Kernel variants and gates (`--dlt_variant`, `--dlt_gate`)
| variant | kernel K_ij(tau) on lags 0..L | reference point |
|---|---|---|
| `ci` | none (channel-independent) | RLinear |
| `scalar` / `scalardyn` | a_ij at tau = 0 (static / input-adaptive) | iTransformer, VCformer-style mixing |
| `shift` | a_ij at the CCF peak lag d_ij | LIFT |
| `tap3` | free on tau = 0, 1, 2 | CrossLinear |
| `full` | free on tau = 0..L | unconstrained |
| `topk` | free weights on the K lags with the largest training-split \|CCF\| of the pair | MillGNN lag sets |
| `mill` | a_ij ReLU(exp(-be_ij tau) - (1 - att_ij) exp(-bi_ij tau)) on the same K lags | MillGNN decay-aware weights |
| `milldyn` | `mill` with att_ij(X) = softmax over the k sources of q_i . k_j / sqrt(d) | MillGNN (dynamic) |
| `dlt` / `dltdyn` | sum_r alpha_ijr phi_r(tau), shared learnable basis (static / input-adaptive) | proposed |

The MillGNN-style variants transfer MillGNN's lag sets (Eq. 5) and decay-aware weights (Eq. 9) to the
time-step resolution of this operator; they are time-invariant (no patch-position embedding), and a signed
per-pair amplitude a_ij is added because MillGNN's ReLU weights are non-negative.

`--dlt_gate`: `none` (default) y = y_self + y_cross; `moe` y = g_0 y_self + g_1 y_cross with
g = softmax(W_g x_i) per variable (MoVE's gate); `static` y = y_self + sigmoid(theta_i) y_cross with
theta_i = `--dlt_gate_init` (default -3, i.e. the cross branch starts at weight 0.047).

Verification: `scripts/p0/test_variants.py` (repository root) checks every variant against an explicit loop,
the lag sets against a reference CCF loop, and that the original 8 variants are unchanged (same parameters
for the same seed, identical outputs) relative to commit `af398bd`.

## Neighbour selection and computation paths
`--dlt_neighbours ccf|random|all`: CCF top-k (default), k random sources (drawn with `--fix_seed`), or all other
variables (k = C - 1). `--dlt_impl dense` evaluates dlt / dltdyn with a dense C x C x R coefficient tensor, which is
the cheapest path when k is close to C - 1. Verification: `scripts/p0/test_p1p2.py` (dense path, neighbour sets,
DLTPlug, LagTS loss against explicit loops).

## Reproducing P1 / P2
See `scripts/exps.py` (job lists per suite, `--device cpu|gpu`), `scripts/run_queue.sh`, `scripts/p1/hpsearch.py`,
`scripts/p2/efficiency.py`, `scripts/p2/kernels.py` and the aggregation scripts in `scripts/p1`, `scripts/p2`
(`scripts/p1/plugin_table.py` for the plug-in study with all backbones). `scripts/smoke_jobs.py <list>` builds every
configuration of a job list and trains it for one step before a long queue starts.
