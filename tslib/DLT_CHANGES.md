# DLT changes to this Time-Series-Library snapshot

Upstream: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library),
commit `4e938a1767106324dd753b2a44832bf870a0252e` (2026-04-18), MIT License (see `LICENSE`).
Everything else in this directory is unchanged upstream code.

| file | change |
|---|---|
| `models/DLT.py` | **new** – DLT model (11 kernel variants incl. the MillGNN-style `topk` / `mill` / `milldyn`), neighbour and lag-set selection, `prepare()` hook, optional branch gates (`--dlt_gate none|moe|static`); two identical computation paths for `dlt`/`dltdyn` (`--dlt_impl kernel` = default, less memory; `basis`) |
| `data_provider/data_loader.py` | **new** `Dataset_PEMS` (PEMS03/04/07/08 `.npz`, 6:2:2 split, rebuilt 5-min time marks) |
| `data_provider/data_factory.py` | register `--data PEMS` |
| `exp/exp_long_term_forecasting.py` | call `model.prepare(train_data)` before training if the model defines it (DataParallel-safe); optional `--no_epoch_test`, `--no_save_preds`, `--result_jsonl` (one JSON line per run with metrics, epochs, best epoch and timing) |
| `exp/exp_basic.py` | fall back to CPU when CUDA / MPS is requested but not available (instead of crashing) |
| `utils/timefeatures.py` | pandas ≥ 3 compatibility: map TSLib `--freq` codes (`t`, `h`, `s`, `d`, `b`, `w`, `m`, `q`, `a`) to the new pandas aliases (`min`, `h`, `s`, `D`, `B`, `W`, `ME`, `QE`, `YE`); falls back to the original string on older pandas |
| `run.py` | `--dlt_variant --dlt_k --dlt_L --dlt_R --dlt_dyn_dim --dlt_ccf_len --dlt_impl --dlt_topk --dlt_gate --dlt_gate_init`; `--fix_seed` (default 2021 = upstream behaviour); DLT experiment names get a `_{variant}_k{k}_L{L}_R{R}` suffix (plus `_K{topk}`, `_g{gate}`, `_seed{n}` when used) so configurations do not overwrite each other |
| `scripts/long_term_forecast/DLT_script/` | **new** run scripts (ETTh1/2, ETTm1/2, Weather, ECL, Traffic, Exchange, PEMS08, kernel ablation) |
| `requirements_dlt.txt` | **new** minimal dependency list |

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
