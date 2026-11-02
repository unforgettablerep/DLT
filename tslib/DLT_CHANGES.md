# DLT changes to this Time-Series-Library snapshot

Upstream: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library),
commit `4e938a1767106324dd753b2a44832bf870a0252e` (2026-04-18), MIT License (see `LICENSE`).
Everything else in this directory is unchanged upstream code.

| file | change |
|---|---|
| `models/DLT.py` | **new** – DLT model (8 kernel variants), neighbour selection, `prepare()` hook; two identical computation paths for `dlt`/`dltdyn` (`--dlt_impl kernel` = default, less memory; `basis`) |
| `data_provider/data_loader.py` | **new** `Dataset_PEMS` (PEMS03/04/07/08 `.npz`, 6:2:2 split, rebuilt 5-min time marks) |
| `data_provider/data_factory.py` | register `--data PEMS` |
| `exp/exp_long_term_forecasting.py` | call `model.prepare(train_data)` before training if the model defines it (DataParallel-safe) |
| `exp/exp_basic.py` | fall back to CPU when CUDA / MPS is requested but not available (instead of crashing) |
| `utils/timefeatures.py` | pandas ≥ 3 compatibility: map TSLib `--freq` codes (`t`, `h`, `s`, `d`, `b`, `w`, `m`, `q`, `a`) to the new pandas aliases (`min`, `h`, `s`, `D`, `B`, `W`, `ME`, `QE`, `YE`); falls back to the original string on older pandas |
| `run.py` | `--dlt_variant --dlt_k --dlt_L --dlt_R --dlt_dyn_dim --dlt_ccf_len --dlt_impl`; DLT experiment names get a `_{variant}_k{k}_L{L}_R{R}` suffix so variants do not overwrite each other |
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
