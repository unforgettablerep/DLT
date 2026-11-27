# AGENTS.md

## Language

- Conversation with the user (explanations, progress, reports): Simplified Chinese
- Everything in the repository (Python, shell, README, AGENTS.md, `output/**/*.md`, plots): English

## Environment

Reference GPU machine: one RTX 4090 (24 GB), host CUDA 13.0, conda env `dlt` (Python 3.12), PyTorch 2.10.0+cu130,
`mamba_ssm` 2.3.2.post1 / `causal_conv1d` 1.7.0 wheels for S-Mamba (README.md, Installation). Development and every
CPU result: 4-core CPU, PyTorch 2.5.1 / 2.15 CPU builds. `environment.yml` pins only Python; PyTorch is installed
separately; the remaining dependencies are in `pyproject.toml`. Tasks: `poetry run poe <task>` (`poe` lists them).

## Data

Datasets are git-ignored under `./data/` (Time-Series-Library layout or flat), else `../data` next to the
repository (`src/utils/paths.py`, the only place that knows locations; `python -m src.utils.paths data|temp|log|output|root <file>|file <file>|jobs <list>`).
`bash scripts/download_data.sh` fetches all files. Tracked: `src/`, `scripts/`, `tests/`, `assets/`, `output/`.

| Dataset(s) | `--data` | Dataset class | enc_in | pred_lens |
| --- | --- | --- | --- | --- |
| ETTh1, ETTh2 | `ETTh1`, `ETTh2` | `Dataset_ETT_hour` | 7 | 96,192,336,720 |
| ETTm1, ETTm2 | `ETTm1`, `ETTm2` | `Dataset_ETT_minute` | 7 | 96,192,336,720 |
| Exchange / Weather / ECL / Traffic | `custom` | `Dataset_Custom` | 8 / 21 / 321 / 862 | 96,192,336,720 |
| Solar | `Solar` | `Dataset_Solar` | 137 | 96,192,336,720 |
| PEMS03/04/07/08 | `PEMS` | `Dataset_PEMS` | 358/307/883/170 | 12,24,48,96 |

- `seq_len=96` everywhere; `label_len` 48, 0 for DLT on PEMS (the baselines use 48).
- One protocol for every run and model, that of the official iTransformer / S-Mamba loaders (`protocol()` in
  `src/configs/dlt_profiles.py`, emitted into every job by `scripts/utils/jobs.py` and the defaults of
  `src/configs/common.py`): `--freq h --drop_last --test_batch_size 1`, Solar and PEMS with `--no_time_marks`
  (`x_mark = None`); `Dataset_PEMS` starts the validation / test windows inside their split. No other protocol.
- `DATASET_META`, `canonical_dataset` (aliases such as `electricity` → `ECL`), `resolve`, `resolve_meta`: the
  single source of dataset metadata. Do not copy these tables into scripts.

## Quick Start

```bash
python -u scripts/run.py --is_training 1 --model DLT --des quickstart \
    $(python -m src.configs.dlt_cli Weather 96 --with-meta)
```

`scripts/run.py` is the only training entry point. Required: `--model` (aliases `SMamba`, `S-Mamba` → `S_Mamba`).
Modes: `--is_training 1` trains and tests (the checkpoint of the best validation loss is kept, `--keep_checkpoint
0` deletes it after the test); `--is_training 2` tests the latest checkpoint, or `--checkpoint N`. `--seed`
(alias `--fix_seed`, default 2021) seeds every RNG and the random neighbour selection; results and checkpoints are
seed-dependent. `--save_tmp 0` (alias `--no_save_preds`) streams the test metrics instead of keeping all
predictions (needed for Electricity / Traffic at long horizons). There is no per-epoch test pass unless `--epoch_test`
is given; that pass creates a data-loader iterator, which draws from the RNG, so such a run differs from the recorded
ones (all of which ran without it).
Defaults of the shared arguments are those of Time-Series-Library (`d_model` 512, `d_ff` 2048, ...), so that the
official baseline settings reproduce.

## Centralized Configuration

- `src/configs/dlt_profiles.py`: `CONSTANTS` (the paper's DLT-dyn recipe: `dltdyn`, k 8, L 48, R 4, `basis`, lr
  3e-3, batch 32, 10 epochs, patience 3), `resolve(ds, H, profile)` with the profiles `default` and `efficiency`
  (S-Mamba Fig. 5 setting: batch 16, its epochs, the recipes selected in `output/efficiency/eff_tune.md`).
  Emitter: `python -m src.configs.dlt_cli <ds> <H> [--with-meta] [--profile efficiency]`.
- `src/configs/baseline_profiles.py`: the six baselines (`BASELINES`), their settings read from the copies of the
  official scripts in `scripts/benchmark/official/` (with documented fallbacks), `protocol_args`, `runtime_args`
  (`--tide_checkpoint` on data with more than 100 variables). Emitter: `python -m src.configs.baseline_cli <model> <ds> <H> --with-meta`;
  `python -m src.configs.baseline_profiles [models]` prints every setting with its source.
- `src/ablation/registry.py`: the P0 configurations (kernel variant, gate), their labels and further ablation groups.
- `scripts/multivariate_forecasting/make_scripts.py` writes the 91 per-dataset scripts from these profiles
  (`--check` reports stale ones; a unit test runs it). Never edit the generated scripts by hand.

## Job Lists, Queues and Schedulers

- Job generators: `scripts/<category>/make_jobs.py <suite> [--device cpu|gpu] [--seeds ...] [--datasets ...]
  [--models ...] [--dlt "<args>"] [--list <name>]` (and `scripts/hyperparam/hpsearch.py make|tuned`) write
  `temp/jobs/jobs_<list>.txt` via `scripts/utils/jobs.py` (`JobList`): one `scripts/run.py` command per run with
  `--des <job name> --seed <s> --result_file output/<category>/<suite>.tsv`, log in `log/<category>/<suite>/`.
- `scripts/utils/run_queue.sh <list> [n]` (CPU) and `run_gpu_queue.sh <list> [gpu ids]` (one worker per id,
  `CUDA_VISIBLE_DEVICES` per job) run the lines that `run_missing.py` reports as missing: no row with the job's
  `--des` and `--seed` in its `--result_file` from the job's device (`--no_use_gpu`: CPU, else GPU; a row without a
  device counts for both). `smoke_jobs.py <list>` trains every configuration for one step first.
- `scripts/multivariate_forecasting/run_all_forecasts.sh [--model M] [--gpu 0,1] [--para N]` dispatches the
  per-dataset scripts with least-loaded GPU placement; log `log/multivariate_forecasting/scheduler.log`, lock
  `temp/forecasts/run.lock`. `scripts/utils/gpu_queue.sh` offers round-robin GPU slots to shell runners.
- `scripts/batch/make_gpu_jobs.sh` writes every list that needs a GPU; `run_cpu_campaign.sh [stage]` runs the CPU
  part of the study stage by stage.
- GPU study: tables use GPU records only (`--device gpu`; default of `baseline_check.py`). Plug-in lists run the
  + DLT-dyn arms only (`scripts/plugin/make_jobs.py --skip_alone`); the backbone alone is the baseline-check run
  (`src/utils/baseline_runs.py`, tables with `--alone baselines`). Each baseline is reproduced once with its official
  settings (no learning-rate check, one seed); `baseline_check.py` compares it with the published result.

## Results and Output

`src/utils/result_io.py` owns the TSV result stores: one row per run, columns `METADATA_COLUMNS` (dataset, model,
lengths, mse, mae, status, checkpoint, timestamp, des, seed) + `CONFIG_COLUMNS` (CLI flag names: `learning_rate`,
`dlt_k`, `plug_base`, ...) + `RUN_COLUMNS` (rmse, mape variants, best_vali, epochs, timings, n_params, device, torch,
...). Rows are appended under an `fcntl` lock; a row with new keys extends the header. Use `write_test_result`,
`read_results` (typed: bool, int, float, JSON lists), `latest` (last row per `(des, seed)`), `compact_results`,
`migrate_record` (legacy JSONL records: `lr` → `learning_rate`, `variant` → `dlt_variant`, `k` → `dlt_k`, ...)
instead of hand-written TSV I/O (`python -m src.utils.result_io sort|migrate|show`). `status` is `ok` or `diverged`
(non-finite metric). `scripts/utils/result_common.sh` provides `result_done <file.tsv> key=value ...` for shell
resume checks.

| Target | File |
| --- | --- |
| `--result_file f` or env `RESULT_FILE=f` | `f` |
| `ABLATION_OUTPUT_DIR=dir` | `{parent(dir)}/{basename(dir)}.tsv` |
| `BENCHMARK_OUTPUT_DIR=dir` | `{dir}/{dataset}.tsv` |
| default | `./output/long_term_forecast/{dataset}.tsv` |

`output/<category>/`: stores (`<suite>.tsv`), tables (`*.md`), small JSON / JSONL measurements; conclusions in
`output/README.md`. `temp/checkpoints/{dataset}/`, `temp/results/`, `temp/test_results/` (predictions with
`--save_tmp 1`), `temp/jobs/`, `log/` are scratch (`bash scripts/clean.sh [dataset]`).

## Architecture: DLT

| Component | File |
| --- | --- |
| `VARIANTS`, `GATES`, `NEIGHBOUR_MODES`, `IMPLS`; CCF neighbour and lag-set selection, lag basis, pair convolutions | `src/layers/DLT_layers.py` |
| Stand-alone model, `prepare(train_data)` hook (neighbours from the training split), gates; guards `--dlt_train`, `--dlt_select variable|significant`, `--dlt_lr` (self head + correction via `parts()`, as the plug-in); `--dlt_anchor`, `--dlt_cross_input` | `src/model/DLT.py` |
| Plug-in: backbone + branch, `--dlt_fusion`, `--dlt_train`, `--dlt_select`, `--dlt_lr` | `src/model/DLTPlug.py` |
| S-Mamba (`mamba_ssm` block or `src/layers/MambaPy.py`), RLinear (official code) | `src/model/S_Mamba.py`, `src/model/RLinear.py` |
| Experiment loop (calls `prepare`, per-group learning rates, streaming test metrics, result rows) | `src/experiments/exp_long_term_forecasting.py` |

| `--dlt_variant` | kernel K_ij(τ) on lags 0..L | reference point |
| --- | --- | --- |
| `ci` | none (channel-independent) | RLinear |
| `scalar` / `scalardyn` | a_ij at τ = 0 (static / input-adaptive) | iTransformer-style mixing |
| `shift` | a_ij at the CCF peak lag | LIFT |
| `tap3` | free on τ = 0, 1, 2 | CrossLinear |
| `full` | free on τ = 0..L | unconstrained |
| `topk` / `mill` / `milldyn` | the K lags of largest training \|CCF\|: free weights / decay-aware weights / with attention | MillGNN |
| `dlt` / `dltdyn` | Σ_r α_ijr φ_r(τ), shared learnable basis (static / input-adaptive) | proposed |

`--dlt_gate`: `none` y = y_self + y_cross; `moe` softmax gate per variable (MoVE); `static` y_self + σ(θ_i) y_cross
with θ_i = `--dlt_gate_init` (−3: weight 0.047). The model creates no tensor on a fixed device (CPU, CUDA,
DataParallel, MPS, AMP); neighbour selection runs once per run in float64 on the model's device (CPU on MPS).

New model registration:

1. Add `src/model/YourModel.py` with `class Model(nn.Module)`; a model that needs the training split defines
   `prepare(train_data)`, one with an auxiliary loss `extra_loss(x, pred, true)`.
2. Register it in `src/experiments/exp_basic.py` (`model_dict`) and in the lazy loader `src/model/__init__.py`.
3. Add `src/configs/custom_YourModel.py` and register the parser in `src/configs/model_parser_dict.py`.
4. For a baseline: its official script under `scripts/benchmark/official/`, the reader in `baseline_profiles.py`,
   then `python scripts/multivariate_forecasting/make_scripts.py`.

Checkpoints: `temp/checkpoints/{dataset}/{dataset}_{model}_sl{seq_len}_pl{pred_len}_{des}_s{seed}_{seq}.pth`
(`src/utils/checkpoint.py`); `seq` auto-increments, and `checkpoints_{dataset}.jsonl` logs the configuration of each
file, so analyses find checkpoints by configuration (`find_checkpoints(root, dataset, **config)`).

## Upstream Code

Derived from Time-Series-Library (commit `4e938a1`, MIT): the data loaders, the experiment loop, `Embed`,
`SelfAttention_Family`, `Transformer_EncDec`, `Autoformer_EncDec`, DLinear, PatchTST, TiDE, iTransformer and
`src/utils/{metrics,tools,timefeatures,masking}.py`. Changes, all off by default unless a recipe sets them:
`Dataset_PEMS` and `Dataset_Solar` (5- / 10-minute time marks rebuilt, the splits of the iTransformer loader);
`--no_time_marks`, `--test_batch_size` (default 1), `--drop_last` (default on); `--loss MAE` (unused by the study); `prepare()` and `extra_loss()` hooks; per-group learning rates kept by `adjust_learning_rate`
(`lr_scale`); streaming test metrics (with the MSE of every forecast step on request, `--step_metrics`); masked and clipped MAPE; iTransformer honours `--use_norm`; TiDE
`--tide_checkpoint` (recomputation, identical results); CPU fallback when CUDA / MPS is missing and a
`CUDA_VISIBLE_DEVICES` mask set by the caller is kept; pandas ≥ 3 frequency aliases in `timefeatures.py`. The other
models and tasks of the library were removed.

## Verification

`python -m unittest discover -s tests -v` (`poe test`):
- `tests/test_variants.py`: every kernel variant against an explicit loop, lag sets against a reference CCF loop, the
  original eight variants unchanged against git commit `af398bd`, gates, checkpoint round trip;
- `tests/test_plugin_protocol.py`: dense path, neighbour modes, DLTPlug and its fusions against explicit sums,
  RLinear, TiDE checkpointing, the official settings and protocol of the six baselines, S-Mamba against the
  reference Mamba block (`tests/reference/`);
- `tests/test_framework.py`: profiles and emitters, CLI parsing on all datasets, the result store, checkpoints, resume
  logic, generated scripts.

Equivalence with the code before the refactor (tag `pre-refactor`) was checked on model internals, end-to-end
training, the arguments of 1,325 jobs and every analysis output.

## Code Style

- Spaces, 4-space indentation, max line 180 (`.editorconfig`). `src/`: double quotes; `getattr(configs, ...)` for
  optional parameters. Ported analysis scripts keep their single-quote style; match the surrounding file.
- Every script starts with a docstring (or comment) that says how to run it, what it reads and what it writes.
- No auto-formatter. Vendored files keep their headers (`scripts/benchmark/official/`, `tests/reference/`).

## Plotting Colors

Source figure colours from `scripts/color.py` (`categorical` in fixed order, `neutral_dict` for ink, grid and muted
marks); text uses ink colours, never a series colour. The paper's figures are drawn in the `DLTpaper` repository.
