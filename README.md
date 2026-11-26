# DLT: Distributed-Lag Transfer for Multivariate Time Series Forecasting

Cross-variable effects in multivariate series are often **lagged and spread over several lags** (e.g. upstream →
downstream traffic sensors). Channel-dependent models usually mix whole series with one scalar per pair
(iTransformer-style), shift a source by one estimated lag (LIFT-style) or use very short static kernels
(CrossLinear-style). **DLT** gives every target–source pair its own multi-lag kernel built from a small bank of shared
basis kernels, at a cost linear in the number of variables:

$$z_i(t)=\sum_{j\in N(i)}\sum_{\tau=0}^{L}K_{ij}(\tau)\,\tilde x_j(t-\tau),\qquad K_{ij}(\tau)=\sum_{r=1}^{R}\alpha_{ijr}\,\phi_r(\tau)$$

**DLT-dyn** makes the coefficients input-adaptive, α_ijr(X) = α_ijr · (1 + tanh(MLP([s_i ‖ s_j]))). The forecast is
y_i = W_self x̃_i + W_cross z_i on RevIN-normalised inputs; the neighbours N(i) are the k sources with the largest
|CCF| on the training split. `--model DLTPlug` adds the same branch to another forecaster.

The paper (LaTeX, figures, tables) lives in the `DLTpaper` repository; this repository holds the code, the run
records and the result tables.

## Installation

### Environment Requirements

- Python 3.12 (3.10–3.13 work)
- PyTorch 2.10.0+cu130 on the reference GPU machine (one RTX 4090, host CUDA 13.0); on the CPU any PyTorch ≥ 2.5
  (tested with 2.5.1 and 2.15, identical DLT-dyn results)

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yml && conda activate dlt
```

### Step 2: Install PyTorch

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130 &&
pip install poetry
```

### Step 3: Install Additional Dependencies

```bash
poetry install                      # or: pip install numpy pandas scipy scikit-learn matplotlib
```

S-Mamba uses the CUDA kernels of `mamba_ssm` when they import, else a pure-PyTorch block (same model, one Python step
per variate token, slow on Electricity and Traffic). For CUDA 13 + torch 2.10 + Python 3.12 the prebuilt pair is
installed with `--no-deps` (S-Mamba needs neither `transformers` nor the other dependencies of `mamba_ssm`):

```bash
pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Without these wheels S-Mamba also finds the source checkouts `../mamba` and `../causal-conv1d` next to the
repository.

### Verify Setup

```bash
poetry run poe cuda             # PyTorch, CUDA and the GPU
poetry run poe check-mamba      # must end with "S-Mamba will use mamba_ssm ... Nothing to do." before S-Mamba runs
poetry run poe check-gpu        # GPU against CPU for DLT-dyn; must end with "all checks passed"
python scripts/utils/check_gpu.py --all     # the ten representative configurations (DLT, plug-in, S-Mamba)
poetry run poe test             # unit tests (CPU)
```

## Data

`bash scripts/download_data.sh [groups] [--mirror]` fetches every dataset into `./data/` and checks each file
against its published size (`python scripts/utils/check_data.py` repeats the check; `--mirror` takes hf-mirror.com
and jsDelivr first where huggingface.co or GitHub are blocked):

- ETT (ETTh1/h2/m1/m2), Exchange, Weather, Electricity and Traffic: the Time-Series-Library copy on
  [Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library);
- PEMS03/04/07/08 (`PEMS0x.npz`, flow in channel 0) with their road networks (`PEMS0x_edges.csv`): the repository
  of the dataset's authors, else a Hugging Face mirror;
- Solar-Energy (`Solar/solar_AL.txt`): the LSTNet data repository.

Files can sit in the Time-Series-Library layout (`data/ETT-small/ETTh1.csv`, `data/weather/weather.csv`,
`data/PEMS/PEMS08.npz`, `data/Solar/solar_AL.txt`, …) or directly in `data/`. When `./data` (or the older
`./dataset`) does not exist, `../data` next to the repository is used (`src/utils/paths.py`). Data, `temp/` and
`log/` are ignored by git.

## Quick Start

Stand-alone DLT-dyn on Weather, horizon 96, seed 2021 (about 25 s per epoch on one CPU thread):

```bash
python -u scripts/run.py --is_training 1 --model DLT --des quickstart --seed 2021 \
    $(python -m src.configs.dlt_cli Weather 96 --with-meta)
```

Append `--no_use_gpu --num_workers 0` for the CPU. The run trains, tests the best checkpoint and appends one row to
`output/long_term_forecast/Weather.tsv`; the checkpoint goes to `temp/checkpoints/Weather/`. Every run uses the
protocol of the official iTransformer / S-Mamba loaders (`protocol()` in `src/configs/dlt_profiles.py`). On a GPU
expect the CPU result to about three decimals (cuDNN and TensorFloat-32 are not bit-identical to the CPU). As in every
recorded run there is no per-epoch test pass;
`--epoch_test` adds it, and because that pass draws from the random number generator, such a run differs slightly.
`--is_training 2` tests the latest checkpoint again (`--checkpoint N` a given one).

The recipe of every dataset lives in `src/configs/dlt_profiles.py` and is emitted by `python -m
src.configs.dlt_cli <dataset> <horizon> [--with-meta] [--profile efficiency]`; the six baselines are emitted the same
way from their official scripts by `python -m src.configs.baseline_cli <model> <dataset> <horizon> --with-meta`. Do
not copy these settings into commands. Per-dataset scripts for all horizons and seeds, generated from the same
profiles:

```bash
bash scripts/multivariate_forecasting/Weather/DLT.sh                      # DLT-dyn, horizons 96-720, seeds 2021-2023
SEEDS="2021" bash scripts/multivariate_forecasting/PEMS/iTransformer_PEMS08.sh
bash scripts/multivariate_forecasting/run_all_forecasts.sh --model DLT --gpu 0,1 --para 2   # all 13 datasets
```

## Architecture

| Component | File |
| --- | --- |
| Lag kernels (11 variants), neighbour and lag-set selection, basis, computation paths | `src/layers/DLT_layers.py` |
| Stand-alone model (`--model DLT`): RevIN, self head, cross branch, gates | `src/model/DLT.py` |
| Plug-in (`--model DLTPlug --plug_base <model>`): the branch added to a backbone | `src/model/DLTPlug.py` |
| Baselines: S-Mamba, iTransformer, PatchTST, RLinear, DLinear, TiDE | `src/model/*.py` |
| CLI parsers | `src/configs/common.py`, `custom_<model>.py`, `model_parser_dict.py` |
| Recipes and emitters | `src/configs/dlt_profiles.py`, `dlt_cli.py`, `baseline_profiles.py`, `baseline_cli.py` |
| Ablation registry (P0 kernels and gates) | `src/ablation/registry.py` |

`--dlt_impl kernel | basis | dense` are three computation paths with identical forecasts; `basis` is the path of all
reported DLT-dyn runs (the recipes set it), `dense` is the cheapest when k is close to C − 1.

## Supported Datasets

| Dataset | `--data` | `--data_path` | `enc_in` | `seq_len` | pred_lens |
| --- | --- | --- | --- | --- | --- |
| ETTh1, ETTh2 | ETTh1, ETTh2 | ETTh1.csv, ETTh2.csv | 7 | 96 | 96, 192, 336, 720 |
| ETTm1, ETTm2 | ETTm1, ETTm2 | ETTm1.csv, ETTm2.csv | 7 | 96 | 96, 192, 336, 720 |
| Exchange | custom | exchange_rate.csv | 8 | 96 | 96, 192, 336, 720 |
| Weather | custom | weather.csv | 21 | 96 | 96, 192, 336, 720 |
| ECL | custom | electricity.csv | 321 | 96 | 96, 192, 336, 720 |
| Traffic | custom | traffic.csv | 862 | 96 | 96, 192, 336, 720 |
| Solar | Solar | solar_AL.txt | 137 | 96 | 96, 192, 336, 720 |
| PEMS03 / 04 / 07 / 08 | PEMS | PEMS0x.npz | 358 / 307 / 883 / 170 | 96 | 12, 24, 48, 96 |

Every run, of every model, follows the protocol of the official iTransformer / S-Mamba loaders: four hourly time
features, the last incomplete training / validation batch dropped, test batch 1; Solar and PEMS without time marks,
PEMS with the validation and test windows inside their own split.

## Key Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--dlt_variant` | `dltdyn` | Kernel: `dltdyn`, `dlt`, `ci` (no cross branch), `scalar`, `scalardyn`, `shift`, `tap3`, `full`, `topk`, `mill`, `milldyn` |
| `--dlt_k` | 8 | Source neighbours per variable |
| `--dlt_L` | 48 | Maximum lag of the kernels |
| `--dlt_R` | 4 | Shared basis kernels |
| `--dlt_dyn_dim` | 16 | Summary dimension of the input-adaptive coefficients |
| `--dlt_impl` | `kernel` | Computation path (`kernel`, `basis`, `dense`); the recipes use `basis` |
| `--dlt_neighbours` | `ccf` | Neighbour selection: CCF top-k, `random`, `all` |
| `--dlt_gate` | `none` | Branch combination: `none`, `moe` (MoVE-style), `static` (sigmoid, init `--dlt_gate_init` −3) |
| `--dlt_topk` | 5 | Lags per pair of `topk` / `mill` / `milldyn` |
| `--plug_base` | `DLinear` | `DLTPlug`: backbone the branch is added to |
| `--dlt_fusion` | `output` | `DLTPlug`: `output` (sum of forecasts), `refine`, `input` |
| `--dlt_train` / `--dlt_select` | `joint` / `none` | `DLT` and `DLTPlug`: `decoupled` trains the self head (backbone) first and the branch on its error; `variable` keeps the correction per variable on validation data; `significant` only where it helps consistently over blocks of validation windows (`--dlt_select_alpha`, 0.05) |
| `--dlt_anchor` / `--dlt_cross_input` | `mean` / `level` | `DLT`: forecast relative to the window mean or the last observation; the branch on the neighbours or their differences |
| `--dlt_lr` | 0 | `DLT` and `DLTPlug`: learning rate of the cross-variable branch (0 = `--learning_rate`) |
| `--seed` | 2021 | Seed of every run (alias `--fix_seed`); results and checkpoints depend on it |

The DLT-dyn recipe of the paper: learning rate 3e-3 halved every epoch, batch 32, 10 epochs, patience 3.

## Repository Layout

```
src/            model/ layers/ configs/ data_provider/ experiments/ utils/ ablation/
scripts/run.py  the only training entry point
scripts/<category>/   multivariate_forecasting  benchmark  plugin  ablation  hyperparam  efficiency  mechanism
                batch  utils  (job generators, analyses, queues)
output/<category>/    tracked result stores (one TSV row per run), tables and conclusions
assets/reported.json  numbers copied from the papers (never re-computed)
tests/          unit tests (python -m unittest discover -s tests)
data/  temp/  log/    datasets; checkpoints, job lists, raw predictions; logs (all ignored by git)
```

## Experiment Workflows

Every experiment is a **suite**. Its job generator writes one shell command per run to `temp/jobs/jobs_<list>.txt`;
every run appends one row to the tracked store `output/<category>/<suite>.tsv`; the queue runners run only the jobs
whose job name (`--des`) and seed have no row there yet, so an interrupted queue continues where it stopped:

```bash
python scripts/multivariate_forecasting/make_jobs.py long_small [--device gpu] [--seeds 2021] [--datasets ...]
bash scripts/utils/run_queue.sh long_small 4            # CPU, 4 jobs at a time
bash scripts/utils/run_gpu_queue.sh long_small 0 0      # two jobs at a time on GPU 0
python scripts/utils/run_missing.py long_small          # the jobs without a result
python scripts/utils/smoke_jobs.py long_small           # pre-flight: one training step per configuration, peak memory
```

| Experiment | Entry point | Datasets / grid | Output |
| --- | --- | --- | --- |
| Stand-alone DLT-dyn | `scripts/multivariate_forecasting/make_jobs.py long_small \| long_large \| long_extra`; `long_table.py` (against CI) | ETT, Exchange, Weather; ECL, Traffic; Solar and PEMS; 3 seeds | `output/long_term_forecast/<suite>.tsv`, `summary.md` |
| Baseline check | `scripts/benchmark/make_jobs.py baselines`, then `scripts/benchmark/baseline_check.py` | six baselines, official settings, 13 datasets | `output/benchmark/baselines*.tsv`, `baseline_check.md` |
| Plug-in | `scripts/plugin/make_jobs.py plugin_all \| plugin_fusion`; `plugin_table.py`, `plugin_gain.py`, `fusion_table.py`, `plugin_diag.py`, `leadlag_diag.py` | every backbone alone and + DLT-dyn on 13 datasets; five fusion arms | `output/plugin/` |
| Kernel and gate ablation (P0) | `scripts/ablation/make_jobs.py p0 \| ci_long \| allnb`; `aggregate.py` | 13 configurations × 5 settings × 3 seeds; CI linear; all neighbours | `output/ablation/` |
| Guards of the stand-alone model | `scripts/ablation/make_jobs.py guard`, then `scripts/ablation/guard_table.py` | DLT-dyn with per-variable selection, a slower branch, two-stage training; ETTh1/2, Exchange, Weather, PEMS08; 3 seeds | `output/ablation/guard.*` |
| Hyper-parameters | `scripts/hyperparam/make_jobs.py sensitivity \| neighbours`, `sens_table.py`; `hpsearch.py make \| select \| tuned` | L, R, k one at a time; neighbour selection; 20-trial validation search | `output/hyperparam/` |
| Efficiency | `scripts/efficiency/run_efficiency.py [--setting smamba] [--tuned]`; `make_jobs.py eff_tune`, `eff_tune.py` | parameters, ms per batch, peak memory; S-Mamba setting (batch 16) | `output/efficiency/` |
| Mechanism | `scripts/mechanism/kernels.py`, `pems_lag.py` | learned kernels against the PEMS08 road graph | `output/mechanism/kernels_pems08.json` |

`scripts/batch/run_cpu_campaign.sh [stage]` runs the CPU part of the study stage by stage; `scripts/batch/make_gpu_jobs.sh`
writes the lists of everything that needs a GPU. Every job carries the protocol of the official loaders. Conclusions and tables: [`output/README.md`](output/README.md).

## GPU Runs

The reference machine is one RTX 4090 (24 GB) with the environment above. Before a long queue run `poe check-mamba`
and `poe check-gpu`; records carry `device` and `torch`.

The tables of the GPU study use GPU runs only. A GPU job counts as done only with a GPU row in its store
(`scripts/utils/run_missing.py`), and the plug-in lists run the + DLT-dyn arms only (`--skip_alone`): the backbone
alone is the run of the baseline check, which the tables read with `--alone baselines --device gpu`. Every baseline
is reproduced once with its official settings; a reproduction close to the published result is enough, and one more
than 10 % behind it is marked in the tables, not re-tuned. Done on the GPU (2026-11-25): the baselines except 28
settings, DLT-dyn on Solar and PEMS (three seeds; PEMS08 seed 2021 still to run) and the efficiency measurement
(`output/efficiency/efficiency_smamba_gpu.jsonl`).

```bash
bash scripts/batch/make_gpu_jobs.sh                          # lists in temp/jobs/, with the GPU runs left
bash scripts/utils/run_gpu_queue.sh baselines 0 0            # 1. the baselines alone: the settings without a GPU run
python scripts/benchmark/baseline_check.py                   #    against the published results (GPU, official settings)
bash scripts/utils/run_gpu_queue.sh long_extra 0 0 0         # 2. stand-alone DLT-dyn on Solar and PEMS
bash scripts/utils/run_gpu_queue.sh plugin_fusion 0 0 0      # 3. the five fusions with the six backbones
python scripts/plugin/fusion_table.py --alone baselines --device gpu
bash scripts/utils/run_gpu_queue.sh plugin_main 0 0          # 4. plug-in benchmark: + DLT-dyn, two-stage, per-variable selection
python scripts/plugin/plugin_table.py --alone baselines --device gpu --arm dlt_train-decoupled-select-variable
```

- **Baselines.** One reproduction per setting with the official settings and seed 2021; the comparison with the
  published numbers is `baseline_check.py`. The tools for a learning-rate check remain
  (`scripts/benchmark/make_jobs.py baselines --backbone_lr ...`, `baseline_check.py --lr_check`, `plugin_all --select_lr`)
  but are not part of the study; the tables use the official settings.
- **Memory.** DLT is small (≤ 0.36 M parameters; about 1 GB per step on Traffic at batch 32). The heavy runs are the
  baselines on Traffic (iTransformer 12.4 GB at batch 32) and TiDE, which runs one pass per variable: on data with
  more than 100 variables its jobs carry `--tide_checkpoint` (recomputation in the backward pass, same results).
  DLT-dyn with all neighbours on Traffic (`allnb`) may need 15–20 GB; use `--batch_size 16` if it does not fit.
- **Branch recipes.** `--dlt "<args>"` gives a changed recipe its own job names (e.g. `--dlt "--dlt_gate static"` →
  `..._dlt_gate-static`), compared with `plugin_table.py --arm dlt_gate-static`.
- **Devices.** CPU and GPU runs differ slightly (TF32, non-deterministic kernels): each table row comes from one
  device, and both arms of a plug-in pair run on the same device. The queue counts only rows of the job's device, and
  the tables of the GPU study filter with `--device gpu`.
- **Non-finite forecasts.** `python scripts/utils/diag_nonfinite.py <job name> --list <list>` locates the NaN / inf
  forecasts of a job (it needs the job in its list and its checkpoint in `temp/checkpoints/`).

## Upstream Code and License

The data pipeline, the experiment loop and the baselines DLinear, PatchTST, TiDE and iTransformer derive from
[thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (commit `4e938a1`, MIT, see `LICENSE`);
S-Mamba and RLinear follow their official code. The changes to that code are listed in `AGENTS.md`.
