# Experiments that still need a GPU

**Target machine:** one RTX 4090 (24 GB), host CUDA 13.0, PyTorch 2.10.0+cu130 in its own conda environment
(set-up in section 0). All commands run from the repository root.

**Environment change (2026-11-13).** The first GPU check and the tier-1 results so far
([`results/gpu`](results/gpu/README.md)) were run with PyTorch 2.5.1+cu124. That build refuses the host's CUDA 13.0
compiler for `mamba_ssm`, so the reference environment is now PyTorch 2.10.0+cu130 with the CUDA 13 wheels of
`mamba_ssm`. Result records written from now on carry `torch` and `device`; older records have neither.
- Run the checks of section 0 again in the new environment.
- For GPU tables from one environment only, run tier 1 again in it too: keep the old work directory
  (`mv work work_cu124`) and write the job lists anew (`bash scripts/gpu/make_gpu_jobs.sh`).

Everything that is feasible on the 4-core CPU development machine has been run there:
- P0 kernel and gate study: [`results/p0`](results/p0).
- P1 comparisons: [`results/p1`](results/p1). This covers:
  - PEMS04/08 at horizon 12;
  - the six small long-term datasets and Electricity;
  - the 20-trial validation search on the small datasets;
  - plug-in on ETTh1, Weather and PEMS08;
  - the CI linear model;
  - all-neighbour runs;
  - the light baselines.
- P2 sensitivity, neighbour selection, CPU efficiency and learned kernels: [`results/p2`](results/p2).

This page lists what is left: the parts that are too slow or too large for that machine. For each part it says
what the part verifies and how to run it. Start with the single run of the [README quick start](README.md).

## Status of the GPU check

**First run on the RTX 4090 (2026-11-10, earlier version of `scripts/gpu/check_gpu.py`).**
- 19 of the 23 model configurations passed the strict CPU-vs-GPU comparison of outputs and gradients. These
  include stand-alone DLT-dyn on all three computation paths, both gates and DLTPlug with DLinear.
- The neighbour and lag selection checks passed.
- Four configurations were reported as failed: `DLT milldyn`, `DLTPlug PatchTST + dltdyn`,
  `DLTPlug iTransformer + dltdyn`, `LagTS`.

**Cause: the check, not the models.**
- These four are exactly the configurations with dot-product attention.
- The bias of an attention key projection adds the same amount to every score of a query, and the softmax ignores
  such a shift. Its gradient is therefore exactly zero in exact arithmetic.
- In float32 that gradient holds only rounding noise, about 1e-9 of the largest gradient of the model, and the
  noise differs between devices.
- The old criterion divided the difference of each gradient by the size of that same gradient. For these biases
  it compared noise with noise and got a relative error of about 1.

**Evidence without a GPU.** The same weights were evaluated in float32 and in float64 on the CPU; like a second
device, the two differ only by rounding.
- The old criterion fails for exactly these four configurations and for no other.
- The only offending parameters are `k_proj.bias` (milldyn) and `attention.key_projection.bias` (PatchTST,
  iTransformer, LagTS).
- Their float64 gradient is below 1e-17 of the largest gradient.
- All outputs agree to better than 1e-6, and all other gradients to about 1e-6 or better.

**Fix.** The check now:
- compares in float64 first (tolerance 1e-9);
- measures each gradient error against at least 1e-3 of the largest gradient of the model;
- prints the parameter with the largest error and lists the zero-gradient parameters.

With the same rounding proxy all 23 configurations pass; the largest error is 1.4e-6 against a tolerance of 1e-4.
A re-run on the RTX 4090 with `--ablation` (every configuration) is the remaining confirmation.

## Status of the experiments (2026-11-11)

Evaluation of the results so far: [`results/gpu/README.md`](results/gpu/README.md).

| tier | state |
|---|---|
| 1a Traffic, stand-alone DLT-dyn and CI | **done**, 3 seeds. The cross branch lowers the MSE of the linear model by 15 %; DLT-dyn (0.529) stays far behind iTransformer (0.428 reported). |
| 1b PatchTST ± DLT-dyn on PEMS08-12 | no result seen; check the logs for failed runs |
| 1c plug-in on ECL and Traffic | seeds 2021 and 2022 done. DLinear gains 10–13 %, PatchTST gains 8 % on ECL, iTransformer is unchanged on ECL, and on Traffic the branch costs PatchTST 13.6 % and iTransformer 11.5 %. |
| 1c plug-in on PEMS04-12 | no result seen; check the logs for failed runs |
| 0, 2, 3, 4 | not started |
| plug-in study, all eight backbones (`plugin_all`, 2026-11-13) | not started; commands in the section of that name below |

**What to run next.**
1. Stop the tier 1 queue. The 32 remaining runs repeat finished configurations with seed 2023 and cannot change
   a conclusion; the + DLT-dyn arms on Traffic need a changed recipe first.
2. `python scripts/p1/plugin_diag.py`: epochs, best epoch and best validation loss of every plug-in arm
   (no new runs).
3. The plug-in probe on Traffic at horizon 96 (six runs; the list is written by `make_gpu_jobs.sh`):
   ```bash
   bash scripts/gpu/run_gpu_queue.sh plugin_probe
   python scripts/p1/plugin_diag.py
   ```
4. Tier 0 (efficiency table), and the PEMS runs of tier 1 once the reason for their absence is known:
   `python scripts/run_missing.py gpu_tier1 | grep -c PEMS` counts them, and
   `grep -l -E "Error|Traceback" work/logs/plugin*/*PEMS*.log` lists failed ones. A missing `PEMS04.npz` or
   `PEMS08.npz` is now reported when the job lists are written.
5. Tier 2 for PEMS only. The gaps of the stand-alone model on ECL and Traffic are too large for a search of k, R
   and the learning rate to close. Tier 3 as far as the baseline table needs it.

## Fixed locations

Every location is fixed in [`scripts/paths.py`](scripts/paths.py); no path argument and no environment variable is
needed. On the GPU machine the repository sits next to the data and the Mamba sources, and nothing has to be set up:

```
~/DLT              this repository; work/ inside it holds job lists, logs, checkpoints and result records
~/data             the datasets (used because DLT/dataset does not exist)
~/mamba            state-spaces/mamba source checkout  } used by S-Mamba when mamba_ssm is not installed
~/causal-conv1d    Dao-AILab/causal-conv1d checkout     } in the environment itself
```

In general: the data is `dataset/` inside the repository, or `../data` next to it when `dataset/` does not exist;
the work directory is `work/` inside the repository. Both are ignored by git.

- Data layout: the Time-Series-Library layout (`dataset/ETT-small/ETTh1.csv`, `dataset/weather/weather.csv`,
  `dataset/electricity/electricity.csv`, `dataset/traffic/traffic.csv`, `dataset/exchange_rate/exchange_rate.csv`,
  `dataset/illness/national_illness.csv`, `dataset/PEMS/PEMS03.npz` … `PEMS08.npz` with `PEMS0x_edges.csv`,
  `dataset/Solar/solar_AL.txt`), or all files directly in `dataset/`. `bash scripts/download_data.sh` fetches all
  of them (`--mirror` where huggingface.co or GitHub are blocked) and checks their sizes; `python
  scripts/check_data.py` repeats the check. A missing CSV is downloaded by Time-Series-Library at run time; a
  missing PEMS or Solar file makes its jobs fail.
- `work/`: if results of an earlier run live in another directory, move that directory to `work`
  (`mv <old work directory> work`) or link it (`ln -s <old work directory> work`). Finished runs are recognised
  by job name and seed, wherever the list was generated.
- If the data is elsewhere, link it the same way (`ln -s <data directory> dataset`) or edit the lines of
  `scripts/paths.py`.
- Our per-dataset scripts (`scripts/long_term_forecast/DLT_script/*.sh`, `*_script/SMamba*.sh`) take the data from
  the same place in either layout; the other upstream scripts of `scripts/` expect `./dataset/` in the
  Time-Series-Library layout.

## Overview

Job counts are runs that have no result yet, starting from the CPU records of this repository
(`make_gpu_jobs.sh` merges them into `work/`). Tiers 1, 2 and 4 use seeds 2021, 2022 and 2023; tier 3 uses seed 2021.

| tier | experiments | runs | what it verifies / which table it completes |
|---|---|---:|---|
| 0 | GPU correctness check; efficiency on the GPU | – | The GPU code paths match the CPU reference. Parameters, time per batch and peak GPU memory of DLT-dyn and the eight baselines on Weather, PEMS08, ECL and Traffic (paper efficiency table; MillGNN reports 9.7 GB, iTransformer 7.5 GB). |
| 1a | DLT-dyn and CI linear on Traffic, 4 horizons | 24 | The last dataset of the long-term table, and whether the cross branch helps on Traffic. |
| 1b | PatchTST ± DLT-dyn on PEMS08-12 | 6 | The plug-in claim for the third backbone on PEMS08. |
| 1c | Plug-in on large data: DLinear / PatchTST / iTransformer ± DLT-dyn on ECL and Traffic (4 horizons each) and PEMS04-12 | 162 | Whether the plug-in gains (Weather −7 to −17 %, PEMS08 −4 to −27 %) hold on large data. |
| probe | + DLT-dyn on Traffic-96 with the learning rate of the backbone, with a static gate, and with both (PatchTST, iTransformer) | 6 | Why the plug-in hurts on Traffic ([`results/gpu/README.md`](results/gpu/README.md), section 4). |
| 2 | 20-trial validation search on ECL, Traffic (H = 96) and PEMS04/08 (H = 12), then the selected configurations: ECL/Traffic × 4 horizons, PEMS04/08 at H = 12 | 80 + 30 | Fair tuning budget against MillGNN (20 NNI trials) on the data where DLT is strongest. |
| 3 | The eight representative baselines with their official hyper-parameters on the 8 long-term datasets × 4 horizons, one seed; 81 of the 256 runs were done on the CPU | 175 | Reported vs re-run baseline numbers under one pipeline (fairness). |
| 4 (optional) | All other variables as neighbours: DLT-dyn on ECL; DLT and DLT-dyn on Traffic | 36 | Whether more neighbours help on large data. Low priority: at horizon 12 and for DLT on ECL they overfitted. |

**The eight representative baselines** (`BASELINES` in [`scripts/exps.py`](scripts/exps.py)), one per model family:

| model | family | hyper-parameters |
|---|---|---|
| Informer | classic Transformer | TSLib scripts (one generic setting) |
| TimesNet | CNN on 2-D period maps | TSLib scripts |
| DLinear | linear | TSLib scripts |
| PatchTST | channel-independent patch Transformer | TSLib scripts |
| iTransformer | Transformer over variate tokens | TSLib scripts |
| TimeMixer | multi-scale MLP mixing | TSLib scripts |
| TimeXer | patch and variate tokens | TSLib scripts (ETTh1 setting on Exchange) |
| S-Mamba | bidirectional Mamba over variate tokens | official S-Mamba scripts, as `scripts/long_term_forecast/*_script/SMamba*.sh` |

S-Mamba is not part of Time-Series-Library; `models/SMamba.py` re-implements it after the official code. It uses
`mamba_ssm` (installed, or from `~/mamba` and `~/causal-conv1d`) and a pure-PyTorch block otherwise (same model,
sequential scan); `python scripts/gpu/check_mamba.py` checks which one a GPU run will use.
Trained on the CPU with the official settings it reproduces the published numbers: the average over the four
horizons is within +0.9 % (ETTh1), +0.3 % (ETTh2) and −0.6 % (Exchange) of the paper's MSE, Weather-96 within
+1.1 %; single horizons lie between −1.1 % and +4.2 % (13 runs in `results/p1/results_baselines.jsonl`).
With the PyTorch block one step per variate token is taken in Python, which is slow on Electricity and Traffic:
make sure `check_mamba.py` ends with "S-Mamba will use mamba_ssm" before tier 3 (set-up below). LagTS (our re-implementation, at most 21 variables) is no
longer in the default set; `python scripts/exps.py baselines --models LagTS` adds it.

**Cost of tier 3.** TimesNet is the expensive model: 150 M parameters on Electricity and 302 M on Traffic with its
official settings (112 s and 123 s per training batch on the 4-thread CPU). Order the list by need:
`python scripts/exps.py baselines --seeds 2021 --device gpu --models <models> --datasets <datasets>` writes a
smaller list `baselines`.

**Memory on one 24 GB card.**
- DLT is small: DLT-dyn needs about 1 GB per step on Traffic at batch 32 (measured on the CPU).
- The heavy runs are the baselines on Traffic: iTransformer (12.4 GB at batch 32 on the CPU; its TSLib script
  uses batch 16), PatchTST with the TSLib configuration (d_model 512, batch 4), TimeMixer (batch 8), TimesNet
  (5.9 GB at batch 32 on the CPU) and TimeXer (10.9 GB on Electricity at batch 32 on the CPU; its script uses
  batch 4 there and batch 16 on Traffic).
- DLT-dyn with all neighbours on Traffic (tier 4) is the one to watch: the pair features take about 3 GB per
  tensor at batch 32, so the peak is estimated at 15–20 GB (not measured). Use `--batch_size 16` if it does not fit.

Run tier 0 first: its per-batch times calibrate the time needed for the other tiers.

## 0. Set-up and checks

```bash
# a new conda environment for the host's CUDA 13.0 (the driver must support CUDA 13.0: see nvidia-smi)
conda create -y -n dlt-cu130 python=3.12
conda activate dlt-cu130
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install --no-deps einops==0.8.1 local-attention==1.11.2 reformer-pytorch==1.4.4 hyper-connections \
    axial-positional-embedding product-key-memory==0.1.10                  # imports of the Transformer baselines
# S-Mamba: mamba_ssm and causal_conv1d prebuilt for CUDA 13 + torch 2.10 + Python 3.12; --no-deps: S-Mamba needs
# neither transformers nor the other dependencies of mamba_ssm
pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

python scripts/gpu/check_mamba.py               # must end with "S-Mamba will use mamba_ssm ... Nothing to do."
python scripts/gpu/check_gpu.py                 # stand-alone DLT-dyn; must end with "all checks passed"
python scripts/gpu/check_gpu.py --all           # the ten representative configurations, incl. plug-in and S-Mamba
python scripts/p2/efficiency.py --device gpu    # -> results/p2/efficiency_gpu.jsonl
```

`check_gpu.py` builds each model on the CPU, copies it to the GPU and compares outputs and all gradients:
- in float64: tolerance 1e-9;
- in float32 with TensorFloat-32 off: tolerance 1e-4;
- in float32 as in training (PyTorch enables TF32 for cuDNN convolutions): reported only.

It also checks that neighbour selection on the GPU picks the same sources as on the CPU (ties, for example a
dead sensor, may be broken differently and are accepted when the scores agree) and, when `mamba_ssm` can be
imported, that S-Mamba with the `mamba_ssm` block equals S-Mamba with the pure-PyTorch block. `--ablation` adds the
other kernel variants of the P0 study and LagTS.

**S-Mamba and the Mamba sources.** `models/SMamba.py` imports the Mamba block of `mamba_ssm`
(`mamba_ssm.modules.mamba_simple`) from the environment, else from the checkouts `../mamba` and `../causal-conv1d`;
without it (or on the CPU) it uses the pure-PyTorch block, which is the same model but takes one Python step per
variate token (slow on Electricity and Traffic). Every run prints the block it uses (`[S-Mamba] Mamba block: ...`)
and stores it in its record (`mamba_impl`).
- The block is imported without `mamba_ssm/__init__.py`. That file loads the language models and Mamba-2/3, and
  with them `transformers` and further packages, none of which S-Mamba needs: hence `--no-deps` above.
- The block does need the compiled selective-scan kernel `selective_scan_cuda`. The release wheels contain it.
  A source install of mamba_ssm 2.3 or later builds it only with `MAMBA_KEEP_CUDA_BUILD=TRUE`; without it,
  mamba_ssm imports but fails at the first step. S-Mamba counts such an install as not importable.

`python scripts/gpu/check_mamba.py` shows
- PyTorch, the CUDA version and C++ ABI it was built with, the NVIDIA driver and the nvcc on the PATH,
- the version of the two checkouts,
- whether `selective_scan_cuda` (required), `causal_conv1d_cuda` (optional: fused path) and the Mamba block
  import, and from where, or the exact error,
- on the GPU: one `mamba_ssm` block (fused and unfused path) against the PyTorch block with the same parameters,
  and the speed of both,

and ends with the verdict and, if needed, the commands for the machine it runs on:
- the prebuilt wheels, installed with `--no-deps`. For CUDA 13 with torch 2.9 or 2.10 this is the pair
  mamba_ssm 2.3.2.post1 / causal_conv1d 1.7.0 (Python 3.10–3.13); for CUDA 12 with torch 2.5, the pair 2.2.4 /
  1.5.0.post8. Without access to github.com, download the files elsewhere and `pip install` them.
- or the build from the checkouts with the host's nvcc. PyTorch accepts only an nvcc of the CUDA major version it
  was built for: CUDA 13.0 works with torch +cu130, not with the earlier +cu124.
  ```bash
  pip install ninja packaging
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 pip install --no-build-isolation --no-deps ~/causal-conv1d
  MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE MAX_JOBS=4 pip install --no-build-isolation --no-deps ~/mamba
  ```
  This compiles for every GPU architecture and takes long; the wheels are the quicker way.

## Plug-in study: every backbone alone and + DLT-dyn (suite `plugin_all`)

Each of the eight representative backbones alone and with the DLT-dyn branch, on thirteen datasets: the eight
long-term sets and Solar at horizons 96/192/336/720, PEMS03/04/07/08 at 12/24/48/96. That is 8 × 13 × 4 × 2 = 832
runs per seed.
- Both arms run through `DLTPlug`. "Alone" (`--dlt_variant ci`) builds only the backbone and is identical to
  running the backbone directly; "+ DLT" adds the branch with its default recipe (DLT-dyn, k 8, L 48, R 4, branch
  learning rate 3e-3).
- Backbone settings: the official ones.
  - Long-term sets: the TSLib scripts. PatchTST: its paper's settings, as in the earlier plug-in study.
  - Solar and PEMS: the repositories of iTransformer, S-Mamba and TimeMixer, per horizon (`REPO_OFFICIAL` in
    `scripts/exps.py`; TimeMixer has one PEMS setting, horizon 12, used for every horizon).
  - Models without settings for Solar or PEMS (Informer, TimesNet, TimeXer, DLinear) take the documented
    fallbacks: TimesNet and TimeXer the Electricity configuration, Informer its generic one, DLinear the defaults.
- PEMS follows iTransformer and S-Mamba: input 96, MSE loss, MSE / MAE on the normalised scale. The other suites
  use a different PEMS-12 protocol (L1 loss, metrics on the original scale); the table of this study reads only its
  own records.
- Data: `bash scripts/download_data.sh` fetches Solar and PEMS03/07 too.
- A different branch recipe gets its own job names: `--dlt "--dlt_gate static"` writes `..._dlt_gate-static` jobs,
  and `plugin_table.py --arm dlt_gate-static` compares them. The default runs are neither skipped nor mixed.

One seed, in three stages, small data first so that the first comparisons arrive early:

```bash
cd ~/DLT && conda activate dlt-cu130
python scripts/exps.py plugin_all --device gpu --seeds 2021      # the whole list (832 jobs)
python scripts/smoke_jobs.py plugin_all       # pre-flight: one training step per configuration, peak GPU memory
nohup bash -c '
python scripts/exps.py plugin_all --device gpu --seeds 2021 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather &&
bash scripts/gpu/run_gpu_queue.sh plugin_all 0 0 0
python scripts/exps.py plugin_all --device gpu --seeds 2021 --datasets Solar PEMS03 PEMS04 PEMS08 &&
bash scripts/gpu/run_gpu_queue.sh plugin_all 0 0
python scripts/exps.py plugin_all --device gpu --seeds 2021 --datasets ECL Traffic PEMS07 &&
bash scripts/gpu/run_gpu_queue.sh plugin_all
' > work/plugin_all.out 2>&1 &

tail -n 5 work/plugin_all.out                   # progress; "failed on GPU ..." names a failed job
python scripts/p1/plugin_table.py              # the comparison so far (markdown); --metric mae for MAE
```

- The pre-flight trains every configuration for one step with its own batch size on the GPU and prints the peak
  GPU memory, so a configuration that does not fit fails here and not after hours. Jobs that run at the same time
  on the card need the sum of their peaks.
- The stages run one after the other. Three jobs at a time suit the small datasets, one at a time Electricity,
  Traffic and PEMS07 (862 and 883 variables; TimesNet on Electricity / Traffic: 150 M / 302 M parameters).
- A failed job leaves its log in `work/logs/plugin_all/<job>_s2021.log`. Running the same stage again runs only
  the jobs without a result.
- More seeds later: the same commands with `--seeds 2022 2023`; the table averages the seeds.
- After a stage: `python scripts/collect_results.py` merges the records into `results/p1/results_plugin_all.jsonl`
  and writes the table `results/p1/plugin_all.md`. These two files (and `work/plugin_all.out`, the queue log) are
  what to commit or upload; `results/p1/summary.md` does not contain this study.

## 1–4. Experiments

```bash
bash scripts/gpu/make_gpu_jobs.sh                     # writes the lists into work/ and prints the counts

bash scripts/gpu/run_gpu_queue.sh gpu_tier1           # one job at a time on GPU 0
bash scripts/gpu/run_gpu_queue.sh plugin_probe        # the six diagnosis runs

bash scripts/gpu/run_gpu_queue.sh gpu_tier2_search 0 0 0     # three DLT jobs at a time on GPU 0
python scripts/p1/hpsearch.py select > work/hp_select.md                        # 8 long-term datasets
python scripts/p1/hpsearch.py tuned  --device gpu
python scripts/p1/hpsearch.py select --tag _pems > work/hp_select_pems.md
python scripts/p1/hpsearch.py tuned  --device gpu --horizons 12 --tag _pems
cat work/jobs_tuned.txt work/jobs_tuned_pems.txt > work/jobs_gpu_tier2_tuned.txt
bash scripts/gpu/run_gpu_queue.sh gpu_tier2_tuned 0 0 0

bash scripts/gpu/run_gpu_queue.sh gpu_tier3
bash scripts/gpu/run_gpu_queue.sh gpu_tier4           # optional
```

How the queue runner behaves:
- A list is named by its suite (`work/jobs_<name>.txt`).
- It runs only jobs whose result file has no record with the job's `--des` and seed. After an interruption,
  run the same command again.
- One worker per listed GPU id. On the single RTX 4090, no id (or `0`) runs one job at a time. `0 0 0` runs three
  at a time, which suits the small DLT models of tier 2; keep one at a time for the Traffic baselines.
- Each job sees only its GPU through `CUDA_VISIBLE_DEVICES`. TSLib's `_acquire_device` keeps a mask set by
  the caller.

The tuned step covers the 72 CPU runs again. They are skipped because their records are in `work/`, and the
selection for the 6 small datasets is the same because their search records are there as well.

## Tables

```bash
python scripts/collect_results.py     # merges work/results_*.jsonl into results/p0, p1, p2 and rebuilds the three summary.md
```

Records are merged by job name and seed, so nothing in `results/` is lost; `efficiency.py` writes into
`results/p2` directly. Commit `results/` afterwards.

## Notes

- **Mixing devices.**
  - CPU and GPU runs differ slightly: TF32, non-deterministic cuDNN and atomics. Each table row comes from one
    device, and both arms of every plug-in pair (alone / + DLT-dyn) run in the same tier on the same device.
  - Traffic rows (GPU) next to the other datasets (CPU) are comparable as rows.
  - Do not compare a CPU number with a GPU number for the same configuration as if they were seeds.
- **Not run anywhere, by design.**
  - MillGNN has no public code; it is compared through its reported numbers (Tables 2 and 3).
  - MoVE is compared through Table 4 of its submission (input 96, seed 2024).
  - The MillGNN-style and MoVE-style mechanisms are re-implemented inside DLT for the controlled study (P0).
- **If the GPU budget is small.** Run tier 0, then 1a and 1b, the probe, then 1c with iTransformer only. Run tier 2
  for PEMS only (40 + 6 runs). In tier 3 leave out TimesNet on Electricity and Traffic and use its reported
  numbers (`scripts/p1/reported.json` holds them, next to those of S-Mamba).
