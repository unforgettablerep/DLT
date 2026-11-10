# Experiments that still need a GPU

**Target machine:** one RTX 4090 (24 GB), host CUDA 13.1, PyTorch 2.5.1+cu124. The cu124 wheel ships its own
CUDA 12.4 runtime, so the host only needs a recent NVIDIA driver. All commands run from the repository root.

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
A re-run on the RTX 4090 with `--all` is the remaining confirmation.

## Overview

Job counts are runs that have no result yet; they assume the CPU results are copied into the work directory.
Every run is one training with seed 2021, 2022 or 2023.

| tier | experiments | runs | what it verifies / which table it completes |
|---|---|---:|---|
| 0 | GPU correctness check; efficiency on the GPU | – | The GPU code paths match the CPU reference. Parameters, time per batch and peak GPU memory for 15 models on Weather, PEMS08, ECL and Traffic (paper efficiency table; MillGNN reports 9.7 GB, iTransformer 7.5 GB). |
| 1a | DLT-dyn and CI linear on Traffic, 4 horizons | 24 | The last dataset of the long-term table, and whether the cross branch helps on Traffic. |
| 1b | PatchTST ± DLT-dyn on PEMS08-12 | 6 | The plug-in claim for the third backbone on PEMS08. |
| 1c | Plug-in on large data: DLinear / PatchTST / iTransformer ± DLT-dyn on ECL and Traffic (4 horizons each) and PEMS04-12 | 162 | Whether the plug-in gains (Weather −7 to −17 %, PEMS08 −4 to −27 %) hold on large data. |
| 2 | 20-trial validation search on ECL, Traffic (H = 96) and PEMS04/08 (H = 12), then the selected configurations: ECL/Traffic × 4 horizons, PEMS04/08 at H = 12 | 80 + 30 | Fair tuning budget against MillGNN (20 NNI trials) on the data where DLT is strongest. |
| 3 | Baselines re-run with the official TSLib hyper-parameters, not run on CPU: PatchTST (all 8 datasets); iTransformer and TimeMixer on Weather, ECL, Traffic; DLinear on Traffic; LagTS on Weather | 192 | Reported vs re-run baseline numbers under one pipeline (fairness). |
| 4 (optional) | All other variables as neighbours: DLT-dyn on ECL; DLT and DLT-dyn on Traffic | 36 | Whether more neighbours help on large data. Low priority: at horizon 12 and for DLT on ECL they overfitted. |

The make script generates the complete suites. For tier 3, the generated list also holds the light baselines,
and for tier 4 the PEMS all-neighbour runs. Those were run on the CPU, so their jobs are skipped once the final
CPU results are copied in.

**Memory on one 24 GB card.**
- DLT is small: DLT-dyn needs about 1 GB per step on Traffic at batch 32 (measured on the CPU).
- The heavy runs are the baselines on Traffic: iTransformer (12.4 GB at batch 32 on the CPU; its TSLib script
  uses batch 16), PatchTST with the TSLib configuration (d_model 512, batch 4) and TimeMixer (batch 8).
- DLT-dyn with all neighbours on Traffic (tier 4) is the one to watch: the pair features take about 3 GB per
  tensor at batch 32, so the peak is estimated at 15–20 GB (not measured). Use `--batch_size 16` if it does not fit.

Run tier 0 first: its per-batch times calibrate the time needed for the other tiers.

## 0. Set-up and checks

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-deps einops==0.8.1 local-attention==1.11.2 reformer-pytorch==1.4.4 hyper-connections \
    axial-positional-embedding product-key-memory==0.1.10                  # imports of the Transformer baselines

DATA=/path/to/datasets      # one flat directory: ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv weather.csv
                            # exchange_rate.csv electricity.csv traffic.csv PEMS04.npz PEMS08.npz PEMS08_edges.csv
WORK=/path/to/gpu_work
mkdir -p $WORK && cp results/p1/results_*.jsonl $WORK/      # finished CPU runs are skipped

python scripts/gpu/check_gpu.py --data $DATA                # stand-alone DLT-dyn; must end with "all checks passed"
python scripts/gpu/check_gpu.py --data $DATA --all          # every variant, gate, DLTPlug and LagTS
python scripts/p2/efficiency.py --device gpu --out results/p2/efficiency_gpu.jsonl
```

`check_gpu.py` builds each model on the CPU, copies it to the GPU and compares outputs and all gradients:
- in float64: tolerance 1e-9;
- in float32 with TensorFloat-32 off: tolerance 1e-4;
- in float32 as in training (PyTorch enables TF32 for cuDNN convolutions): reported only.

It also checks that neighbour selection on the GPU picks the same sources as on the CPU. Ties, for example a
dead sensor, may be broken differently and are accepted when the scores agree.

## 1–4. Experiments

```bash
bash scripts/gpu/make_gpu_jobs.sh $DATA $WORK         # writes jobs_gpu_tier{1,2_search,3,4}.txt, prints the counts

bash scripts/gpu/run_gpu_queue.sh $WORK/jobs_gpu_tier1.txt            # one job at a time on GPU 0

bash scripts/gpu/run_gpu_queue.sh $WORK/jobs_gpu_tier2_search.txt 0 0 0     # three DLT jobs at a time on GPU 0
python scripts/p1/hpsearch.py select --work $WORK > $WORK/hp_select.md                      # 8 long-term datasets
python scripts/p1/hpsearch.py tuned  --data $DATA --work $WORK --device gpu
python scripts/p1/hpsearch.py select --work $WORK --tag _pems > $WORK/hp_select_pems.md
python scripts/p1/hpsearch.py tuned  --data $DATA --work $WORK --device gpu --horizons 12 --tag _pems
cat $WORK/jobs_tuned.txt $WORK/jobs_tuned_pems.txt > $WORK/jobs_gpu_tier2_tuned.txt
bash scripts/gpu/run_gpu_queue.sh $WORK/jobs_gpu_tier2_tuned.txt 0 0 0

bash scripts/gpu/run_gpu_queue.sh $WORK/jobs_gpu_tier3.txt
bash scripts/gpu/run_gpu_queue.sh $WORK/jobs_gpu_tier4.txt            # optional
```

How the queue runner behaves:
- It runs only jobs whose result file has no record with the job's `--des` and seed. After an interruption,
  run the same command again.
- One worker per listed GPU id. On the single RTX 4090, `0` runs one job at a time. `0 0 0` runs three at a
  time, which suits the small DLT models of tier 2; keep one at a time for the Traffic baselines.
- Each job sees only its GPU through `CUDA_VISIBLE_DEVICES`. TSLib's `_acquire_device` keeps a mask set by
  the caller.

The tuned step covers the 72 CPU runs again. They are skipped because their results were copied, and the
selection for the 6 small datasets is the same because their search records were copied as well.

## Tables

```bash
python scripts/p1/aggregate.py $WORK results/p1/summary.md      # CPU + GPU rows; PEMS table gains "DLT-dyn (tuned)"
cp $WORK/results_*.jsonl $WORK/tuned*.json $WORK/hp_select*.md results/p1/
python scripts/p2/aggregate.py results/p0/results_p0.jsonl $WORK results/p2 results/p2/summary.md   # adds efficiency_gpu
```

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
- **If the GPU budget is small.** Run tier 0, then 1a and 1b, then 1c with iTransformer only. Run tier 2 for
  PEMS only (40 + 6 runs). Use reported numbers for tier 3.
