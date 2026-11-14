# GPU tier 1, partial results: evaluation (2026-11-11)

Newer: the plug-in study with all eight backbones, stage 1 and PEMS04/08 (2026-11-14):
[`plugin_all_stage1.md`](plugin_all_stage1.md).

**Source.** 136 result lines of `result_long_term_forecast.txt` from the RTX 4090 run (PyTorch 2.5.1+cu124, code
version before the `tslib/` folder was merged into the root), parsed into [`tier1_partial.jsonl`](tier1_partial.jsonl). These lines
hold test MSE and MAE only. Epochs, best epoch and validation loss are in the `results_*.jsonl` files of the work
directory of that run; they are needed for the diagnosis below.

**Coverage.**
- Tier 1a, stand-alone DLT-dyn and the CI linear model on Traffic: complete, 3 seeds × 4 horizons each.
- Tier 1c, plug-in on Electricity (ECL) and Traffic: seeds 2021 and 2022 complete for all three backbones and both
  arms. Of seed 2023: ECL PatchTST (both arms), Traffic PatchTST + DLT-dyn, ECL iTransformer + DLT-dyn.
- Not in the lines: the other 32 runs of seed 2023, and every PEMS run (tier 1b: PatchTST ± DLT-dyn on PEMS08-12,
  6 runs; tier 1c: PEMS04-12, 18 runs). The queue order puts the PEMS runs before and between the finished ones,
  so they have probably failed (for example a missing `PEMS04.npz` / `PEMS08.npz`); check the logs.

## 1. Stand-alone on Traffic (3 seeds)

| horizon | DLT-dyn MSE / MAE | CI linear MSE / MAE | Δ MSE | Δ MAE |
|---|---|---|---:|---:|
| 96 | 0.5186 ± 0.0033 / 0.3503 | 0.6448 ± 0.0000 / 0.3830 | -19.6 % | -8.5 % |
| 192 | 0.5133 ± 0.0028 / 0.3408 | 0.5979 ± 0.0000 / 0.3594 | -14.2 % | -5.2 % |
| 336 | 0.5236 ± 0.0004 / 0.3437 | 0.6053 ± 0.0000 / 0.3620 | -13.5 % | -5.1 % |
| 720 | 0.5616 ± 0.0053 / 0.3642 | 0.6430 ± 0.0001 / 0.3819 | -12.7 % | -4.6 % |
| average | 0.529 / 0.350 | 0.623 / 0.372 | -15.0 % | -5.9 % |

- The cross-variable branch helps the linear model at every horizon (Welch p ≤ 0.002), most at the short horizon.
  This matches Weather, Electricity and PEMS.
- In absolute terms DLT-dyn (0.529 / 0.350) is behind every deep model on Traffic: reported iTransformer
  0.428 / 0.282, MillGNN 0.423 / 0.276, PatchTST 0.481 / 0.304, TimeMixer 0.484 / 0.297, LIFT 0.514 / 0.328. It is
  ahead of DLinear only (reported 0.625 / 0.383).

## 2. Plug-in on Electricity and Traffic

MSE / MAE averaged over the four horizons and the available seeds. Δ = change against the backbone alone.
"Reported backbone" is the published average of that backbone.

| dataset | backbone | seeds (alone / + DLT-dyn) | alone MSE / MAE | + DLT-dyn MSE / MAE | Δ MSE | Δ MAE | Δ MSE per horizon (96 / 192 / 336 / 720) | reported backbone |
|---|---|---|---|---|---:|---:|---|---|
| ECL | DLinear | 2 / 2 | 0.225 / 0.319 | 0.202 / 0.300 | -10.3 % | -5.8 % | -13.6 / -10.7 / -9.2 / -8.2 % | 0.212 / 0.300 |
| ECL | PatchTST | 3 / 3 | 0.230 / 0.329 | 0.211 / 0.305 | -8.2 % | -7.4 % | -10.0 / -8.5 / -7.5 / -7.4 % | 0.205 / 0.290 |
| ECL | iTransformer | 2 / 3 | 0.178 / 0.268 | 0.176 / 0.271 | -0.9 % | +0.9 % | -1.9 / -1.4 / -1.4 / +0.7 % | 0.178 / 0.270 |
| Traffic | DLinear | 2 / 2 | 0.673 / 0.419 | 0.585 / 0.370 | -13.0 % | -11.6 % | -18.8 / -12.3 / -11.5 / -9.1 % | 0.625 / 0.383 |
| Traffic | PatchTST | 2 / 3 | 0.523 / 0.351 | 0.594 / 0.355 | +13.6 % | +1.1 % | +12.5 / +14.5 / +13.7 / +13.8 % | 0.481 / 0.304 |
| Traffic | iTransformer | 2 / 2 | 0.422 / 0.282 | 0.470 / 0.301 | +11.5 % | +6.6 % | +8.7 / +10.9 / +13.2 / +12.8 % | 0.428 / 0.282 |

- **The gain shrinks as the backbone gets stronger.** DLinear gains 10–13 % on both datasets and PatchTST 8 %
  on ECL. iTransformer on ECL is unchanged: MSE −0.9 %, MAE +0.9 %, and at horizon 720 the two seeds of the
  backbone alone differ by 8 %.
- **On Traffic the branch hurts both strong backbones.** PatchTST loses 13.6 % MSE and iTransformer 11.5 %, at
  every horizon and in every seed.
  - For PatchTST the MAE hardly moves (+1.1 %), so the damage comes from few large errors.
  - iTransformer + DLT-dyn (0.470) lies between iTransformer alone (0.422) and stand-alone DLT-dyn (0.529).
  - PatchTST + DLT-dyn (0.594) is worse than both of its parts (0.523 and 0.529).
- **Seed noise is small against these effects.** The backbone alone varies by less than 0.4 % between seeds
  (standard deviation over mean), except ECL iTransformer at horizons 192–720 (1.2–5.4 %). The + DLT-dyn arms
  vary by at most 2.5 %.
- **Strength of the baselines.**
  - iTransformer alone reproduces its published numbers (0.178 / 0.268 on ECL, 0.422 / 0.282 on Traffic).
  - PatchTST (configuration of its paper, training budget of the TSLib script) and DLinear are 6–12 % weaker than
    published.
  - PatchTST + DLT-dyn on ECL (0.211) therefore does not reach the published PatchTST alone (0.205). The 8 % gain
    is real against our re-run, but it is not evidence against a well-tuned PatchTST.

## 3. Can the queue be stopped?

Yes.
- Every conclusion above is fixed by the two complete seeds. In the five clear combinations the effects are
  8–19 %, the seed spread is at most 2.5 %, and the sign is the same in every horizon × seed cell.
- The 32 missing runs repeat the same configurations with seed 2023 and cannot change a sign. The one undecided
  cell, ECL iTransformer, is "no effect" either way.
- The + DLT-dyn arms on Traffic have to be re-run with a changed recipe anyway (section 4), so a third seed of the
  current recipe is wasted GPU time. The arms of the backbone alone stay valid and reusable.
- Nothing is lost by stopping: the queue skips finished runs when it is started again.

## 4. Why the branch hurts on Traffic: hypothesis and checks

Not established yet; the lines above carry no training information.

**Hypothesis: the training recipe, not the branch.**
- The branch trains at learning rate 3e-3, against 1e-4 for PatchTST and 1e-3 for iTransformer.
- On Traffic it takes 3046 steps per epoch with PatchTST (batch 4) and 761 with iTransformer (batch 16). The
  stand-alone model and the DLinear plug-in, which both work, take 380 (batch 32).
- A branch that fits fast and then overfits stops the improvement of the validation loss, so early stopping
  (patience 3) ends the run before the backbone has converged.

**What the CPU runs already show** (`python scripts/p1/plugin_diag.py`). Where the plug-in hurt, the
+ DLT-dyn arm stopped earlier and its best validation loss was worse than that of the backbone alone:

| | best epoch, alone → + DLT-dyn | best validation loss, alone → + DLT-dyn | Δ test |
|---|---|---|---:|
| ETTh1, DLinear | 6.7 → 1.7 | 0.671 → 0.709 | +2.2 % |
| ETTh1, iTransformer | 4.3 → 3.0 | 0.672 → 0.732 | +2.9 % |
| PEMS08-12, iTransformer, all neighbours | 8.7 → 4.0 | 17.12 → 17.71 | +3.7 % |
| Weather, iTransformer (helps) | 3.7 → 5.3 | 0.426 → 0.400 | −6.7 % |
| PEMS08-12, iTransformer, top-8 (helps) | 8.7 → 8.7 | 17.12 → 16.36 | −3.6 % |

In these cases the harm is visible on the validation split. Choosing between "backbone alone" and "backbone +
DLT-dyn" by validation loss would have avoided it.

**Checks, in this order.**
1. `python scripts/p1/plugin_diag.py` with the GPU records in `work/`: epochs, best epoch and best validation loss
   of the Traffic arms. No new runs.
2. Six runs, Traffic at horizon 96, seed 2021, PatchTST and iTransformer: the branch with the learning rate of
   the backbone, with a static gate (it starts at 4.7 % strength), and with both.
   ```bash
   python scripts/exps.py plugin_probe --seeds 2021 --device gpu
   bash scripts/gpu/run_gpu_queue.sh plugin_probe
   python scripts/p1/plugin_diag.py
   ```
3. If a variant removes the loss on Traffic, re-run the + DLT-dyn arms with it (ECL and Weather too, to see that
   their gains stay). If none does, the plug-in claim has to be limited to channel-independent or linear
   backbones, or the branch has to be selected by validation loss.

## 5. Consequences

- **Supported:** the cross-variable branch improves linear models on every coupled dataset, now including Traffic
  (−13 to −20 %); as a plug-in it improves DLinear everywhere it was tried and PatchTST on Weather and ECL.
- **Not supported:** "DLT-dyn improves any backbone". It does nothing for iTransformer on ECL and, with the
  current recipe, damages PatchTST and iTransformer on Traffic.
- **Stand-alone accuracy** on the two large datasets is far from the state of the art (ECL 0.197 against 0.170–0.178,
  Traffic 0.529 against 0.42–0.43). A validation search of k, R and the learning rate will not close gaps of this
  size, so GPU tier 2 is worth running for PEMS only.
- **Still worth the GPU:** the six probe runs, the efficiency table (tier 0), and the PEMS runs of tier 1 once
  their failure is understood.
