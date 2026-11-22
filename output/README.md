# P1 / P2 results: conclusions

**Setting.**
- Time-Series-Library on CPU, input 96, 3 seeds (2021–2023). All P1/P2 runs use code frozen at commit e232fba.
- Default DLT-dyn: k = 8 CCF-selected neighbours, L = 48, R = 4, no gate; Adam 3e-3 halved every epoch,
  batch 32, 10 epochs, early stopping patience 3.

**Sources.**
- Tables: [`statistics/summary_p1.md`](statistics/summary_p1.md) and [`statistics/summary_p2.md`](statistics/summary_p2.md).
- P0 (kernel and gate study): [`ablation/README.md`](ablation/README.md).
- Experiments that still need a GPU: [README.md, GPU Runs](../README.md#gpu-runs). First GPU results (Traffic stand-alone,
  plug-in on Electricity and Traffic): [`plugin/README_gpu_tier1.md`](plugin/README_gpu_tier1.md).
- Numbers marked *reported* are copied from the papers, never re-computed.

## 1. Stand-alone accuracy under the standard protocols

**PEMS04 / PEMS08, horizon 12 (TimeMixer protocol, MAE / MAPE % / RMSE):**

| | PEMS04 | PEMS08 |
|---|---|---|
| DLT-dyn | 22.31 / 13.85 / 35.10 | 17.87 / 10.68 / 28.30 |
| CI linear (same model, no cross branch) | 27.36 / 16.34 / 43.54 | 21.84 / 12.62 / 35.01 |
| DLinear *reported* | 24.62 / 16.12 / 39.51 | 20.26 / 12.09 / 32.38 |
| iTransformer *reported* | 20.16 / 12.40 / 31.64 | 16.01 / 10.05 / 25.13 |
| MillGNN *reported* | 18.93 / 12.24 / 30.72 | 14.86 / 9.54 / 23.46 |

**Long-term benchmark, MSE averaged over the four horizons:**

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | Weather | ECL |
|---|---|---|---|---|---|---|---|
| DLT-dyn (default) | 0.467 | 0.424 | 0.402 | 0.287 | 0.538 | **0.249** | **0.197** |
| DLT-dyn (validation search) | 0.451 | 0.395 | 0.403 | 0.284 | 0.482 | 0.250 | – |
| CI linear (ours) | 0.448 | 0.377 | 0.413 | 0.286 | 0.353 | 0.272 | 0.215 |
| MillGNN *reported* | 0.431 | 0.374 | 0.373 | 0.272 | 0.344 | 0.240 | 0.170 |
| iTransformer *reported* | 0.454 | 0.383 | 0.407 | 0.288 | 0.360 | 0.258 | 0.178 |
| PatchTST *reported* | 0.469 | 0.387 | 0.387 | 0.281 | 0.367 | 0.259 | 0.205 |

Bold marks the datasets where DLT-dyn beats iTransformer and PatchTST.

As a stand-alone linear model, DLT-dyn is competitive only where the variables are coupled:
- On Weather it beats iTransformer, PatchTST and MoVE (0.253). It is behind MillGNN and TimeMixer (0.240).
- On Electricity it beats PatchTST and DLinear, but is behind iTransformer and MillGNN.
- It is behind on the ETT datasets and far behind on Exchange.
- The 20-trial validation search helps on ETTh1 (−3.4 %), ETTh2 (−6.8 %) and Exchange (−10 %), and changes
  nothing elsewhere. The validation rank predicts the test rank only weakly (Spearman 0.29–0.72).

DLT-dyn is therefore not a state-of-the-art forecaster on its own. Its case rests on points 2–6 below.

## 2. Where the cross-variable branch helps

Test MSE of DLT-dyn relative to the same model without the cross branch (CI linear), per horizon:

| | 96 | 192 | 336 | 720 |
|---|---:|---:|---:|---:|
| Weather | −15.7 % | −11.4 % | −7.5 % | −3.5 % |
| Electricity | −13.2 % | −8.8 % | −7.0 % | −6.0 % |
| PEMS08 (P0) | −55 % | | | |
| ETTm1 | −6.9 % | −3.7 % | −1.8 % | +0.5 % |
| ETTm2 | −1.8 % | +0.5 % | +0.2 % | +1.8 % |
| ETTh1 | +5.5 % | +2.3 % | +2.4 % | +7.0 % |
| ETTh2 | +17.0 % | +12.9 % | +10.8 % | +11.1 % |
| Exchange | +34.0 % | +16.4 % | +30.6 % | +69.7 % |

- The gain shrinks with the horizon.
- The branch hurts on the weakly coupled ETTh and Exchange data.
- CI alone matches the reported MillGNN on Exchange (0.082 vs 0.083 at 96, 0.838 vs 0.811 at 720). The
  Exchange failure is therefore caused by the cross branch fitting spurious lagged correlations; the validation
  search reduces it but does not remove it.
- A gate or regulariser that can switch the branch off for such data is the open method problem. The MoVE-style
  softmax gate and a static gate did not solve it in P0.

## 3. Plug-in: backbone alone vs backbone + DLT-dyn (same budget, 3 seeds)

| | DLinear | PatchTST | iTransformer |
|---|---:|---:|---:|
| Weather-96 (MSE) | −17.3 % (p < 0.001) | −9.1 % (p = 0.002) | −6.7 % (p = 0.004) |
| PEMS08-12 (MAE) | −26.7 % (p < 0.001) | GPU only | −3.6 % (p = 0.004) |
| ETTh1-96 (MSE) | +2.2 % (p = 0.016) | −2.6 % (n.s.) | +2.9 % (p = 0.047) |

- On these three datasets the branch improves every backbone where the data is coupled. On ETTh1 it costs
  2–3 %, consistent with point 2.
- **Large data (GPU, two seeds, [`plugin/README_gpu_tier1.md`](plugin/README_gpu_tier1.md)) qualifies this.** Averaged over four horizons:

  | | DLinear | PatchTST | iTransformer |
  |---|---:|---:|---:|
  | Electricity (MSE) | −10.3 % | −8.2 % | −0.9 % |
  | Traffic (MSE) | −13.0 % | +13.6 % | +11.5 % |

  The gain shrinks as the backbone gets stronger, and on Traffic the branch damages PatchTST and iTransformer
  with the current training recipe. Where the plug-in hurt on the CPU, its best validation loss was worse than
  that of the backbone alone, so the harm can be seen before the test split is touched.

## 4. Neighbours (P2 study and follow-up)

- **More neighbours help at horizon 96 on coupled data.**
  - k = 16 vs 8: PEMS08-96 −4.4 %, Weather −2.4 %.
  - ETTh1 is best with k = 4.
- **The CCF ranking is not an accuracy contribution.**
  - Weather: CCF top-8 beats 8 random sources.
  - PEMS08-96: 8 random sources beat the CCF top-8 for DLT (0.763 vs 0.858) and DLT-dyn (0.509 vs 0.523).
  - CCF selection is a way to bound the cost.
- **All other variables (dense path):**
  - Large gains at PEMS08-96: DLT −40 %, DLT-dyn −19 %.
  - At horizon 12 the picture is mixed:
    - static DLT: −4.8 % on PEMS04, −6.5 % on PEMS08;
    - DLT-dyn: −2.8 % on PEMS04, but +2.1 % on PEMS08;
    - iTransformer + DLT-dyn on PEMS08: +7.6 %.
  - On Electricity the static DLT overfits: its best epoch is 1 and it does no better than CI.
  - Weather is within 1 %.
  - The neighbour count therefore has to be chosen on validation; it is in the search space.

## 5. Sensitivity (P2)

- The maximum lag L hardly matters: 12–96 within 3 %.
- More basis functions help coupled data: R = 8 vs 4 gives PEMS08-96 −4.9 % and Weather −1.3 %. R = 2 costs
  +11 % on PEMS08-96.
- The defaults (R = 4, k = 8) are therefore not the best on coupled data.

## 6. Efficiency (CPU, 4 threads, batch 32, H = 96)

| Traffic (862 variables) | parameters | train time per batch | peak memory |
|---|---:|---:|---:|
| DLT-dyn | 49 K | 0.49 s | 1.0 GB |
| iTransformer | 6.4 M | 18.4 s | 12.4 GB |
| iTransformer + DLT-dyn | 6.5 M | 19.8 s | 12.9 GB |

- The plug-in adds about 8 % time and 0.5 GB to iTransformer.
- PatchTST (TSLib configuration) on Electricity and TimeMixer on Traffic exceeded the 13 GB memory limit.
- The GPU table, which is the one to compare with MillGNN's 9.7 GB, is GPU tier 0.

## 7. Learned kernels vs the PEMS08 road graph (P2)

- **Selected sources are near on the road graph.**
  - 1 hop away: 12.3 % of the CCF-selected sources vs 1.9 % of all pairs.
  - 2 hops away: 11.5 % vs 3.7 %.
  - 59 % of the selected sources are upstream of their target.
- **Influence is lagged:** 86–87 % of the kernel mass lies beyond lag 0.
- **The lag does not grow with road distance.**
  - Spearman correlation of distance with the peak lag: −0.20 for DLT, −0.05 for DLT-dyn.
  - With the lag centroid: −0.32 for DLT, −0.20 for DLT-dyn.
  - The kernels also show steps at the basis onsets.
- The interpretability claim must therefore stop at "the selected sources are road-near and act with a lag".
  "The kernels recover travel times" is not supported.

## 8. Baselines re-run on CPU (official hyper-parameters, 277 runs)

**The representative set.** At the time every comparison used eight baselines, one per model family: Informer,
TimesNet, DLinear, PatchTST, iTransformer, TimeMixer, TimeXer and S-Mamba (`BASELINES` of `scripts/baselines.py` then;
the six baselines of today are in [`../src/configs/baseline_profiles.py`](../src/configs/baseline_profiles.py)). The CPU covered DLinear, iTransformer and TimeMixer on the small
datasets and S-Mamba on ETTh1, ETTh2, Exchange and Weather-96; the rest is GPU tier 3. LagTS stays available but is
no longer part of the default set.

**S-Mamba port.** `src/model/S_Mamba.py` re-implements S-Mamba (not part of Time-Series-Library) after the official
code; these runs use its pure-PyTorch Mamba block. With the official hyper-parameters it reproduces the published
numbers (MSE, one seed):

| | 96 | 192 | 336 | 720 | average |
|---|---:|---:|---:|---:|---:|
| ETTh1 | +0.7 % | +0.4 % | +0.4 % | +1.8 % | +0.9 % |
| ETTh2 | +1.6 % | +0.6 % | −0.5 % | −0.2 % | +0.3 % |
| Exchange | +4.2 % | −0.3 % | −0.6 % | −1.1 % | −0.6 % |
| Weather | +1.1 % | | | | |

Change against the S-Mamba paper (Wang et al., Neurocomputing 2025, Tables 3 and 4).

**Re-run vs reported.** MSE averaged over the four horizons; the reported numbers are those cited by MillGNN
(S-Mamba: its own paper; LagTS: its own paper).

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | Weather | ECL |
|---|---|---|---|---|---|---|---|
| DLinear | +1.3 % | +0.7 % | +0.1 % | −0.9 % | −3.2 % | 0.0 % | +6.3 % |
| iTransformer | +0.3 % | −0.1 % | +0.2 % | +1.5 % | +2.0 % | GPU | GPU |
| TimeMixer | +0.6 % | +6.5 % | +0.8 % | +0.7 % | −1.9 % | GPU | GPU |
| S-Mamba (our port, see above) | +0.9 % | +0.3 % | GPU | GPU | −0.6 % | +1.1 % (96 only) | GPU |
| LagTS (our re-implementation) | +5.0 % | +11.1 % | – | – | +23.9 % | GPU | – |

- The pipeline reproduces the published DLinear, iTransformer and TimeMixer numbers within about 2 % in most
  cells. The exceptions are DLinear on ECL and TimeMixer on ETTh2 (about 6 %).
- **LagTS re-implementation.** No code is available, and we fix λ at 0.1 where the paper tunes it. It stays 5–24 % behind the LagTS paper. On
  ETTm1 / ETTm2, which the paper does not cover, it is still improving at the last epoch on ETTm1. Compare with LagTS through its reported
  numbers (now in [`../assets/reported.json`](../assets/reported.json)). The re-implementation is
  supplementary at most.

**Horizon-averaged MSE of the models trained in this pipeline:**

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | Weather | ECL |
|---|---|---|---|---|---|---|---|
| DLT-dyn (default / tuned) | 0.467 / 0.451 | 0.424 / 0.395 | 0.402 / 0.403 | 0.287 / 0.284 | 0.538 / 0.482 | **0.249** | **0.197** |
| CI linear | **0.448** | **0.377** | 0.413 | 0.286 | 0.353 | 0.272 | 0.215 |
| DLinear | 0.462 | 0.563 | 0.403 | 0.347 | **0.343** | 0.265 | 0.225 |
| iTransformer | 0.455 | 0.382 | 0.408 | 0.292 | 0.367 | GPU | GPU |
| TimeMixer | 0.450 | 0.388 | **0.384** | **0.277** | 0.385 | GPU | GPU |
| S-Mamba | 0.459 | 0.382 | GPU | GPU | 0.365 | GPU | GPU |

On Weather and ECL only CI and DLinear were re-run here. The GPU runs are needed before claiming the best
result among re-run models there.

## Consequences for the paper

1. **Lead with:**
   - the controlled kernel study (P0: DLT-dyn beats the MillGNN-style kernels by 5–43 % on coupled data);
   - the gain of the cross branch for linear models on every coupled dataset, Traffic included (−13 to −20 %);
   - the efficiency (point 6).

   The plug-in can be a lead claim only for linear and channel-independent backbones, and only after the Traffic
   result of point 3 is understood ([`plugin/README_gpu_tier1.md`](plugin/README_gpu_tier1.md), section 4).
2. **State the scope:** the branch helps coupled data and must be switched off or regularised for weakly
   coupled data (point 2). Report the Exchange and ETTh results honestly.
3. **Do not claim** CCF neighbour selection as an accuracy contribution (point 4) or travel-time
   interpretability (point 7).
4. **Next steps:**
   - a validation-driven gate or regulariser for weakly coupled data;
   - k and R in the search space;
   - the GPU tiers of [README.md, GPU Runs](../README.md#gpu-runs).
