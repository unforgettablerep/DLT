# Priority-0 experiments: conclusions

Setting: Time-Series-Library, look-back 96, the same linear self branch and training set-up for every row
(k = 8 CCF-selected neighbours, L = 48, R = 4, top-K = 5, Adam 3e-3 halved every epoch, batch 32,
10 epochs, early stopping patience 3, seeds 2021/2022/2023). 13 configurations × 5 settings × 3 seeds =
195 runs, all complete. Tables: [`summary.md`](summary.md); raw records: [`results_p0.jsonl`](results_p0.jsonl).
Rows "MillGNN-static / -dyn" re-implement MillGNN's lag-kernel parameterisation (top-K CCF lags, signed
excitation–inhibition decay, attention over sources) inside the same linear model. MillGNN has no public
code, so this isolates the kernel, not the full MillGNN architecture (its reported numbers are compared in P1).

## Findings

1. **DLT-dyn is the best kernel wherever there is cross-variable coupling.** Test MSE, DLT-dyn against the
   strongest MillGNN-style kernel (MillGNN-dyn):

   | | Weather-96 | ECL-96 | PEMS08-24 | PEMS08-96 |
   |---|---:|---:|---:|---:|
   | MillGNN-dyn | 0.1724 | 0.1864 | 0.1860 | 0.9219 |
   | DLT-dyn | **0.1637** | **0.1713** | **0.1333** | **0.5229** |
   | change | −5.0 % | −8.1 % | −28.3 % | −43.3 % |

   Welch p ≤ 0.001 in all four settings, also against the free full kernel, the free top-5 lags and static DLT.
   Same ordering in MAE.

2. **ETTh1 is the exception.** Every cross-variable branch is worse than the channel-independent model (CI 0.3887).
   DLT-dyn is the worst (0.4102, +5.5 %); its best epoch is 1.3 on average, so the input-adaptive branch overfits
   the 7 weakly coupled variables. The paper has to state this scope honestly; P1 searches the learning rate
   and gate on the validation split for such data.

3. **Where the gain comes from (relative to the static lag-0 "scalar" branch):**

   | | multi-lag basis (DLT) | input-adaptive, lag 0 (scalar-dyn) | both (DLT-dyn) | product of the two |
   |---|---:|---:|---:|---:|
   | Weather-96 | −3.7 % | −0.3 % | −5.7 % | −4.0 % |
   | ECL-96 | −2.2 % | −4.0 % | −9.9 % | −6.1 % |
   | PEMS08-24 | −5.7 % | −21.4 % | −29.9 % | −25.9 % |
   | PEMS08-96 | −15.1 % | −30.3 % | −48.3 % | −40.8 % |

   The combination is better than the product of the two effects in all four coupled settings: making a
   distributed-lag kernel input-adaptive helps more than making a single-lag weight input-adaptive.

4. **The static shared basis buys parameter efficiency, not accuracy.** DLT matches the free 0..48-lag kernel
   within ±0.6 % MSE with 5× fewer parameters at ECL scale (29.1 K vs 144.5 K, including the 9.3 K self branch).

5. **MillGNN's decay prior hurts.** With the same top-5 lags, the excitation–inhibition decay is worse than free
   weights on PEMS08-96 (+9.2 %) and PEMS08-24 (+5.1 %). Attention over sources recovers part of that loss but
   does not beat free weights on Weather or PEMS08.

6. **Gates do not fix ETTh1 and cost accuracy on PEMS08.**
   - MoVE-style softmax gate: ETTh1 +2.0 % vs CI (instead of +5.5 %), but +3.0 % vs the ungated DLT-dyn on PEMS08-96.
   - Static per-variable gate: ETTh1 +4.7 %, PEMS08-96 +4.3 %.
   - On Weather and ECL all three are within noise.

## Decision for P1

The default model is **DLT-dyn without a gate**, because it:
- is best or tied-best in all four coupled settings,
- has the simpler architecture,
- is more clearly distinct from MoVE.

The gate type (none / static / MoVE-style) remains one dimension of the 20-trial validation search.

## Note on one re-run

The Electricity CI run with seed 2021 was killed by the 13 GB memory limit during testing, because TSLib held the
whole test split in memory. It was re-run after the test metrics were changed to accumulate batch by batch
(commit e232fba). Training is unchanged, and the streamed metrics agree with the array metrics to ~1e-8.
