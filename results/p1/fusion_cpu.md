# How the DLT branch enters a backbone: alternatives to the output sum (2026-11-17)

## Why the current plug-in is crude

DLTPlug so far adds the branch to the backbone's forecast (Eq. 2):

    y_hat = Backbone(X) + sigma_i * W_c z_i,   z_i(t) = sum_{j in N(i)} sum_l K_ij(l) x~_j(t - l),  t = 1..T

Three weaknesses showed in the plug-in study ([plugin_all_gain.md](../gpu/plugin_all_gain.md),
[plugin_all_gate.md](../gpu/plugin_all_gate.md)):
1. **The lag structure stops at the window.** z is computed over the look-back window only; a free linear head W_c
   (T -> H, shared by the variables) maps it to the horizon. Kernel and head together are an unconstrained linear map
   from the neighbours' windows to the future, so the distributed-lag form says nothing about the forecast itself.
2. **Joint training lets the branch drag the backbone.** On ETTh2 and Exchange the + DLT arm reached its best
   validation loss after one or two epochs: the branch fits relations that do not carry over, early stopping halts,
   and the backbone stays undertrained. The data diagnostic showed the same split: the neighbours lower the training
   error by 19-22 % and raise the test error by 13-40 % there.
3. **No use of what the backbone already predicts, and no way to decline.** The branch never sees the backbone's
   forecast, and it acts on every variable even where it only adds noise.

## Alternatives (models/DLTPlug.py)

| option | what changes |
|---|---|
| `--dlt_fusion refine` | the kernels run over each neighbour's observed past followed by the backbone's forecast of it: the correction at horizon step t+h takes the neighbours' observed values at lags >= h and their forecasts at lags < h. No free head; the correction keeps the distributed-lag form in time, starts at zero, and leading neighbours inform the near horizon with observed data. |
| `--dlt_fusion input` | the neighbour signal sigma_i z_i is added to the backbone's input with a per-variable gain that starts at zero; the backbone processes it with its own (non-linear) machinery. |
| `--dlt_train decoupled` | the backbone is trained by its own loss exactly as if it were alone (a unit test checks the gradient), the branch by the loss of backbone + correction with the backbone's forecast held fixed (boosting on the backbone's error). The branch can no longer slow the backbone down. |
| `--dlt_select variable` | after training, the correction is kept only for the variables whose validation MSE it lowers; with decoupled training the backbone part is a trained forecaster of its own, so dropping the correction is safe. |

The same run also scores the backbone part alone and the uncorrected sum (`mse_backbone`, `mse_full`).

## CPU comparison (RLinear and DLinear; ETTh2-96, Exchange-96, Weather-96, PEMS08-12)

Suite `plugin_fusion` of `scripts/exps.py`, run here on the CPU with the settings and protocol of the baseline check
and the default branch recipe (DLT-dyn, k 8, L 48, R 4, branch learning rate 3e-3). Seeds averaged where both ran
(2021, partly 2022). Table: `python scripts/p1/fusion_table.py --files results/p1/results_plugin_fusion.jsonl`.

Change of the test MSE against the backbone alone; brackets for the decoupled arms: without the per-variable selection / the backbone part of the same run.

| dataset | H | backbone | alone | output (Eq. 2) | output, decoupled + select | refine | refine, decoupled + select | input |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ETTh2 | 96 | DLinear | 0.3473 | +6.2 % | +0.7 % (+21.8 % / +0.3 %) | +99.0 % | +0.3 % (+49.4 % / +0.3 %) | +18.7 % |
| ETTh2 | 96 | RLinear | 0.2894 | +25.9 % | +0.1 % (+17.2 % / +0.1 %) | +5.8 % | +0.5 % (+6.8 % / +0.1 %) | +8.8 % |
| Exchange | 96 | DLinear | 0.0952 | +4.3 % | +6.7 % (+16.9 % / -0.8 %) | +32.7 % | +16.9 % (+23.9 % / -0.8 %) | +7.5 % |
| Exchange | 96 | RLinear | 0.0833 | +10.6 % | +4.2 % (+27.6 % / -0.0 %) | +7.2 % | +1.5 % (+11.2 % / -0.0 %) | +6.4 % |
| PEMS08 | 12 | DLinear | 0.1521 | **-43.6 %** | **-40.2 % (-39.7 % / +0.5 %)** | **-37.6 %** | | **-38.0 %** |
| PEMS08 | 12 | RLinear | 0.1223 | **-30.8 %** | **-32.5 % (-32.5 % / -0.1 %)** | **-30.2 %** | **-31.0 % (-31.0 % / -0.1 %)** | **-27.7 %** |
| Weather | 96 | DLinear | 0.1960 | **-15.8 %** | **-13.4 % (-13.4 % / -0.3 %)** | **-2.7 %** | **-6.2 % (-6.1 % / -0.3 %)** | **-16.6 %** |
| Weather | 96 | RLinear | 0.1931 | **-14.7 %** | **-12.1 % (-12.1 % / +0.3 %)** | **-10.4 %** | **-5.2 % (-5.1 % / +0.3 %)** | **-7.3 %** |

| arm | settings | mean change | helps | worst |
|---|---:|---:|---:|---|
| output (Eq. 2) | 8 | -7.2 % | 4 of 8 | +25.9 % (ETTh2-96 RLinear) |
| output, decoupled + select | 8 | -10.8 % | 4 of 8 | +6.7 % (Exchange-96 DLinear) |
| refine | 8 | +8.0 % | 4 of 8 | +99.0 % (ETTh2-96 DLinear) |
| refine, decoupled + select | 7 | -3.3 % | 3 of 7 | +16.9 % (Exchange-96 DLinear) |
| input | 8 | -6.0 % | 4 of 8 | +18.7 % (ETTh2-96 DLinear) |


## What the comparison shows

1. **Output sum (Eq. 2), joint training:** the largest gains where the branch helps (PEMS08 −44 % / −31 %, Weather
   −16 % / −15 % for DLinear / RLinear) and no protection where it does not (ETTh2 +6 % / +26 %, Exchange +4 % /
   +11 %). Mean −7.2 %, worst +25.9 %.
2. **Output sum, two-stage decoupled training, per-variable selection:**
   - It keeps most of the gains (PEMS08 −40 % / −33 %, Weather −13 % / −12 %) and removes the harm on ETTh2
     (+0.7 % / +0.1 %, where the uncorrected sum is +22 % / +17 %).
   - Mean −10.8 %, worst +6.7 %: better than joint training on average and far better in the worst case.
   - On Exchange it still loses 4–7 %, there slightly more than joint training with DLinear. The validation period
     of Exchange disagrees with its test period, so a choice made on validation data keeps corrections that hurt on
     test. The validation rule of the plug-in study failed there for the same reason.
   - The backbone part of these runs is within ±0.8 % of the backbone trained alone: stage 1 trains the backbone
     as if it were alone.
3. **Refine:** as good as the output sum on PEMS08 (RLinear −30 %), where the neighbours carry real information.
   It is the most harmful arm on ETTh2 and Exchange (DLinear +99 % and +33 %) and weaker on Weather (−3 % / −10 %).
   Mixing the neighbours' forecasts into a variable's forecast spreads relations that do not carry over to the test
   period. Decoupled training with selection contains it on ETTh2 but not on Exchange (+17 %). Not recommended in
   this form.
4. **Input:** close to the output sum with DLinear (PEMS08 −38 %, Weather −17 %), weaker with RLinear (Weather −7 %),
   and harmful on ETTh2 and Exchange (+6 % to +19 %). With the branch inside the backbone there is no backbone alone
   to fall back to.

## Recommendation

- **Plug-in recipe:** the output sum with two-stage decoupled training and per-variable selection
  (`--dlt_train decoupled --dlt_select variable`). For the paper: "the branch is fitted to the error of the trained
  backbone and kept for a variable only if it lowers that variable's validation error". The backbone alone is
  always one of the candidates, so the plug-in rarely hurts, and one run gives both the backbone alone and
  backbone + DLT (`mse_backbone`, `mse_full` in the record).
- **Next, on the GPU:** `plugin_fusion` with all six backbones (the strong backbones are where joint training hurt
  most), then `plugin_all` with the chosen recipe (GPU_RUNS.md).

## Limits

- CPU only: two linear backbones, one horizon per dataset, one or two seeds per setting (seed 2022 only in part).
- Exchange: no rule that decides on validation data can repair a validation/test mismatch.
- The refine fusion was changed twice during the comparison: a floor on the window scale, then centring without
  scaling. The table holds only runs of the final code; runs of the earlier versions were discarded.
