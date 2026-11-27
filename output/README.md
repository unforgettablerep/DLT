# Results

Every run of the study uses one protocol, that of the official iTransformer / S-Mamba loaders
(`src/configs/dlt_profiles.py`, `protocol()`): four hourly time features for every dataset, the last incomplete
training / validation batch dropped, test batch 1, no time marks on PEMS and Solar, and the PEMS split of the official
loader (validation and test windows take their look-back inside their own split). The results of the earlier
TSLib-default protocol and of the TimeMixer PEMS protocol were removed on 2026-11-26 (the git history keeps them); the
studies were re-run with this protocol on 2026-11-26 and 27: the CPU part is complete (below); the GPU part waits for
`scripts/batch/make_gpu_jobs.sh`.

| Study | Entry point | Store | Device | Status |
| --- | --- | --- | --- | --- |
| Baselines alone, official settings | `scripts/benchmark/make_jobs.py baselines` | `benchmark/baselines*.tsv` | GPU | done except 28 settings ([`baseline_check.md`](benchmark/baseline_check.md)) |
| Stand-alone DLT-dyn: Solar, PEMS | `multivariate_forecasting/make_jobs.py long_extra` | `long_term_forecast/long_extra.tsv` | GPU | done (PEMS08 seed 2021 left) |
| Stand-alone DLT-dyn: ETT, Exchange, Weather; CI | `long_small`; `ablation/make_jobs.py ci_long` | `long_small.tsv`, `ablation/ci_long.tsv` | CPU | done (stage 1, [`summary.md`](long_term_forecast/summary.md)) |
| Stand-alone DLT-dyn and CI: Electricity, Traffic | `long_large`; `ci_long` | `long_large.tsv`, `ci_long.tsv` | GPU | `make_gpu_jobs.sh` |
| Guards of the stand-alone model | `ablation/make_jobs.py guard` | `ablation/guard.tsv` | CPU | done (stage 2, [`guard.md`](ablation/guard.md)) |
| P0 kernel and gate study | `ablation/make_jobs.py p0` | `ablation/p0.tsv` | CPU (ECL: GPU) | done on the CPU (stage 3, [`summary.md`](ablation/summary.md)); ECL: `p0_ecl` |
| Sensitivity, neighbour selection | `hyperparam/make_jobs.py sensitivity \| neighbours` | `hyperparam/*.tsv` | CPU | done (stage 4, [`sensitivity.md`](hyperparam/sensitivity.md)) |
| Validation search and searched configurations | `hyperparam/hpsearch.py make \| tuned` | `hyperparam/hp.tsv`, `tuned.tsv` | CPU (ECL, Traffic, PEMS: GPU) | done on the CPU (stage 5, [`hp_select.md`](hyperparam/hp_select.md)) |
| All variables as neighbours; lead-lag diagnostic | `ablation/make_jobs.py allnb`; `plugin/leadlag_diag.py` | `ablation/allnb.tsv`, `plugin/leadlag_diag.json` | CPU (ECL, Traffic: GPU) | done on the CPU (stage 6, [`allnb.md`](ablation/allnb.md)); ECL, Traffic: `allnb_large` |
| Learned kernels on PEMS08 | `mechanism/kernels.py` | `mechanism/kernels_pems08.json` | CPU | done (stage 3) |
| Fusions with the six backbones; plug-in benchmark | `plugin/make_jobs.py plugin_fusion \| plugin_all` | `plugin/*.tsv` | GPU | `make_gpu_jobs.sh` |
| Efficiency (time, memory) | `efficiency/run_efficiency.py`, `eff_tune.py` | `efficiency/` | CPU, GPU | done |

## Summary of the CPU re-run

984 CPU runs (stages 1 to 6 of `scripts/batch/run_cpu_campaign.sh` and the CCF counterparts of the all-neighbour
suite), none failed. Every conclusion drawn under the removed protocol holds under this one:

- DLT-dyn helps where the variables are coupled (Weather −9.6 % against CI, PEMS08 −43 % to −55 %) and hurts where
  they are not (ETTh1 +4.6 %, ETTh2 +12.6 %, Exchange +37.8 %).
- Two-stage training with significant per-variable selection keeps every ETT and Exchange setting within +1.7 % of CI
  and keeps the coupled gains; the validation search does not (ETTh2 +9.6 %, Exchange +22.2 % against CI).
- DLT-dyn is the best of thirteen cross-variable kernels on Weather and PEMS08; the CCF ranking of the neighbours
  is no accuracy gain; the maximum lag hardly matters.
- The numbers moved little. The MSE of a stand-alone setting moved by at most 2.4 %; the changes against the
  reference by at most 1.7 points in P0, the sensitivity study and the all-neighbour suite, and by up to 5 points for
  the unguarded arms of the guard study on ETTh2-720 and Exchange, whose spread over the seeds is large (two-stage
  training without selection on Exchange-720: +85.5 % then, +80.4 % now). The validation search chose other
  configurations on three datasets, whose best trials lead by less than 1 % of the validation loss.

The conclusions of each study follow.

## Stand-alone DLT-dyn against CI (stage 1, CPU, seeds 2021-2023)

Table: [`long_term_forecast/summary.md`](long_term_forecast/summary.md) (`scripts/multivariate_forecasting/long_table.py`).
CI is the same model without the cross-variable branch. Change of the test MSE of DLT-dyn against CI, mean over the
four horizons (96, 192, 336, 720):

| ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | Weather |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +4.6 % | +12.6 % | −2.8 % | −0.0 % | +37.8 % | −9.6 % |

1. The branch helps on Weather at every horizon (−16.2 % at 96 to −3.4 % at 720, p ≤ 0.002) and on ETTm1 at the short
   horizons (−6.2 % at 96, −4.1 % at 192, p ≤ 0.01); on ETTm2 it changes nothing (−1.7 % to +1.3 %, p ≥ 0.08).
2. It hurts on ETTh1 (+2.7 % to +6.6 %), ETTh2 (+8.1 % to +15.8 %) and Exchange (+16.8 % to +69.0 %), with a much
   larger spread over the seeds than CI (Exchange-720: ± 0.186 against ± 0.005).
3. The protocol does not change these results: against the runs of the removed TSLib-default protocol (commit
   `343b23f`), the MSE of a setting moved by at most 2.4 % (DLT-dyn, ETTh2-720) and the mean change per dataset by at
   most 0.4 %; findings 1 and 2 hold under both protocols.

## Guards of the stand-alone model (stage 2, CPU, seeds 2021-2023)

Table: [`ablation/guard.md`](ablation/guard.md) (`scripts/ablation/guard_table.py`). Stand-alone DLT-dyn with the guards
of the plug-in: per-variable selection (`--dlt_select variable`: keep the correction of a variable where it lowers its
validation MSE; `significant`: only where it does so consistently over blocks of pred_len validation windows, one-sided
t-test, never with fewer than three blocks), two-stage training (`--dlt_train decoupled`: the self head first, then the
branch on its error), a slower branch (`--dlt_lr`), a last-value anchor and a difference input. Change of the test MSE
against CI:

| | ETTh1-96 | ETTh2-96 | ETTh2-720 | ETTm1-96 | ETTm2-96 | Exchange-96 | Exchange-720 | PEMS08-96 | Weather-96 | ETT + Exchange: mean / worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DLT-dyn | +5.8 % | +15.8 % | +8.1 % | −6.2 % | −1.7 % | +34.6 % | +69.0 % | −55.3 % | −16.2 % | +17.9 % / +69.0 % |
| two-stage + selection | +0.3 % | −0.2 % | +1.7 % | −4.0 % | −1.4 % | +4.4 % | +32.8 % | −58.5 % | −13.5 % | +4.8 % / +32.8 % |
| two-stage + significant selection | +0.3 % | −0.2 % | −0.6 % | −3.9 % | −1.1 % | +1.7 % | +0.7 % | −57.9 % | −9.3 % | −0.4 % / +1.7 % |
| the same at level 0.25 | +0.3 % | −0.2 % | −0.6 % | −4.0 % | −1.5 % | +3.0 % | +0.7 % | −58.3 % | −13.5 % | −0.3 % / +3.0 % |
| last-value anchor + two-stage + sign. selection | +1.0 % | +0.5 % | −0.0 % | −4.9 % | −0.6 % | +0.4 % | −1.4 % | −56.9 % | −7.9 % | −0.7 % / +1.0 % |
| difference input + two-stage + sign. selection | +0.3 % | −0.2 % | −0.6 % | −2.9 % | −1.2 % | +1.7 % | +0.7 % | −55.5 % | −8.6 % | −0.3 % / +1.7 % |

1. **Two-stage training with significant selection is the stable guard.** At level 0.05 it brings all seven ETT and
   Exchange settings to within +1.7 % of CI or better (ETTm1 −3.9 %), against +5.8 % to +69.0 % for DLT-dyn, and keeps
   the coupled gains: PEMS08 −57.9 % (better than DLT-dyn, p 0.050), Weather −9.3 % (65 % of the variables kept). On
   ETTh1, ETTh2 and Exchange it keeps no correction at all; the small differences to CI there (Exchange-96 +1.7 %)
   come from the self head of stage 1, not from the branch.
2. **The level trades Weather against Exchange.** At 0.25 the test keeps 89 % of the Weather variables and Weather
   gains −13.5 % (p 0.14 against DLT-dyn's −16.2 %); it then keeps 8 % of the Exchange-96 variables, at +3.0 %.
3. **Plain selection does not repair Exchange-720** (+32.8 %, 58 % of the variables kept): validation and test
   periods disagree there. The significant selection keeps nothing: the validation split of Exchange-720 has 41
   windows, not one block of 720 (+0.7 %).
4. **What did not help:** the last-value anchor gives the smallest worst case on the weak data (+1.0 %) but the
   smallest Weather gain (−7.9 %); the difference input of the branch helps neither (Weather −8.6 %, PEMS08 −55.5 %). A
   branch at a tenth of the learning rate loses a quarter of the PEMS08 gain (−42.2 %) and still costs +25.7 % on
   ETTh2-96.
5. The arms `dyn` and `ci` reproduce the 48 runs of stage 1 they share to the last digit, so the guards leave the default path of the model
   unchanged. Against the removed protocol the conclusions are the same (significant selection: worst case +1.0 %
   then, +1.7 % now; Weather −8.8 % then, −9.3 % now).

## P0: cross-variable kernels and gates (stage 3, CPU, seeds 2021-2023)

Table: [`ablation/summary.md`](ablation/summary.md) (`scripts/ablation/aggregate.py`). Thirteen configurations of the
same model (self head + cross branch, k = 8 CCF neighbours, L = 48) on ETTh1-96, Weather-96, PEMS08-24 and PEMS08-96;
Electricity-96 is the GPU list `p0_ecl`. Change of the test MSE against CI:

| configuration | ETTh1-96 | Weather-96 | PEMS08-24 | PEMS08-96 |
|---|---:|---:|---:|---:|
| scalar (lag 0) | +2.6 % | −10.9 % | −18.5 % | −13.2 % |
| tap3 (CrossLinear-style) | +1.3 % | −11.6 % | −20.9 % | −18.5 % |
| full (free kernel, lags 0..48) | +2.9 % | −13.7 % | −23.6 % | −27.5 % |
| MillGNN-static / MillGNN-dyn | +1.0 % / +1.7 % | −12.8 % / −11.7 % | −17.8 % / −20.3 % | −15.0 % / −20.9 % |
| scalar-dyn | +2.7 % | −11.1 % | −36.2 % | −39.5 % |
| DLT (static basis kernel) | +2.2 % | −14.0 % | −23.4 % | −26.8 % |
| **DLT-dyn** | +5.8 % | **−16.2 %** | **−43.1 %** | **−55.3 %** |
| DLT-dyn + MoVE softmax gate / static gate | +3.7 % / +5.2 % | −16.0 % / −15.9 % | −42.6 % / −43.0 % | −53.9 % / −53.3 % |

1. **DLT-dyn is the best kernel wherever the variables are coupled.** On Weather and PEMS08 it beats every other
   kernel, by 2.4 % to 6.7 % on Weather and 12.2 % to 98.0 % on PEMS08 (the six Welch t-tests of the summary,
   3 against 3 seeds: p ≤ 0.02); against the MillGNN-style kernels the margin is 4.0 to 5.3 % on Weather and 40 to
   90 % on PEMS08. The input-adapted
   weights carry most of it: the static DLT kernel gains −26.8 % on PEMS08-96, DLT-dyn −55.3 %.
2. **On ETTh1 every cross kernel loses to CI** (+1.0 % to +5.8 %), DLT-dyn the most; its best validation epoch is
   1.3 on average against 5.3 for CI. No gate repairs this (MoVE softmax gate +3.7 %, static gate +5.2 %), and the
   gates keep the coupled gains (PEMS08-96 −53.9 % and −53.3 % against −55.3 %); the guards of stage 2 repair it.
3. The protocol does not change the study. On Weather and PEMS08 every change against CI is within 0.7 points of
   the runs under the removed protocol. On ETTh1 CI is 0.9 % better now (0.3853 against 0.3887), which raises the
   changes of the cross kernels by 0.3 to 1.6 points (largest: MoVE gate, +2.0 % then, +3.7 % now). DLT-dyn's
   margins over the other kernels moved by less than 1 point everywhere.

## Learned kernels against the PEMS08 road graph (stage 3)

`output/mechanism/kernels_pems08.json` (`scripts/mechanism/kernels.py`, the P0 checkpoints of DLT and DLT-dyn at
H = 96, three seeds; the figure goes to the paper repository). The statistics equal those of the removed protocol
to the second decimal:

- The CCF-selected sources are near on the road graph: 12.3 % are one hop from their target (1.9 % of all pairs),
  11.5 % two hops (3.7 %); 59 % are upstream of their target.
- The influence is lagged: 86 % (DLT) and 87 % (DLT-dyn) of the kernel mass lies beyond lag 0.
- The lag does not grow with road distance: Spearman correlation of distance and peak lag −0.20 (DLT) and −0.05
  (DLT-dyn), of distance and lag centroid −0.32 and −0.20. The kernels show road-near sources acting with a lag,
  not travel times.

## Sensitivity and neighbour selection (stage 4, CPU, seeds 2021-2023)

Table: [`hyperparam/sensitivity.md`](hyperparam/sensitivity.md) (`scripts/hyperparam/sens_table.py`). DLT-dyn with one
of L, R and k changed (defaults L 48, R 4, k 8: the P0 runs), and DLT and DLT-dyn with 8 random or all other variables
as neighbours instead of the CCF top-8. Change of the test MSE against the default:

| | ETTh1-96 | Weather-96 | PEMS08-96 |
|---|---:|---:|---:|
| L 12 / 24 / 96 | −0.4 / +0.1 / −0.3 % | +0.5 / +0.1 / −0.2 % | +2.8 / +1.5 / +0.5 % |
| R 2 / 8 | −1.6 / −0.5 % | +1.6 / −0.8 % | +11.5 / −5.1 % |
| k 4 / 16 | −1.7 % / (k is capped at 6) | +2.6 / −2.2 % | +11.8 / −4.7 % |
| DLT-dyn: 8 random / all other variables as neighbours | | +0.7 / −1.4 % | −2.3 / −18.7 % |
| DLT: 8 random / all other variables as neighbours | | +2.1 / −2.0 % | −11.2 / −40.7 % |

1. **The maximum lag hardly matters:** L from 12 to 96 changes the MSE by −0.4 % to +2.8 % (p ≥ 0.19).
2. **More basis functions and more neighbours help on PEMS08:** R 2 costs +11.5 % (p 0.044) and k 4 +11.8 %
   (p 0.001); R 8 gains −5.1 % (p 0.08) and k 16 −4.7 % (p 0.15). Weather moves the same way by less (R 2 +1.6 %,
   k 4 +2.6 %, p ≤ 0.013). On ETTh1 nothing is significant (p ≥ 0.13).
3. **The CCF selection is no accuracy gain.** All other variables as neighbours beat the CCF top-8 on PEMS08
   (DLT-dyn −18.7 %, p 0.001; DLT −40.7 %) and on Weather (DLT −2.0 %, p 0.002; DLT-dyn −1.4 %, p 0.33). Eight
   random neighbours are as good as the CCF top-8 for DLT-dyn (+0.7 % and −2.3 %, p ≥ 0.14); only DLT on Weather is
   worse with them (+2.1 %, p 0.021). The gain comes from the cross-variable information, not from the choice of the
   neighbours by cross-correlation.
4. Against the removed protocol every change in this section moved by at most 0.8 points (DLT-dyn with all
   neighbours on PEMS08: −19.5 % then, −18.7 % now).

## Validation search (stage 5, CPU)

20 random trials per dataset at H = 96, seed 2021 (learning rate, batch size, k, L, R, size of the dynamic weights,
gate; `scripts/hyperparam/hpsearch.py`); the trial with the lowest validation loss
([`hyperparam/hp_select.md`](hyperparam/hp_select.md), `tuned.json`) is trained at the four horizons with three seeds
(`tuned.tsv`; column "tuned" of [`long_term_forecast/summary.md`](long_term_forecast/summary.md)). Change of the test
MSE, mean over the four horizons:

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | Weather |
|---|---:|---:|---:|---:|---:|---:|
| searched against default DLT-dyn | −3.2 % | −2.6 % | ±0 (default chosen) | −1.1 % | −10.9 % | +0.5 % |
| searched against CI | +1.3 % | +9.6 % | −2.8 % | −1.1 % | +22.2 % | −9.2 % |

1. **The search helps where the default is worst** (Exchange −10.9 %, ETTh1 −3.2 %, ETTh2 −2.6 %) and changes little
   elsewhere; on Weather it is +2.6 % worse at H = 96 (p < 0.01). It picks a learning rate of 0.01 and a gate on four
   of the six datasets.
2. **It does not replace the guards.** Against CI the searched model is still +9.6 % on ETTh2 and +22.2 % on Exchange
   (Exchange-720 +45.4 %), where two-stage training with significant selection stays within +1.7 % (stage 2).
3. **The choice is fragile.** The best trial leads the second by 0.03 % to 0.70 % of the validation loss (Exchange
   7 %), and the validation rank predicts the test rank only weakly (Spearman 0.26 to 0.73 over the 20 trials).
   Under the removed protocol the search chose other configurations on ETTh2, ETTm1 and Weather; its gains over the
   default were similar (ETTh1 −3.6 %, Exchange −11.3 %) except on ETTh2 (−7.0 % then, −2.6 % now).

## All other variables as neighbours (stage 6, CPU, seeds 2021-2023)

Table: [`ablation/allnb.md`](ablation/allnb.md) (`scripts/ablation/allnb_table.py`). All other variables as neighbours
(dense path) against the CCF top-8, same device, protocol and seeds: stand-alone DLT and DLT-dyn on PEMS04 and PEMS08
at H = 12 and on Weather at the four horizons (DLT-dyn with the CCF top-8 there: `long_small`), and iTransformer +
DLT-dyn on PEMS08-12 (joint training, branch at 3e-3). Change of the test MSE of all neighbours against the CCF top-8:

| | PEMS04-12 | PEMS08-12 | Weather (96 / 192 / 336 / 720) |
|---|---:|---:|---:|
| DLT | −5.7 % | −6.0 % | −2.0 / −1.8 / −1.6 / −1.1 % |
| DLT-dyn | −4.2 % | +3.6 % (p 0.11) | −1.4 / −0.5 / −1.1 / −0.9 % |
| iTransformer + DLT-dyn | | +7.7 % (p 0.07) | |

1. **All neighbours help the static kernel everywhere** (DLT: −5.7 % and −6.0 % on PEMS, −1.1 % to −2.0 % on
   Weather, p ≤ 0.02), as at PEMS08-96 in stage 4 (−40.7 %).
2. **For DLT-dyn the picture is mixed:** −4.2 % on PEMS04-12 (p 0.03), +3.6 % on PEMS08-12, within 1.4 % on Weather
   (significant only at 720, −0.9 %); at PEMS08-96 all neighbours gained −18.7 % (stage 4). Inside iTransformer, all
   neighbours are worse than the CCF top-8 (+7.7 %). The number of neighbours is a setting to choose on validation
   data; it is in the search space of stage 5.
3. **iTransformer + DLT-dyn with the CCF top-8 gains −8.7 % on PEMS08-12** (0.0809 against 0.0886, p < 0.001, three
   seeds). iTransformer alone is 12 % behind its published result there (0.079; the baseline check on the GPU finds
   the same, 0.089), and the plug-in arm reaches about the published number: part of the gain may close a
   reproduction gap rather than add information.
4. Against the removed protocol (TimeMixer PEMS protocol at H = 12 then): static DLT −4.8 % / −6.5 % then, −5.7 % /
   −6.0 % now; DLT-dyn −2.8 % / +2.1 % then, −4.2 % / +3.6 % now; iTransformer + DLT-dyn +7.6 % then, +7.7 % now.

## Lead-lag diagnostic (stage 6)

`output/plugin/leadlag_diag.json` (`scripts/plugin/leadlag_diag.py`): ridge regressions on the data alone, no network
is trained. Change of the test MSE against one shared channel-independent linear forecaster, mean over the four
horizons (training split in brackets). NB: a correction from the windows of the 8 CCF neighbours; BOTH vs OWN: what
the neighbours' windows add beyond the variable's own window.

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | Weather | ECL | Traffic | Solar | PEMS03 | PEMS04 | PEMS07 | PEMS08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NB | +0.6 (−8.6) | +13.4 (−22.0) | −5.2 (−13.6) | −0.9 (−10.7) | +39.8 (−18.9) | −6.0 (−13.8) | −3.0 (−9.6) | −2.5 (−18.9) | −2.3 (−10.4) | −13.6 (−44.2) | −36.6 (−45.3) | −21.2 (−36.1) | −32.3 (−39.8) |
| BOTH vs OWN | −0.5 (−6.1) | +11.9 (−19.9) | −1.6 (−7.8) | +0.7 (−8.3) | +22.1 (−15.7) | +0.3 (−7.8) | −3.2 (−9.7) | −3.2 (−18.4) | −7.0 (−18.7) | −11.2 (−46.5) | −42.8 (−50.5) | −19.8 (−35.6) | −36.8 (−44.5) |

1. **The diagnostic flags the harmful datasets before any network is trained.** On ETTh2 and Exchange the neighbour
   correction lowers the training error (−22.0 %, −18.9 %) and raises the test error (+13.4 %, +39.8 %); these are
   the datasets where DLT-dyn loses most to CI (stage 1: +12.6 %, +37.8 %).
2. **The neighbours' pasts carry the most on PEMS** (BOTH vs OWN −11.2 % to −42.8 %), where DLT-dyn gains most
   (stage 3: PEMS08 −43.1 % and −55.3 %).
3. The statistics are those of the removed protocol on the nine datasets whose splits are the same; on PEMS, whose
   validation and test windows now take their look-back inside their split, they moved by at most 0.7 points.

## What the lags do (CPU, seeds 2021-2023)

Table: [`ablation/lagrole.md`](ablation/lagrole.md) (suite `lagrole` of `scripts/ablation/make_jobs.py`,
`scripts/ablation/lagrole_table.py`). Weather-96 and PEMS08-96: CI, the lag-0 kernel with input-dependent weights
(scalar-dyn) and DLT-dyn with the maximum lag L = 1, 2, 4, 8, 12 and 48. Only L changes: every arm has the same
neighbours (CCF search over lags 0..48, `--dlt_ccf_L 48`), and every run records the test MSE of each forecast step
(`--step_metrics`). A lead of the neighbours by tau steps can only inform the first tau steps of the horizon; a better
estimate of the current state helps every step.

| change of the MSE against lag 0 | L 1 | L 2 | L 4 | L 8 | L 12 | L 48 |
|---|---:|---:|---:|---:|---:|---:|
| Weather-96 | −4.7 % | −4.9 % | −5.6 % | −4.8 % | −5.9 % | −5.7 % |
| share of the L-48 gain | 83 % | 86 % | 99 % | 85 % | 104 % | 100 % |
| PEMS08-96 | −16.3 % | −19.1 % | −22.2 % | −23.2 % | −23.3 % | −26.0 % |
| share of the L-48 gain | 62 % | 73 % | 85 % | 89 % | 89 % | 100 % |

| DLT-dyn (L 48) against lag 0, by forecast step | 1 | 2-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65-96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weather-96 (10-minute steps) | −3.2 % | −4.4 % | −6.7 % | −8.2 % | −7.6 % | −5.6 % | −4.7 % |
| PEMS08-96 (5-minute steps) | +0.9 % | +0.2 % | −1.3 % | −5.8 % | −16.5 % | −25.4 % | −29.1 % |

1. **The lags help, and the first one does most of it.** Every L from 1 up beats lag 0 (p 0.013 to 0.049). One lag,
   the current value and the one before it, gives 83 % (Weather) and 62 % (PEMS08) of the gain of the full kernel;
   four lags give 99 % and 85 %. With one lag the kernel of a pair is a weighted sum of the neighbour's level and of
   its last change, so the gain starts with each neighbour's own recent change, not with a distant lag.
2. **The gain is not where a lead would put it.** On PEMS08 the lags add nothing to the first eight steps (40
   minutes: +0.9 % to −1.3 %) and more with every bin after (−29.1 % at steps 65 to 96); on Weather they help every
   step, most at steps 9 to 32. The lag-0 neighbours behave the same way (CI against lag 0 on PEMS08: +3.1 % at step 1,
   +77.7 % at steps 33 to 64). The neighbours do not announce the target's next steps; their recent trajectories
   tell the model the current state of the system and where it is going, which matters most far ahead.
3. This fits the earlier results: most CCF peaks of the selected pairs are at lag 0 (76 % on Weather, 60 % on PEMS08),
   aligning each neighbour at its CCF peak lag does not help (P0, `shift`), the learned lags do not grow with road
   distance, and inside S-Mamba, whose tokens already carry each variable's own trajectory, the lags add nothing
   (mambadlt).
4. Checks: the arms CI, scalar-dyn and DLT-dyn reproduce the P0 runs exactly (18 runs). The neighbour search of the
   sensitivity study (lags 0..L, so that its neighbours changed with L) moved its L-12 result by less than 1 %
   (Weather 0.1643 against 0.1630 here, PEMS08 0.5244 against 0.5292).
5. Limits: two settings, one horizon (96), the CPU; whether the profile holds at the PEMS horizons 12 to 48 and on
   other data is not tested.

