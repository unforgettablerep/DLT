# Results

Every run of the study uses one protocol, that of the official iTransformer / S-Mamba loaders
(`src/configs/dlt_profiles.py`, `protocol()`): four hourly time features for every dataset, the last incomplete
training / validation batch dropped, test batch 1, no time marks on PEMS and Solar, and the PEMS split of the official
loader (validation and test windows take their look-back inside their own split). The results of the earlier
TSLib-default protocol and of the TimeMixer PEMS protocol were removed on 2026-11-26 (the git history keeps them); the
studies are being re-run with this protocol.

| Study | Entry point | Store | Device | Status |
| --- | --- | --- | --- | --- |
| Baselines alone, official settings | `scripts/benchmark/make_jobs.py baselines` | `benchmark/baselines*.tsv` | GPU | done except 28 settings ([`baseline_check.md`](benchmark/baseline_check.md)) |
| Stand-alone DLT-dyn: Solar, PEMS | `multivariate_forecasting/make_jobs.py long_extra` | `long_term_forecast/long_extra.tsv` | GPU | done (PEMS08 seed 2021 left) |
| Stand-alone DLT-dyn: ETT, Exchange, Weather; CI | `long_small`; `ablation/make_jobs.py ci_long` | `long_small.tsv`, `ablation/ci_long.tsv` | CPU | done (stage 1, [`summary.md`](long_term_forecast/summary.md)) |
| Stand-alone DLT-dyn and CI: Electricity, Traffic | `long_large`; `ci_long` | `long_large.tsv`, `ci_long.tsv` | GPU | `make_gpu_jobs.sh` |
| Guards of the stand-alone model | `ablation/make_jobs.py guard` | `ablation/guard.tsv` | CPU | done (stage 2, [`guard.md`](ablation/guard.md)) |
| P0 kernel and gate study | `ablation/make_jobs.py p0` | `ablation/p0.tsv` | CPU (ECL: GPU) | stage 3, running |
| Sensitivity, neighbour selection | `hyperparam/make_jobs.py sensitivity \| neighbours` | `hyperparam/*.tsv` | CPU | stage 4 |
| Validation search and searched configurations | `hyperparam/hpsearch.py make \| tuned` | `hyperparam/hp.tsv`, `tuned.tsv` | CPU (ECL, Traffic, PEMS: GPU) | stage 5 |
| All variables as neighbours; lead-lag diagnostic | `ablation/make_jobs.py allnb`; `plugin/leadlag_diag.py` | `ablation/allnb.tsv`, `plugin/leadlag_diag.json` | CPU (ECL, Traffic: GPU) | stage 6 |
| Learned kernels on PEMS08 | `mechanism/kernels.py` | `mechanism/kernels_pems08.json` | CPU | after stage 3 |
| Fusions with the six backbones; plug-in benchmark | `plugin/make_jobs.py plugin_fusion \| plugin_all` | `plugin/*.tsv` | GPU | `make_gpu_jobs.sh` |
| Efficiency (time, memory) | `efficiency/run_efficiency.py`, `eff_tune.py` | `efficiency/` | CPU, GPU | done |

The conclusions of each study follow here when its runs are complete.

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
