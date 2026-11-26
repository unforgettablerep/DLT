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
| Guards of the stand-alone model | `ablation/make_jobs.py guard` | `ablation/guard.tsv` | CPU | stage 2, running (PEMS08 done) |
| P0 kernel and gate study | `ablation/make_jobs.py p0` | `ablation/p0.tsv` | CPU (ECL: GPU) | stage 3 |
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
