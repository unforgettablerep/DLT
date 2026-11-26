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
| Stand-alone DLT-dyn: ETT, Exchange, Weather; CI | `long_small`; `ablation/make_jobs.py ci_long` | `long_small.tsv`, `ablation/ci_long.tsv` | CPU | stage 1 of `run_cpu_campaign.sh` |
| Stand-alone DLT-dyn and CI: Electricity, Traffic | `long_large`; `ci_long` | `long_large.tsv`, `ci_long.tsv` | GPU | `make_gpu_jobs.sh` |
| Guards of the stand-alone model | `ablation/make_jobs.py guard` | `ablation/guard.tsv` | CPU | stage 2 (PEMS08 done) |
| P0 kernel and gate study | `ablation/make_jobs.py p0` | `ablation/p0.tsv` | CPU (ECL: GPU) | stage 3 |
| Sensitivity, neighbour selection | `hyperparam/make_jobs.py sensitivity \| neighbours` | `hyperparam/*.tsv` | CPU | stage 4 |
| Validation search and searched configurations | `hyperparam/hpsearch.py make \| tuned` | `hyperparam/hp.tsv`, `tuned.tsv` | CPU (ECL, Traffic, PEMS: GPU) | stage 5 |
| All variables as neighbours; lead-lag diagnostic | `ablation/make_jobs.py allnb`; `plugin/leadlag_diag.py` | `ablation/allnb.tsv`, `plugin/leadlag_diag.json` | CPU (ECL, Traffic: GPU) | stage 6 |
| Learned kernels on PEMS08 | `mechanism/kernels.py` | `mechanism/kernels_pems08.json` | CPU | after stage 3 |
| Fusions with the six backbones; plug-in benchmark | `plugin/make_jobs.py plugin_fusion \| plugin_all` | `plugin/*.tsv` | GPU | `make_gpu_jobs.sh` |
| Efficiency (time, memory) | `efficiency/run_efficiency.py`, `eff_tune.py` | `efficiency/` | CPU, GPU | done |

The conclusions of each study follow here when its runs are complete.
