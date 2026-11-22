# Official settings of the six baselines

Unchanged copies of the official run scripts; `scripts/baselines.py` reads every baseline's settings from them
(`python scripts/baselines.py` prints them with their source). Only the hyper-parameters of a line are taken; the
data, paths and run names come from `scripts/exps.py`, and every baseline runs with the protocol of the official
loaders (`baselines.protocol`).

| folder | model | source | commit |
|---|---|---|---|
| `SMamba/` | S-Mamba | github.com/wzhwzhwzh0921/S-D-Mamba, `scripts/multivariate_forecasting/` | `e7e8bf0` (2025-05-20) |
| `iTransformer/` | iTransformer | github.com/thuml/iTransformer, `scripts/multivariate_forecasting/` | `c2426e6` (2025-07-17) |
| `TSLib/` | PatchTST, DLinear, TiDE | github.com/thuml/Time-Series-Library, `scripts/long_term_forecast/` (their long-term forecasting scripts) | `4e938a1` (2026-04-18) |
| `RLinear/` | RLinear | github.com/plumprc/RTSF, `script.md` | `0fec001` (2023-09-13) |

Where an official line is missing, `scripts/baselines.py` (`FALLBACK`) takes another dataset's line and says so:
PatchTST on Solar and PEMS takes the Electricity line; TiDE, which has a line for ETTh1 only, takes that one
everywhere; RLinear on Exchange takes ETTh1's and on Traffic, Solar and PEMS Electricity's; DLinear without a line
runs with the library defaults (it has no settings of its own besides training).

The S-Mamba paper's baseline numbers equal those of the iTransformer paper, which reproduced its baselines in
Time-Series-Library "built on the configurations provided by each model's original paper or official code"
(iTransformer paper, Appendix A.2). The copies here are those configurations as far as they are published.
