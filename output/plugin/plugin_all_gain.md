# Plug-in study: which backbones and which datasets gain from the DLT branch (2026-11-15)

**Source.** [`plugin_all.jsonl`](plugin_all.jsonl): seed 2021, RTX 4090, the same records as
[`plugin_all_gate.md`](plugin_all_gate.md); default arm `dlt`. All tables come from
`python scripts/p1/plugin_gain.py --files results/gpu/plugin_all.jsonl --diag results/gpu/leadlag_diag.json`.
The data diagnostic [`leadlag_diag.json`](leadlag_diag.json) was computed on the CPU with
`python scripts/p1/leadlag_diag.py --out results/gpu/leadlag_diag.json`, on all 13 datasets of plugin_all.

**Which comparisons count.** There are 260 settings with both arms; Traffic is left out because the TimesNet MSE is
NaN. Of these, 204 settings are kept:
- 176 where the backbone alone is within 10 % of its published result;
- 28 where it has no published result (TimeXer on Exchange and PEMS; TimeMixer and Informer on PEMS).

The other 56 settings are left out. There the backbone alone is more than 10 % behind its published result:
Informer on five sets, PatchTST on ETTm1, Exchange and PEMS, iTransformer on PEMS, S-Mamba on PEMS08, TimeMixer on
Exchange, and TimeXer on ETTh2.

Of the kept settings, 60 are marked **old**: their runs used settings or a protocol that plugin_all no longer
generates. These are PatchTST everywhere (its paper's settings instead of the TSLib scripts) and every PEMS run
(with time marks and the TSLib split). The old comparisons are internally consistent, because both arms used the
same settings. But they are to be run again.

Change = (+ DLT) / alone − 1 of the test MSE; negative means the branch helps. Single seed: earlier seeds varied by
less than 0.4 % (backbone alone) and up to 2.5 % (+ DLT) ([README](README.md)), so changes below about 2 % are
within noise.

## 1. Backbones

Kept comparisons only. The "cells" column gives the range of the change of the horizon-averaged MSE per dataset.

| backbone | settings | median change | branch helps | cells | with the validation rule |
|---|---:|---:|---:|---|---:|
| DLinear | 32 (8 old) | −2.8 % | 19 of 32 | −48.1 % (PEMS08) to +10.0 % (Exchange) | −12.8 % |
| TimesNet | 36 (8 old) | −0.9 % | 20 of 36 | −16.2 % (PEMS08) to +12.3 % (Exchange) | −4.2 % |
| PatchTST | 16 (all old) | −0.2 % | 9 of 16 | −6.7 % (Weather) to +14.1 % (ETTh2) | −1.9 % |
| iTransformer | 24 | +1.7 % | 10 of 24 | −4.7 % (Weather) to +32.0 % (Exchange) | −1.0 % |
| S-Mamba | 28 (4 old) | +2.0 % | 10 of 28 | −2.8 % (PEMS04) to +44.8 % (Exchange) | +1.8 % |
| TimeMixer | 28 (8 old) | +1.2 % | 6 of 28 | −1.9 % (PEMS08) to +10.1 % (ETTh2) | +0.1 % |
| TimeXer | 28 (8 old) | +2.2 % | 3 of 28 | +0.9 % (PEMS08) to +36.7 % (Exchange) | +1.8 % |
| Informer | 12 (8 old) | −12.8 % | 11 of 12 | −22.1 % (PEMS08) to −9.0 % (PEMS04) | −14.1 % |

**DLinear gains most.**
- PEMS04 −46.4 %, PEMS08 −48.1 %, Weather −9.8 %.
- About neutral on the ETT sets (−0.9 % to +2.5 %), and +10.0 % on Exchange.
- Its PEMS runs are old, but DLinear ignores time marks, and the backbone alone matches the published table to
  1–3 %. These gains are the most solid in the study.
- They repair a model that lacks what PEMS needs: DLinear + DLT stays 16–129 % behind the best backbone alone
  (TimeMixer) on PEMS.
- On Weather it reaches the best backbone alone (TimeXer): +3.1 %, +0.2 %, −0.8 % and −3.7 % at horizons 96 to 720.

**TimesNet is second.**
- PEMS04 −14.4 %, PEMS08 −16.2 %, Weather −4.8 %.
- Electricity −1.8 %, −3.8 % and −2.4 % at horizons 96 to 336, but +19.2 % at 720.
- On ETT and Exchange it ranges from +0.2 % to +12.3 %.

**PatchTST, iTransformer and S-Mamba gain clearly on Weather only.**
- Weather: −6.7 % (PatchTST, old), −4.7 % (iTransformer), −2.2 % (S-Mamba); also −2.8 % for S-Mamba on PEMS04
  (old).
- They lose clearly on ETTh2 (+11 % to +14 %) and on Exchange (+32 % to +45 %).
- Their large PEMS "gains" (−12 % to −47 %) sit on backbones that trained badly and are left out.

**TimeMixer and TimeXer gain nothing.**
- The branch helps in 6 and 3 of 28 settings; their best cells are −1.9 % and +0.9 %.
- Together they are the best backbone alone in 26 of the 32 dataset × horizon settings (TimeMixer 14,
  TimeXer 12).

**Informer's gains do not count.** Informer alone is several times worse than the other backbones: on ETTh2 its MSE
is 3.0 to 6.3, against 0.3 to 0.8. A gain over it says nothing about DLT on a competitive backbone.

## 2. Datasets

| dataset | settings | median change | branch helps | cells | with the validation rule |
|---|---:|---:|---:|---|---:|
| PEMS08 | 20 (all old) | −15.1 % | 17 of 20 | −48.1 % (DLinear) to +0.9 % (TimeXer) | −18.6 % |
| PEMS04 | 24 (all old) | −4.0 % | 18 of 24 | −46.4 % (DLinear) to +1.6 % (TimeMixer) | −11.7 % |
| Weather | 28 (4 old) | −3.6 % | 21 of 28 | −9.8 % (DLinear) to +1.1 % (TimeXer) | −4.5 % |
| Electricity | 4 (TimesNet only) | −2.1 % | 3 of 4 | +3.5 % (horizon 720: +19.2 %) | −2.0 % |
| ETTm2 | 28 (4 old) | +0.6 % | 10 of 28 | −0.5 % (iTransformer) to +4.9 % (TimeXer) | −0.1 % |
| ETTm1 | 24 | +2.0 % | 6 of 24 | −0.2 % (iTransformer) to +6.0 % (TimesNet) | +0.9 % |
| ETTh1 | 28 (4 old) | +3.0 % | 6 of 28 | −0.8 % (PatchTST) to +4.2 % (iTransformer) | +0.1 % |
| ETTh2 | 28 (4 old) | +9.5 % | 5 of 28 | −13.0 % (Informer) to +14.4 % (S-Mamba) | −1.7 % |
| Exchange | 20 | +14.6 % | 2 of 20 | +10.0 % (DLinear) to +44.8 % (S-Mamba) | +6.6 % |

- **The branch helps on PEMS08, PEMS04 and Weather and harms on Exchange, ETTh2, ETTh1 and ETTm1.**
  - A two-sided sign test over the settings gives p < 0.05 for each of these seven datasets.
  - The p-values are optimistic: settings of one dataset share data.
  - ETTm2 is neutral.
- **The validation rule removes most of the harm.** It keeps the branch only where it lowers the validation loss.
  - Over all 204 settings, the mean change goes from 0.0 % to −3.3 %.
  - Settings made worse by more than 2 % drop from 77 to 16.
  - Exchange stays harmed (+6.6 %): there, validation and test periods disagree ([gate study](plugin_all_gate.md)).
- **Electricity and Traffic so far.** In this suite, only TimesNet on Electricity has both arms. Earlier tier-1c runs
  (older code and settings, 2–3 seeds; [README](README.md)) point the same way as the pattern above:
  - DLinear: −10.3 % on Electricity and −13.0 % on Traffic;
  - PatchTST: −8.2 % on Electricity and +13.6 % on Traffic;
  - iTransformer: −0.9 % on Electricity and +11.5 % on Traffic.

## 3. What the size of the gain goes with

**Headroom.** The further the backbone alone is behind the best backbone alone of the same dataset and horizon, the
larger the gain:

| | all 204 kept settings | Weather | PEMS08 | PEMS04 | ETTh2 | ETTh1 | ETTm2 | ETTm1 | Exchange |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spearman ρ | −0.55 | −0.92 | −0.74 | −0.72 | −0.61 | −0.60 | −0.50 | −0.32 | −0.03 |

DLT mostly closes the gap to the best backbone; it rarely moves past it. The best + DLT result against the best
backbone alone, per horizon:

| dataset | per horizon |
|---|---|
| ETTh1 | +3.1 % to +4.4 % |
| ETTh2 | +6.0 % to +18.9 % |
| ETTm1 | +1.1 % to +6.3 % |
| ETTm2 | +1.3 % to +4.1 % |
| Exchange | +8.3 % to +11.5 % |
| Weather | +0.7 %, +0.2 %, −0.8 %, −3.7 % |
| PEMS04 | +1.9 %, +0.3 %, −2.9 %, +6.2 % |
| PEMS08 | −0.7 %, +0.8 %, −0.5 %, −4.5 % |

On Weather, DLinear + DLT is best at horizons 192 to 720. On PEMS, TimeMixer + DLT and TimeMixer alone are about level.

**The data.** `leadlag_diag.py` fits linear ridge forecasters with input 96, on the splits and the scaling of the
runs, for at most 32 sampled target variables per dataset. It starts from SHARED, one set of weights for every
variable (as DLinear), and measures what three additions change on the test split:
- **OWN** adds per-variable weights;
- **NB** adds the input windows of the 8 neighbours that DLT selects, as DLT adds a cross-variable branch to
  DLinear;
- **BOTH vs OWN** measures what the neighbours add beyond the variable's own past.

The training split is in brackets. Lag 0 is the share of neighbour pairs whose cross-correlation peaks at lag 0
(no lead). Level r is the mean |correlation| of a variable with its first neighbour; near-copies have about 1.

| dataset | lag 0 | level r | OWN | NB | BOTH vs OWN | DLinear + DLT | median over the backbones |
|---|---:|---:|---:|---:|---:|---:|---:|
| PEMS04 | 77 % | 0.89 | −5.2 % (−7.0 %) | −36.4 % (−45.2 %) | −42.7 % (−50.5 %) | −46.4 % | −4.0 % |
| PEMS08 | 60 % | 0.89 | −4.7 % (−7.1 %) | −31.6 % (−39.8 %) | −36.2 % (−44.4 %) | −48.1 % | −15.1 % |
| PEMS07 | 80 % | 0.96 | −6.1 % (−7.4 %) | −21.3 % (−36.1 %) | −19.9 % (−35.6 %) | not run | not run |
| PEMS03 | 81 % | 0.96 | −5.1 % (−7.4 %) | −13.5 % (−44.1 %) | −11.2 % (−46.6 %) | not run | not run |
| Weather | 76 % | 0.70 | −9.0 % (−10.2 %) | −6.0 % (−13.8 %) | +0.3 % (−7.8 %) | −9.8 % | −3.6 % |
| ETTm1 | 63 % | 0.70 | −3.7 % (−6.6 %) | −5.2 % (−13.6 %) | −1.6 % (−7.8 %) | +1.2 % | +2.0 % |
| Electricity | 76 % | 0.77 | −8.1 % (−8.7 %) | −3.0 % (−9.6 %) | −3.2 % (−9.7 %) | not run | −2.1 % (TimesNet) |
| Traffic | 98 % | 0.90 | −1.5 % (−4.1 %) | −2.5 % (−18.9 %) | −3.2 % (−18.4 %) | NaN / not run | NaN / not run |
| Solar | 72 % | 0.96 | +1.5 % (−2.7 %) | −2.3 % (−10.4 %) | −7.0 % (−18.7 %) | not run | not run |
| ETTm2 | 71 % | 0.50 | −1.7 % (−2.6 %) | −0.9 % (−10.7 %) | +0.7 % (−8.3 %) | +1.0 % | +0.6 % |
| ETTh1 | 51 % | 0.69 | +1.2 % (−3.0 %) | +0.6 % (−8.6 %) | −0.5 % (−6.1 %) | +2.5 % | +3.0 % |
| ETTh2 | 71 % | 0.36 | +1.7 % (−2.9 %) | +13.4 % (−22.0 %) | +11.9 % (−19.9 %) | −0.9 % | +9.5 % |
| Exchange | 100 % | 0.67 | +14.6 % (−3.9 %) | +39.8 % (−18.9 %) | +22.1 % (−15.7 %) | +10.0 % | +14.6 % |

**NB ranks the datasets as the runs do.** Its Spearman correlation is +0.81 with the DLinear + DLT change (8
datasets) and +0.93 with the median change over the backbones (9 datasets). Three cases:

- **PEMS: real cross-variable information.**
  - The neighbours' pasts remove 11–43 % of the error beyond a sensor's own past.
  - Most of this in-sample gain carries over to the test split (PEMS04: −50.5 % on training, −42.7 % on test).
  - Per-variable weights add little (−5 %): the sensors are alike, and what a sensor lacks is the state of the road
    around it.
- **Weather: no lead-lag, but near-copies.**
  - The neighbours add nothing beyond the own past (+0.3 %).
  - What helps is per-variable information (OWN −9.0 %). The neighbours carry it because several Weather variables
    are near-copies of each other:
    - T and Tpot, level correlation 0.996;
    - VPact, sh and H2OC, 1.000;
    - SWDR and PAR, 0.999.
  - So the branch hands a shared-weight backbone the history of the target itself. This fits the ranking on
    Weather: DLinear (−9.8 %) and PatchTST (−6.7 %) share weights across variables; TimeXer and TimeMixer gain
    nothing.
- **Exchange and ETTh2: relations that do not carry over.**
  - The neighbours lower the training error by 19–22 % and raise the test error by 13–40 %.
  - On Exchange every selected pair peaks at lag 0: the rates move together, and none leads.
  - What the branch fits on the training period does not hold later, and the more flexible the backbone, the more it
    loses (Exchange: DLinear +10 %, S-Mamba +45 %).
  - The exception is DLinear on ETTh2 (−0.9 %), where the diagnostic predicts harm.

The diagnostic ranks the datasets; it does not predict the size of a gain. On Electricity and Traffic it gives −3.0 %
and −2.5 % for the shared linear model, while DLinear + DLT gained 10–13 % there in tier 1c. For the datasets not
run yet it predicts:
- clear gains on PEMS07 and PEMS03, below PEMS04 and PEMS08;
- small gains on Electricity, Traffic and Solar.

**Horizon.** Mean change per horizon over the kept settings:
- PEMS: −16.5 %, −16.0 %, −15.6 %, −11.0 % (12 to 96);
- Weather: −6.8 %, −5.6 %, −3.5 %, −1.6 % (96 to 720);
- ETT and Exchange: +6.8 %, +2.9 %, +5.2 %, +8.7 %.

## 4. What holds and what is open

**Solid.** These hold with valid baselines, over several backbones, in one direction:
- the harm on Exchange and ETTh2, and the small harm on ETTh1 and ETTm1;
- the gain on Weather;
- the gains of DLinear and TimesNet on PEMS04/08;
- no gain for TimeMixer and TimeXer.

**Open.**
- **All PEMS numbers** are to be repeated without time marks. iTransformer and S-Mamba should then reach their
  published PEMS results, and their gains will be smaller than the invalid −12 % to −31 % here.
- **PatchTST everywhere** is to be repeated with the TSLib settings.
- **Not run yet:** Electricity (except TimesNet), Traffic (TimesNet NaN, see
  [gate study](plugin_all_gate.md)), Solar, PEMS03 and PEMS07.
- **One seed only.**

**For the paper.** The data do not support "DLT improves any backbone". They support three claims:
- DLT gives cross-variable information to backbones that lack it: DLinear and TimesNet here, and probably PatchTST.
- It helps where the variables carry information the backbone does not use: the sensor networks of PEMS, and
  Weather.
- With the validation rule (a model-selection step on validation data), it rarely hurts.

The diagnostic gives the paper a way to tell in advance where the branch will help. NB on the training and the
validation split costs a few minutes on the CPU.
