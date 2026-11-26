# Stand-alone DLT-dyn with the guards of the plug-in

Test MSE (mean ± std over seeds) of the arms of suite `guard`; vs dyn / vs ci = change of the mean MSE against
DLT-dyn as in the paper / the self head alone; p = Welch t-test against dyn; kept = share of variables whose
correction the per-variable selection kept; best ep. = epoch of the best validation loss (two-stage: stage 1 +
branch epochs).

## PEMS08 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 1.1411 ± 0.0023 | +123.6 % |  | 0.000 |  | 4.0 |
| DLT-dyn (paper) | 3 | 0.5103 ± 0.0088 |  | -55.3 % |  |  | 9.3 |
| + per-variable selection | 3 | 0.5085 ± 0.0093 | -0.3 % | -55.4 % | 0.826 | 100% | 9.3 |
| branch at 1/10 learning rate | 3 | 0.6599 ± 0.0132 | +29.3 % | -42.2 % | 0.000 |  | 9.3 |
| 1/10 learning rate + selection | 3 | 0.6622 ± 0.0150 | +29.8 % | -42.0 % | 0.000 | 99% | 9.3 |
| two-stage training | 3 | 0.4732 ± 0.0121 | -7.3 % | -58.5 % | 0.016 |  | 5.3+10.0 |
| two-stage + selection | 3 | 0.4736 ± 0.0115 | -7.2 % | -58.5 % | 0.014 | 100% | 5.3+10.0 |
| two-stage + significant selection | 3 | 0.4803 ± 0.0147 | -5.9 % | -57.9 % | 0.050 | 99% | 5.3+10.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.4757 ± 0.0149 | -6.8 % | -58.3 % | 0.036 | 100% | 5.3+10.0 |
| self head alone, last-value anchor | 3 | 1.1415 ± 0.0038 | +123.7 % | +0.0 % | 0.000 |  | 5.0 |
| DLT-dyn, last-value anchor | 3 | 0.5266 ± 0.0206 | +3.2 % | -53.9 % | 0.305 |  | 9.3 |
| last-value anchor + two-stage + significant selection | 3 | 0.4922 ± 0.0101 | -3.5 % | -56.9 % | 0.081 | 99% | 5.0+10.0 |
| difference input + two-stage + significant selection | 3 | 0.5073 ± 0.0031 | -0.6 % | -55.5 % | 0.630 | 99% | 5.3+8.7 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.5081 ± 0.0117 | -0.4 % | -55.5 % | 0.812 | 99% | 5.0+7.3 |

## Over the settings: change of the MSE against the self head alone (CI)

| arm | PEMS08-96 | ETT + Exchange: mean | worst | coupled: mean |
|---|---:|---:|---:|---:|
| DLT-dyn (paper) | -55.3 % |  |  | -55.3 % |
| + per-variable selection | -55.4 % |  |  | -55.4 % |
| branch at 1/10 learning rate | -42.2 % |  |  | -42.2 % |
| 1/10 learning rate + selection | -42.0 % |  |  | -42.0 % |
| two-stage training | -58.5 % |  |  | -58.5 % |
| two-stage + selection | -58.5 % |  |  | -58.5 % |
| two-stage + significant selection | -57.9 % |  |  | -57.9 % |
| two-stage + significant selection, level 0.25 | -58.3 % |  |  | -58.3 % |
| self head alone, last-value anchor | +0.0 % |  |  | +0.0 % |
| DLT-dyn, last-value anchor | -53.9 % |  |  | -53.9 % |
| last-value anchor + two-stage + significant selection | -56.9 % |  |  | -56.9 % |
| difference input + two-stage + significant selection | -55.5 % |  |  | -55.5 % |
| last-value anchor + difference input + two-stage + significant selection | -55.5 % |  |  | -55.5 % |

