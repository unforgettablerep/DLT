# Stand-alone DLT-dyn with the guards of the plug-in

Test MSE (mean ± std over seeds) of the arms of suite `guard`; vs dyn / vs ci = change of the mean MSE against
DLT-dyn as in the paper / the self head alone; p = Welch t-test against dyn; kept = share of variables whose
correction the per-variable selection kept; best ep. = epoch of the best validation loss (two-stage: stage 1 +
branch epochs).

## ETTh1 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.3887 ± 0.0062 | -5.2 % |  | 0.011 |  | 4.7 |
| DLT-dyn (paper) | 3 | 0.4102 ± 0.0055 |  | +5.5 % |  |  | 1.3 |
| + per-variable selection | 3 | 0.4009 ± 0.0087 | -2.3 % | +3.1 % | 0.201 | 19% | 1.3 |
| branch at 1/10 learning rate | 3 | 0.4055 ± 0.0036 | -1.1 % | +4.3 % | 0.295 |  | 1.0 |
| 1/10 learning rate + selection | 3 | 0.4024 ± 0.0029 | -1.9 % | +3.5 % | 0.116 | 43% | 1.0 |
| two-stage training | 3 | 0.4108 ± 0.0046 | +0.1 % | +5.7 % | 0.900 |  | 3.0+1.0 |
| two-stage + selection | 3 | 0.3927 ± 0.0092 | -4.3 % | +1.0 % | 0.060 | 0% | 3.0+1.0 |

## ETTh2 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.2902 ± 0.0000 | -14.5 % |  | 0.023 |  | 4.7 |
| DLT-dyn (paper) | 3 | 0.3396 ± 0.0130 |  | +17.0 % |  |  | 1.0 |
| + per-variable selection | 3 | 0.2983 ± 0.0006 | -12.2 % | +2.8 % | 0.031 | 0% | 1.0 |
| branch at 1/10 learning rate | 3 | 0.3641 ± 0.0052 | +7.2 % | +25.5 % | 0.067 |  | 5.7 |
| 1/10 learning rate + selection | 3 | 0.2944 ± 0.0037 | -13.3 % | +1.4 % | 0.020 | 5% | 5.7 |
| two-stage training | 3 | 0.3292 ± 0.0034 | -3.1 % | +13.4 % | 0.299 |  | 6.3+2.0 |
| two-stage + selection | 3 | 0.2893 ± 0.0001 | -14.8 % | -0.3 % | 0.022 | 0% | 6.3+2.0 |

## ETTh2 H=720

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.4222 ± 0.0016 | -10.0 % |  | 0.007 |  | 6.3 |
| DLT-dyn (paper) | 3 | 0.4690 ± 0.0077 |  | +11.1 % |  |  | 2.7 |
| + per-variable selection | 3 | 0.4233 ± 0.0071 | -9.7 % | +0.3 % | 0.002 | 10% | 2.7 |
| branch at 1/10 learning rate | 3 | 0.4481 ± 0.0120 | -4.5 % | +6.1 % | 0.075 |  | 1.0 |
| 1/10 learning rate + selection | 3 | 0.4392 ± 0.0062 | -6.4 % | +4.0 % | 0.007 | 14% | 1.0 |
| two-stage training | 3 | 0.4669 ± 0.0164 | -0.4 % | +10.6 % | 0.854 |  | 6.3+3.0 |
| two-stage + selection | 3 | 0.4411 ± 0.0078 | -6.0 % | +4.5 % | 0.012 | 14% | 6.3+3.0 |

## Exchange H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.0824 ± 0.0003 | -25.4 % |  | 0.045 |  | 4.7 |
| DLT-dyn (paper) | 3 | 0.1104 ± 0.0106 |  | +34.0 % |  |  | 2.0 |
| + per-variable selection | 3 | 0.0859 ± 0.0020 | -22.2 % | +4.2 % | 0.053 | 21% | 2.0 |
| branch at 1/10 learning rate | 3 | 0.0891 ± 0.0013 | -19.3 % | +8.1 % | 0.071 |  | 1.0 |
| 1/10 learning rate + selection | 3 | 0.0863 ± 0.0007 | -21.8 % | +4.7 % | 0.058 | 4% | 1.0 |
| two-stage training | 3 | 0.1203 ± 0.0110 | +9.0 % | +46.0 % | 0.325 |  | 5.7+0.7 |
| two-stage + selection | 3 | 0.0841 ± 0.0015 | -23.9 % | +2.0 % | 0.048 | 17% | 5.7+0.7 |

## Exchange H=720

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.8381 ± 0.0027 | -41.1 % |  | 0.018 |  | 5.0 |
| DLT-dyn (paper) | 3 | 1.4226 ± 0.1367 |  | +69.7 % |  |  | 2.7 |
| + per-variable selection | 3 | 1.0904 ± 0.0888 | -23.4 % | +30.1 % | 0.031 | 62% | 2.7 |
| branch at 1/10 learning rate | 3 | 1.0435 ± 0.0463 | -26.6 % | +24.5 % | 0.030 |  | 4.0 |
| 1/10 learning rate + selection | 3 | 0.9170 ± 0.0046 | -35.5 % | +9.4 % | 0.023 | 46% | 4.0 |
| two-stage training | 3 | 1.5546 ± 0.2360 | +9.3 % | +85.5 % | 0.460 |  | 3.0+1.7 |
| two-stage + selection | 3 | 1.1028 ± 0.1477 | -22.5 % | +31.6 % | 0.052 | 50% | 3.0+1.7 |

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

## Weather H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.1942 ± 0.0006 | +18.6 % |  | 0.000 |  | 7.7 |
| DLT-dyn (paper) | 3 | 0.1637 ± 0.0006 |  | -15.7 % |  |  | 6.3 |
| + per-variable selection | 3 | 0.1666 ± 0.0022 | +1.8 % | -14.2 % | 0.147 | 84% | 6.3 |
| branch at 1/10 learning rate | 3 | 0.1649 ± 0.0014 | +0.8 % | -15.1 % | 0.256 |  | 7.7 |
| 1/10 learning rate + selection | 3 | 0.1669 ± 0.0014 | +2.0 % | -14.0 % | 0.041 | 84% | 7.7 |
| two-stage training | 3 | 0.1669 ± 0.0012 | +2.0 % | -14.0 % | 0.027 |  | 9.3+3.0 |
| two-stage + selection | 3 | 0.1669 ± 0.0012 | +2.0 % | -14.0 % | 0.027 | 90% | 9.3+3.0 |

## Over the settings: change against the self head alone (CI)

| arm | weakly coupled: mean | worst | coupled: mean | least gain |
|---|---:|---:|---:|---:|
| DLT-dyn (paper) | +27.5 % | +69.7 % | -35.5 % | -15.7 % |
| + per-variable selection | +8.1 % | +30.1 % | -34.8 % | -14.2 % |
| branch at 1/10 learning rate | +13.7 % | +25.5 % | -28.6 % | -15.1 % |
| 1/10 learning rate + selection | +4.6 % | +9.4 % | -28.0 % | -14.0 % |
| two-stage training | +32.2 % | +85.5 % | -36.3 % | -14.0 % |
| two-stage + selection | +7.8 % | +31.6 % | -36.3 % | -14.0 % |
