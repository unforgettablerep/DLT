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
| two-stage + significant selection | 3 | 0.3927 ± 0.0092 | -4.3 % | +1.0 % | 0.060 | 0% | 3.0+1.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.3927 ± 0.0092 | -4.3 % | +1.0 % | 0.060 | 0% | 3.0+1.0 |
| self head alone, last-value anchor | 3 | 0.3884 ± 0.0030 | -5.3 % | -0.1 % | 0.008 |  | 5.3 |
| DLT-dyn, last-value anchor | 3 | 0.4165 ± 0.0074 | +1.5 % | +7.1 % | 0.314 |  | 1.0 |
| last-value anchor + two-stage + significant selection | 3 | 0.3988 ± 0.0127 | -2.8 % | +2.6 % | 0.255 | 5% | 2.7+0.0 |
| difference input + two-stage + significant selection | 3 | 0.3927 ± 0.0092 | -4.3 % | +1.0 % | 0.060 | 0% | 3.0+0.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.3974 ± 0.0094 | -3.1 % | +2.2 % | 0.127 | 19% | 2.7+0.7 |

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
| two-stage + significant selection | 3 | 0.2893 ± 0.0001 | -14.8 % | -0.3 % | 0.022 | 0% | 6.3+2.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.2893 ± 0.0001 | -14.8 % | -0.3 % | 0.022 | 0% | 6.3+2.0 |
| self head alone, last-value anchor | 3 | 0.2916 ± 0.0006 | -14.1 % | +0.5 % | 0.024 |  | 5.7 |
| DLT-dyn, last-value anchor | 3 | 0.3334 ± 0.0045 | -1.8 % | +14.9 % | 0.504 |  | 6.3 |
| last-value anchor + two-stage + significant selection | 3 | 0.2915 ± 0.0004 | -14.2 % | +0.4 % | 0.024 | 0% | 6.3+1.3 |
| difference input + two-stage + significant selection | 3 | 0.2893 ± 0.0001 | -14.8 % | -0.3 % | 0.022 | 0% | 6.3+0.7 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.2915 ± 0.0004 | -14.2 % | +0.4 % | 0.024 | 0% | 6.3+0.7 |

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
| two-stage + significant selection | 3 | 0.4217 ± 0.0009 | -10.1 % | -0.1 % | 0.008 | 0% | 6.3+3.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.4217 ± 0.0009 | -10.1 % | -0.1 % | 0.008 | 0% | 6.3+3.0 |
| self head alone, last-value anchor | 3 | 0.4242 ± 0.0021 | -9.6 % | +0.5 % | 0.007 |  | 7.0 |
| DLT-dyn, last-value anchor | 3 | 0.4615 ± 0.0060 | -1.6 % | +9.3 % | 0.260 |  | 2.7 |
| last-value anchor + two-stage + significant selection | 3 | 0.4229 ± 0.0004 | -9.8 % | +0.2 % | 0.009 | 0% | 8.7+3.0 |
| difference input + two-stage + significant selection | 3 | 0.4217 ± 0.0009 | -10.1 % | -0.1 % | 0.008 | 0% | 6.3+1.0 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.4229 ± 0.0004 | -9.8 % | +0.2 % | 0.009 | 0% | 8.7+0.7 |

## ETTm1 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.3510 ± 0.0009 | +7.4 % |  | 0.002 |  | 3.0 |
| DLT-dyn (paper) | 3 | 0.3267 ± 0.0026 |  | -6.9 % |  |  | 5.0 |
| + per-variable selection | 3 | 0.3349 ± 0.0030 | +2.5 % | -4.6 % | 0.024 | 71% | 5.0 |
| branch at 1/10 learning rate | 3 | 0.3330 ± 0.0014 | +1.9 % | -5.1 % | 0.032 |  | 8.0 |
| 1/10 learning rate + selection | 3 | 0.3359 ± 0.0018 | +2.8 % | -4.3 % | 0.009 | 76% | 8.0 |
| two-stage training | 3 | 0.3290 ± 0.0059 | +0.7 % | -6.3 % | 0.597 |  | 2.0+5.3 |
| two-stage + selection | 3 | 0.3368 ± 0.0062 | +3.1 % | -4.0 % | 0.091 | 86% | 2.0+5.3 |
| two-stage + significant selection | 3 | 0.3372 ± 0.0067 | +3.2 % | -3.9 % | 0.100 | 43% | 2.0+5.3 |
| two-stage + significant selection, level 0.25 | 3 | 0.3368 ± 0.0062 | +3.1 % | -4.0 % | 0.091 | 86% | 2.0+5.3 |
| self head alone, last-value anchor | 3 | 0.3505 ± 0.0033 | +7.3 % | -0.2 % | 0.001 |  | 3.0 |
| DLT-dyn, last-value anchor | 3 | 0.3325 ± 0.0049 | +1.8 % | -5.3 % | 0.166 |  | 5.7 |
| last-value anchor + two-stage + significant selection | 3 | 0.3360 ± 0.0045 | +2.8 % | -4.3 % | 0.049 | 48% | 4.0+3.0 |
| difference input + two-stage + significant selection | 3 | 0.3388 ± 0.0023 | +3.7 % | -3.5 % | 0.004 | 43% | 2.0+2.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.3384 ± 0.0068 | +3.6 % | -3.6 % | 0.083 | 52% | 4.0+2.7 |

## ETTm2 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.1821 ± 0.0003 | +1.8 % |  | 0.040 |  | 5.7 |
| DLT-dyn (paper) | 3 | 0.1788 ± 0.0013 |  | -1.8 % |  |  | 2.7 |
| + per-variable selection | 3 | 0.1790 ± 0.0009 | +0.1 % | -1.7 % | 0.902 | 38% | 2.7 |
| branch at 1/10 learning rate | 3 | 0.1776 ± 0.0026 | -0.7 % | -2.5 % | 0.515 |  | 6.3 |
| 1/10 learning rate + selection | 3 | 0.1772 ± 0.0014 | -0.9 % | -2.7 % | 0.215 | 48% | 6.3 |
| two-stage training | 3 | 0.1818 ± 0.0008 | +1.7 % | -0.2 % | 0.034 |  | 4.0+1.3 |
| two-stage + selection | 3 | 0.1798 ± 0.0009 | +0.6 % | -1.2 % | 0.336 | 33% | 4.0+1.3 |
| two-stage + significant selection | 3 | 0.1801 ± 0.0011 | +0.7 % | -1.1 % | 0.258 | 29% | 4.0+1.3 |
| two-stage + significant selection, level 0.25 | 3 | 0.1798 ± 0.0009 | +0.6 % | -1.2 % | 0.336 | 33% | 4.0+1.3 |
| self head alone, last-value anchor | 3 | 0.1823 ± 0.0010 | +1.9 % | +0.1 % | 0.023 |  | 5.0 |
| DLT-dyn, last-value anchor | 3 | 0.1811 ± 0.0010 | +1.2 % | -0.6 % | 0.081 |  | 3.7 |
| last-value anchor + two-stage + significant selection | 3 | 0.1812 ± 0.0015 | +1.3 % | -0.5 % | 0.112 | 29% | 3.3+0.3 |
| difference input + two-stage + significant selection | 3 | 0.1801 ± 0.0007 | +0.7 % | -1.1 % | 0.207 | 24% | 4.0+1.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.1803 ± 0.0011 | +0.8 % | -1.0 % | 0.202 | 33% | 3.3+0.3 |

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
| two-stage + significant selection | 3 | 0.0823 ± 0.0003 | -25.5 % | -0.2 % | 0.044 | 0% | 5.7+0.7 |
| two-stage + significant selection, level 0.25 | 3 | 0.0839 ± 0.0015 | -24.0 % | +1.8 % | 0.047 | 12% | 5.7+0.7 |
| self head alone, last-value anchor | 3 | 0.0826 ± 0.0005 | -25.2 % | +0.2 % | 0.045 |  | 3.0 |
| DLT-dyn, last-value anchor | 3 | 0.0989 ± 0.0034 | -10.5 % | +20.0 % | 0.193 |  | 1.0 |
| last-value anchor + two-stage + significant selection | 3 | 0.0824 ± 0.0008 | -25.4 % | -0.1 % | 0.044 | 0% | 5.3+1.0 |
| difference input + two-stage + significant selection | 3 | 0.0823 ± 0.0003 | -25.5 % | -0.1 % | 0.044 | 4% | 5.7+0.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.0825 ± 0.0007 | -25.3 % | +0.0 % | 0.044 | 4% | 5.3+0.0 |

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
| two-stage + significant selection | 3 | 0.8443 ± 0.0073 | -40.7 % | +0.7 % | 0.018 | 0% | 3.0+1.7 |
| two-stage + significant selection, level 0.25 | 3 | 0.8443 ± 0.0073 | -40.7 % | +0.7 % | 0.018 | 0% | 3.0+1.7 |
| self head alone, last-value anchor | 3 | 0.8345 ± 0.0077 | -41.3 % | -0.4 % | 0.017 |  | 6.0 |
| DLT-dyn, last-value anchor | 3 | 1.1515 ± 0.0973 | -19.1 % | +37.4 % | 0.055 |  | 1.0 |
| last-value anchor + two-stage + significant selection | 3 | 0.8195 ± 0.0093 | -42.4 % | -2.2 % | 0.016 | 0% | 2.0+1.0 |
| difference input + two-stage + significant selection | 3 | 0.8443 ± 0.0073 | -40.7 % | +0.7 % | 0.018 | 0% | 3.0+4.0 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.8195 ± 0.0093 | -42.4 % | -2.2 % | 0.016 | 0% | 2.0+1.0 |

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
| two-stage + significant selection | 3 | 0.1770 ± 0.0012 | +8.1 % | -8.8 % | 0.001 | 63% | 9.3+3.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.1692 ± 0.0051 | +3.4 % | -12.9 % | 0.200 | 89% | 9.3+3.0 |
| self head alone, last-value anchor | 3 | 0.1933 ± 0.0008 | +18.1 % | -0.4 % | 0.000 |  | 5.0 |
| DLT-dyn, last-value anchor | 3 | 0.1638 ± 0.0015 | +0.1 % | -15.6 % | 0.894 |  | 8.7 |
| last-value anchor + two-stage + significant selection | 3 | 0.1790 ± 0.0010 | +9.3 % | -7.8 % | 0.000 | 63% | 7.3+0.0 |
| difference input + two-stage + significant selection | 3 | 0.1794 ± 0.0019 | +9.6 % | -7.6 % | 0.003 | 67% | 9.3+0.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.1805 ± 0.0006 | +10.3 % | -7.1 % | 0.000 | 65% | 7.3+0.0 |

## Over the settings: change of the MSE against the self head alone (CI)

| arm | ETTh1-96 | ETTh2-96 | ETTh2-720 | ETTm1-96 | ETTm2-96 | Exchange-96 | Exchange-720 | PEMS08-96 | Weather-96 | ETT + Exchange: mean | worst | coupled: mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DLT-dyn (paper) | +5.5 % | +17.0 % | +11.1 % | -6.9 % | -1.8 % | +34.0 % | +69.7 % | -55.3 % | -15.7 % | +18.4 % | +69.7 % | -35.5 % |
| + per-variable selection | +3.1 % | +2.8 % | +0.3 % | -4.6 % | -1.7 % | +4.2 % | +30.1 % | -55.4 % | -14.2 % | +4.9 % | +30.1 % | -34.8 % |
| branch at 1/10 learning rate | +4.3 % | +25.5 % | +6.1 % | -5.1 % | -2.5 % | +8.1 % | +24.5 % | -42.2 % | -15.1 % | +8.7 % | +25.5 % | -28.6 % |
| 1/10 learning rate + selection | +3.5 % | +1.4 % | +4.0 % | -4.3 % | -2.7 % | +4.7 % | +9.4 % | -42.0 % | -14.0 % | +2.3 % | +9.4 % | -28.0 % |
| two-stage training | +5.7 % | +13.4 % | +10.6 % | -6.3 % | -0.2 % | +46.0 % | +85.5 % | -58.5 % | -14.0 % | +22.1 % | +85.5 % | -36.3 % |
| two-stage + selection | +1.0 % | -0.3 % | +4.5 % | -4.0 % | -1.2 % | +2.0 % | +31.6 % | -58.5 % | -14.0 % | +4.8 % | +31.6 % | -36.3 % |
| two-stage + significant selection | +1.0 % | -0.3 % | -0.1 % | -3.9 % | -1.1 % | -0.2 % | +0.7 % | -57.9 % | -8.8 % | -0.6 % | +1.0 % | -33.4 % |
| two-stage + significant selection, level 0.25 | +1.0 % | -0.3 % | -0.1 % | -4.0 % | -1.2 % | +1.8 % | +0.7 % | -58.3 % | -12.9 % | -0.3 % | +1.8 % | -35.6 % |
| self head alone, last-value anchor | -0.1 % | +0.5 % | +0.5 % | -0.2 % | +0.1 % | +0.2 % | -0.4 % | +0.0 % | -0.4 % | +0.1 % | +0.5 % | -0.2 % |
| DLT-dyn, last-value anchor | +7.1 % | +14.9 % | +9.3 % | -5.3 % | -0.6 % | +20.0 % | +37.4 % | -53.9 % | -15.6 % | +11.8 % | +37.4 % | -34.7 % |
| last-value anchor + two-stage + significant selection | +2.6 % | +0.4 % | +0.2 % | -4.3 % | -0.5 % | -0.1 % | -2.2 % | -56.9 % | -7.8 % | -0.6 % | +2.6 % | -32.3 % |
| difference input + two-stage + significant selection | +1.0 % | -0.3 % | -0.1 % | -3.5 % | -1.1 % | -0.1 % | +0.7 % | -55.5 % | -7.6 % | -0.5 % | +1.0 % | -31.6 % |
| last-value anchor + difference input + two-stage + significant selection | +2.2 % | +0.4 % | +0.2 % | -3.6 % | -1.0 % | +0.0 % | -2.2 % | -55.5 % | -7.1 % | -0.6 % | +2.2 % | -31.3 % |

