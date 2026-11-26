# Stand-alone DLT-dyn with the guards of the plug-in

Test MSE (mean ± std over seeds) of the arms of suite `guard`; vs dyn / vs ci = change of the mean MSE against
DLT-dyn as in the paper / the self head alone; p = Welch t-test against dyn; kept = share of variables whose
correction the per-variable selection kept; best ep. = epoch of the best validation loss (two-stage: stage 1 +
branch epochs).

## ETTh1 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.3853 ± 0.0004 | -5.5 % |  | 0.013 |  | 5.3 |
| DLT-dyn (paper) | 3 | 0.4078 ± 0.0045 |  | +5.8 % |  |  | 1.3 |
| + per-variable selection | 3 | 0.3985 ± 0.0071 | -2.3 % | +3.4 % | 0.140 | 19% | 1.3 |
| branch at 1/10 learning rate | 3 | 0.4035 ± 0.0019 | -1.0 % | +4.7 % | 0.240 |  | 1.0 |
| 1/10 learning rate + selection | 3 | 0.4004 ± 0.0014 | -1.8 % | +3.9 % | 0.095 | 43% | 1.0 |
| two-stage training | 3 | 0.4095 ± 0.0040 | +0.4 % | +6.3 % | 0.652 |  | 4.3+1.0 |
| two-stage + selection | 3 | 0.3863 ± 0.0032 | -5.3 % | +0.3 % | 0.004 | 0% | 4.3+1.0 |
| two-stage + significant selection | 3 | 0.3863 ± 0.0032 | -5.3 % | +0.3 % | 0.004 | 0% | 4.3+1.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.3863 ± 0.0032 | -5.3 % | +0.3 % | 0.004 | 0% | 4.3+1.0 |
| self head alone, last-value anchor | 3 | 0.3870 ± 0.0005 | -5.1 % | +0.4 % | 0.015 |  | 6.0 |
| DLT-dyn, last-value anchor | 3 | 0.4155 ± 0.0088 | +1.9 % | +7.8 % | 0.272 |  | 1.0 |
| last-value anchor + two-stage + significant selection | 3 | 0.3893 ± 0.0035 | -4.5 % | +1.0 % | 0.006 | 0% | 3.7+1.0 |
| difference input + two-stage + significant selection | 3 | 0.3863 ± 0.0032 | -5.3 % | +0.3 % | 0.004 | 0% | 4.3+0.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.3905 ± 0.0037 | -4.2 % | +1.4 % | 0.008 | 14% | 3.7+0.0 |

## ETTh2 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.2899 ± 0.0005 | -13.6 % |  | 0.034 |  | 6.3 |
| DLT-dyn (paper) | 3 | 0.3356 ± 0.0150 |  | +15.8 % |  |  | 1.3 |
| + per-variable selection | 3 | 0.2974 ± 0.0014 | -11.4 % | +2.6 % | 0.047 | 0% | 1.3 |
| branch at 1/10 learning rate | 3 | 0.3644 ± 0.0052 | +8.6 % | +25.7 % | 0.067 |  | 5.3 |
| 1/10 learning rate + selection | 3 | 0.2944 ± 0.0032 | -12.3 % | +1.6 % | 0.037 | 5% | 5.3 |
| two-stage training | 3 | 0.3244 ± 0.0067 | -3.4 % | +11.9 % | 0.328 |  | 7.7+1.7 |
| two-stage + selection | 3 | 0.2894 ± 0.0001 | -13.8 % | -0.2 % | 0.033 | 0% | 7.7+1.7 |
| two-stage + significant selection | 3 | 0.2894 ± 0.0001 | -13.8 % | -0.2 % | 0.033 | 0% | 7.7+1.7 |
| two-stage + significant selection, level 0.25 | 3 | 0.2894 ± 0.0001 | -13.8 % | -0.2 % | 0.033 | 0% | 7.7+1.7 |
| self head alone, last-value anchor | 3 | 0.2912 ± 0.0002 | -13.2 % | +0.5 % | 0.036 |  | 7.3 |
| DLT-dyn, last-value anchor | 3 | 0.3326 ± 0.0068 | -0.9 % | +14.7 % | 0.770 |  | 3.7 |
| last-value anchor + two-stage + significant selection | 3 | 0.2914 ± 0.0004 | -13.2 % | +0.5 % | 0.036 | 0% | 7.0+1.7 |
| difference input + two-stage + significant selection | 3 | 0.2894 ± 0.0001 | -13.8 % | -0.2 % | 0.033 | 0% | 7.7+0.0 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.2914 ± 0.0004 | -13.2 % | +0.5 % | 0.036 | 0% | 7.0+0.0 |

## ETTh2 H=720

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.4238 ± 0.0023 | -7.5 % |  | 0.015 |  | 5.0 |
| DLT-dyn (paper) | 3 | 0.4580 ± 0.0085 |  | +8.1 % |  |  | 1.7 |
| + per-variable selection | 3 | 0.4275 ± 0.0150 | -6.6 % | +0.9 % | 0.051 | 19% | 1.7 |
| branch at 1/10 learning rate | 3 | 0.4478 ± 0.0116 | -2.2 % | +5.6 % | 0.293 |  | 1.0 |
| 1/10 learning rate + selection | 3 | 0.4423 ± 0.0086 | -3.4 % | +4.4 % | 0.088 | 19% | 1.0 |
| two-stage training | 3 | 0.4555 ± 0.0094 | -0.5 % | +7.5 % | 0.750 |  | 5.0+1.7 |
| two-stage + selection | 3 | 0.4311 ± 0.0086 | -5.9 % | +1.7 % | 0.019 | 10% | 5.0+1.7 |
| two-stage + significant selection | 3 | 0.4215 ± 0.0005 | -8.0 % | -0.6 % | 0.017 | 0% | 5.0+1.7 |
| two-stage + significant selection, level 0.25 | 3 | 0.4215 ± 0.0005 | -8.0 % | -0.6 % | 0.017 | 0% | 5.0+1.7 |
| self head alone, last-value anchor | 3 | 0.4244 ± 0.0025 | -7.3 % | +0.1 % | 0.015 |  | 6.7 |
| DLT-dyn, last-value anchor | 3 | 0.4640 ± 0.0093 | +1.3 % | +9.5 % | 0.452 |  | 3.0 |
| last-value anchor + two-stage + significant selection | 3 | 0.4237 ± 0.0012 | -7.5 % | -0.0 % | 0.019 | 0% | 6.0+2.3 |
| difference input + two-stage + significant selection | 3 | 0.4215 ± 0.0005 | -8.0 % | -0.6 % | 0.017 | 0% | 5.0+1.0 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.4237 ± 0.0012 | -7.5 % | -0.0 % | 0.019 | 0% | 6.0+1.0 |

## ETTm1 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.3505 ± 0.0012 | +6.6 % |  | 0.009 |  | 3.0 |
| DLT-dyn (paper) | 3 | 0.3288 ± 0.0044 |  | -6.2 % |  |  | 6.7 |
| + per-variable selection | 3 | 0.3369 ± 0.0053 | +2.5 % | -3.9 % | 0.114 | 71% | 6.7 |
| branch at 1/10 learning rate | 3 | 0.3332 ± 0.0017 | +1.3 % | -4.9 % | 0.216 |  | 7.0 |
| 1/10 learning rate + selection | 3 | 0.3363 ± 0.0017 | +2.3 % | -4.1 % | 0.085 | 71% | 7.0 |
| two-stage training | 3 | 0.3278 ± 0.0019 | -0.3 % | -6.5 % | 0.737 |  | 1.7+3.7 |
| two-stage + selection | 3 | 0.3364 ± 0.0017 | +2.3 % | -4.0 % | 0.081 | 86% | 1.7+3.7 |
| two-stage + significant selection | 3 | 0.3368 ± 0.0020 | +2.4 % | -3.9 % | 0.070 | 48% | 1.7+3.7 |
| two-stage + significant selection, level 0.25 | 3 | 0.3364 ± 0.0017 | +2.3 % | -4.0 % | 0.081 | 86% | 1.7+3.7 |
| self head alone, last-value anchor | 3 | 0.3496 ± 0.0021 | +6.3 % | -0.3 % | 0.006 |  | 3.0 |
| DLT-dyn, last-value anchor | 3 | 0.3311 ± 0.0067 | +0.7 % | -5.5 % | 0.645 |  | 7.7 |
| last-value anchor + two-stage + significant selection | 3 | 0.3333 ± 0.0018 | +1.4 % | -4.9 % | 0.213 | 43% | 5.7+3.3 |
| difference input + two-stage + significant selection | 3 | 0.3403 ± 0.0007 | +3.5 % | -2.9 % | 0.042 | 43% | 1.7+4.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.3363 ± 0.0030 | +2.3 % | -4.1 % | 0.080 | 43% | 5.7+1.0 |

## ETTm2 H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.1822 ± 0.0003 | +1.7 % |  | 0.080 |  | 4.3 |
| DLT-dyn (paper) | 3 | 0.1792 ± 0.0016 |  | -1.7 % |  |  | 2.7 |
| + per-variable selection | 3 | 0.1789 ± 0.0008 | -0.2 % | -1.8 % | 0.793 | 38% | 2.7 |
| branch at 1/10 learning rate | 3 | 0.1777 ± 0.0023 | -0.8 % | -2.5 % | 0.422 |  | 5.7 |
| 1/10 learning rate + selection | 3 | 0.1773 ± 0.0012 | -1.1 % | -2.7 % | 0.183 | 48% | 5.7 |
| two-stage training | 3 | 0.1829 ± 0.0019 | +2.1 % | +0.4 % | 0.063 |  | 5.7+1.0 |
| two-stage + selection | 3 | 0.1796 ± 0.0006 | +0.2 % | -1.4 % | 0.694 | 38% | 5.7+1.0 |
| two-stage + significant selection | 3 | 0.1801 ± 0.0003 | +0.5 % | -1.1 % | 0.413 | 19% | 5.7+1.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.1795 ± 0.0004 | +0.2 % | -1.5 % | 0.771 | 29% | 5.7+1.0 |
| self head alone, last-value anchor | 3 | 0.1822 ± 0.0009 | +1.7 % | -0.0 % | 0.066 |  | 5.0 |
| DLT-dyn, last-value anchor | 3 | 0.1830 ± 0.0017 | +2.1 % | +0.4 % | 0.050 |  | 3.3 |
| last-value anchor + two-stage + significant selection | 3 | 0.1811 ± 0.0004 | +1.1 % | -0.6 % | 0.175 | 38% | 3.3+1.0 |
| difference input + two-stage + significant selection | 3 | 0.1801 ± 0.0003 | +0.5 % | -1.2 % | 0.424 | 24% | 5.7+2.0 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.1810 ± 0.0015 | +1.0 % | -0.7 % | 0.234 | 29% | 3.3+1.7 |

## Exchange H=96

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.0820 ± 0.0001 | -25.7 % |  | 0.044 |  | 7.0 |
| DLT-dyn (paper) | 3 | 0.1104 ± 0.0106 |  | +34.6 % |  |  | 2.0 |
| + per-variable selection | 3 | 0.0859 ± 0.0020 | -22.2 % | +4.7 % | 0.053 | 21% | 2.0 |
| branch at 1/10 learning rate | 3 | 0.0908 ± 0.0032 | -17.8 % | +10.7 % | 0.075 |  | 1.3 |
| 1/10 learning rate + selection | 3 | 0.0869 ± 0.0008 | -21.3 % | +5.9 % | 0.061 | 8% | 1.3 |
| two-stage training | 3 | 0.1185 ± 0.0071 | +7.3 % | +44.4 % | 0.344 |  | 4.3+0.7 |
| two-stage + selection | 3 | 0.0857 ± 0.0033 | -22.4 % | +4.4 % | 0.046 | 21% | 4.3+0.7 |
| two-stage + significant selection | 3 | 0.0834 ± 0.0023 | -24.4 % | +1.7 % | 0.042 | 0% | 4.3+0.7 |
| two-stage + significant selection, level 0.25 | 3 | 0.0845 ± 0.0042 | -23.5 % | +3.0 % | 0.037 | 8% | 4.3+0.7 |
| self head alone, last-value anchor | 3 | 0.0826 ± 0.0007 | -25.2 % | +0.7 % | 0.045 |  | 3.3 |
| DLT-dyn, last-value anchor | 3 | 0.0985 ± 0.0028 | -10.8 % | +20.1 % | 0.185 |  | 1.3 |
| last-value anchor + two-stage + significant selection | 3 | 0.0823 ± 0.0008 | -25.5 % | +0.4 % | 0.044 | 0% | 4.0+1.0 |
| difference input + two-stage + significant selection | 3 | 0.0834 ± 0.0023 | -24.4 % | +1.7 % | 0.042 | 0% | 4.3+0.3 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.0824 ± 0.0007 | -25.3 % | +0.5 % | 0.044 | 4% | 4.0+0.0 |

## Exchange H=720

| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| self head alone (CI) | 3 | 0.8378 ± 0.0048 | -40.8 % |  | 0.033 |  | 4.7 |
| DLT-dyn (paper) | 3 | 1.4155 ± 0.1860 |  | +69.0 % |  |  | 3.3 |
| + per-variable selection | 3 | 1.0873 ± 0.0943 | -23.2 % | +29.8 % | 0.073 | 62% | 3.3 |
| branch at 1/10 learning rate | 3 | 1.0434 ± 0.0279 | -26.3 % | +24.5 % | 0.071 |  | 5.0 |
| 1/10 learning rate + selection | 3 | 0.9090 ± 0.0122 | -35.8 % | +8.5 % | 0.042 | 42% | 5.0 |
| two-stage training | 3 | 1.5117 ± 0.2965 | +6.8 % | +80.4 % | 0.663 |  | 5.0+0.7 |
| two-stage + selection | 3 | 1.1122 ± 0.1475 | -21.4 % | +32.8 % | 0.095 | 58% | 5.0+0.7 |
| two-stage + significant selection | 3 | 0.8439 ± 0.0101 | -40.4 % | +0.7 % | 0.033 | 0% | 5.0+0.7 |
| two-stage + significant selection, level 0.25 | 3 | 0.8439 ± 0.0101 | -40.4 % | +0.7 % | 0.033 | 0% | 5.0+0.7 |
| self head alone, last-value anchor | 3 | 0.8284 ± 0.0061 | -41.5 % | -1.1 % | 0.032 |  | 2.7 |
| DLT-dyn, last-value anchor | 3 | 1.1609 ± 0.1261 | -18.0 % | +38.6 % | 0.131 |  | 1.0 |
| last-value anchor + two-stage + significant selection | 3 | 0.8258 ± 0.0173 | -41.7 % | -1.4 % | 0.031 | 0% | 4.7+0.3 |
| difference input + two-stage + significant selection | 3 | 0.8439 ± 0.0101 | -40.4 % | +0.7 % | 0.033 | 0% | 5.0+0.7 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.8258 ± 0.0173 | -41.7 % | -1.4 % | 0.031 | 0% | 4.7+1.0 |

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
| self head alone (CI) | 3 | 0.1949 ± 0.0012 | +19.3 % |  | 0.000 |  | 8.0 |
| DLT-dyn (paper) | 3 | 0.1634 ± 0.0005 |  | -16.2 % |  |  | 8.3 |
| + per-variable selection | 3 | 0.1661 ± 0.0021 | +1.7 % | -14.7 % | 0.143 | 84% | 8.3 |
| branch at 1/10 learning rate | 3 | 0.1647 ± 0.0013 | +0.8 % | -15.5 % | 0.217 |  | 7.7 |
| 1/10 learning rate + selection | 3 | 0.1666 ± 0.0014 | +2.0 % | -14.5 % | 0.041 | 84% | 7.7 |
| two-stage training | 3 | 0.1661 ± 0.0005 | +1.7 % | -14.7 % | 0.003 |  | 7.7+0.0 |
| two-stage + selection | 3 | 0.1686 ± 0.0038 | +3.2 % | -13.5 % | 0.138 | 89% | 7.7+0.0 |
| two-stage + significant selection | 3 | 0.1768 ± 0.0010 | +8.2 % | -9.3 % | 0.000 | 65% | 7.7+0.0 |
| two-stage + significant selection, level 0.25 | 3 | 0.1686 ± 0.0038 | +3.2 % | -13.5 % | 0.138 | 89% | 7.7+0.0 |
| self head alone, last-value anchor | 3 | 0.1944 ± 0.0015 | +19.0 % | -0.2 % | 0.000 |  | 6.7 |
| DLT-dyn, last-value anchor | 3 | 0.1643 ± 0.0013 | +0.6 % | -15.7 % | 0.348 |  | 7.7 |
| last-value anchor + two-stage + significant selection | 3 | 0.1794 ± 0.0016 | +9.8 % | -7.9 % | 0.002 | 65% | 5.7+8.3 |
| difference input + two-stage + significant selection | 3 | 0.1782 ± 0.0015 | +9.1 % | -8.6 % | 0.001 | 73% | 7.7+0.0 |
| last-value anchor + difference input + two-stage + significant selection | 3 | 0.1795 ± 0.0012 | +9.9 % | -7.9 % | 0.000 | 62% | 5.7+7.7 |

## Over the settings: change of the MSE against the self head alone (CI)

| arm | ETTh1-96 | ETTh2-96 | ETTh2-720 | ETTm1-96 | ETTm2-96 | Exchange-96 | Exchange-720 | PEMS08-96 | Weather-96 | ETT + Exchange: mean | worst | coupled: mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DLT-dyn (paper) | +5.8 % | +15.8 % | +8.1 % | -6.2 % | -1.7 % | +34.6 % | +69.0 % | -55.3 % | -16.2 % | +17.9 % | +69.0 % | -35.7 % |
| + per-variable selection | +3.4 % | +2.6 % | +0.9 % | -3.9 % | -1.8 % | +4.7 % | +29.8 % | -55.4 % | -14.7 % | +5.1 % | +29.8 % | -35.1 % |
| branch at 1/10 learning rate | +4.7 % | +25.7 % | +5.6 % | -4.9 % | -2.5 % | +10.7 % | +24.5 % | -42.2 % | -15.5 % | +9.1 % | +25.7 % | -28.8 % |
| 1/10 learning rate + selection | +3.9 % | +1.6 % | +4.4 % | -4.1 % | -2.7 % | +5.9 % | +8.5 % | -42.0 % | -14.5 % | +2.5 % | +8.5 % | -28.2 % |
| two-stage training | +6.3 % | +11.9 % | +7.5 % | -6.5 % | +0.4 % | +44.4 % | +80.4 % | -58.5 % | -14.7 % | +20.6 % | +80.4 % | -36.6 % |
| two-stage + selection | +0.3 % | -0.2 % | +1.7 % | -4.0 % | -1.4 % | +4.4 % | +32.8 % | -58.5 % | -13.5 % | +4.8 % | +32.8 % | -36.0 % |
| two-stage + significant selection | +0.3 % | -0.2 % | -0.6 % | -3.9 % | -1.1 % | +1.7 % | +0.7 % | -57.9 % | -9.3 % | -0.4 % | +1.7 % | -33.6 % |
| two-stage + significant selection, level 0.25 | +0.3 % | -0.2 % | -0.6 % | -4.0 % | -1.5 % | +3.0 % | +0.7 % | -58.3 % | -13.5 % | -0.3 % | +3.0 % | -35.9 % |
| self head alone, last-value anchor | +0.4 % | +0.5 % | +0.1 % | -0.3 % | -0.0 % | +0.7 % | -1.1 % | +0.0 % | -0.2 % | +0.0 % | +0.7 % | -0.1 % |
| DLT-dyn, last-value anchor | +7.8 % | +14.7 % | +9.5 % | -5.5 % | +0.4 % | +20.1 % | +38.6 % | -53.9 % | -15.7 % | +12.2 % | +38.6 % | -34.8 % |
| last-value anchor + two-stage + significant selection | +1.0 % | +0.5 % | -0.0 % | -4.9 % | -0.6 % | +0.4 % | -1.4 % | -56.9 % | -7.9 % | -0.7 % | +1.0 % | -32.4 % |
| difference input + two-stage + significant selection | +0.3 % | -0.2 % | -0.6 % | -2.9 % | -1.2 % | +1.7 % | +0.7 % | -55.5 % | -8.6 % | -0.3 % | +1.7 % | -32.0 % |
| last-value anchor + difference input + two-stage + significant selection | +1.4 % | +0.5 % | -0.0 % | -4.1 % | -0.7 % | +0.5 % | -1.4 % | -55.5 % | -7.9 % | -0.5 % | +1.4 % | -31.7 % |

