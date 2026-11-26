# Priority-0 experiments: kernel and gate comparison

156 runs. Protocol of the official loaders, look-back 96, k = 8 neighbours, L = 48, R = 4, top-K = 5, Adam lr 3e-3 (halved every epoch), batch 32, 10 epochs, early stopping patience 3 on the validation MSE, seeds 2021/2022/2023. Entries: mean ± std over seeds; n < 3 seeds is marked.

## Test MSE

| configuration | ETTh1_96 | Weather_96 | ECL_96 | PEMS08_24 | PEMS08_96 |
|---|---:|---:|---:|---:|---:|
| CI (no cross branch) | **0.3853 ± 0.0004** | 0.1949 ± 0.0012 | – | 0.2332 ± 0.0002 | 1.1411 ± 0.0023 |
| scalar (lag 0) | 0.3954 ± 0.0055 | 0.1736 ± 0.0004 | – | 0.1901 ± 0.0001 | 0.9909 ± 0.0190 |
| shift (CCF peak lag, LIFT-style) | 0.3955 ± 0.0065 | 0.1743 ± 0.0006 | – | 0.1940 ± 0.0001 | 1.0104 ± 0.0126 |
| tap3 (CrossLinear-style) | 0.3903 ± 0.0007 | 0.1722 ± 0.0009 | – | 0.1845 ± 0.0004 | 0.9301 ± 0.0183 |
| full (free, lags 0..48) | 0.3965 ± 0.0038 | 0.1682 ± 0.0008 | – | 0.1781 ± 0.0008 | 0.8276 ± 0.0036 |
| |  |  |  |  |  |
| top-5 lags, free weights | 0.3923 ± 0.0011 | 0.1673 ± 0.0002 | – | 0.1817 ± 0.0001 | 0.8837 ± 0.0132 |
| MillGNN-static (top-5 + decay) | <u>0.3892 ± 0.0006</u> | 0.1700 ± 0.0002 | – | 0.1916 ± 0.0003 | 0.9700 ± 0.0041 |
| MillGNN-dyn (top-5 + decay + attention) | 0.3920 ± 0.0022 | 0.1721 ± 0.0008 | – | 0.1859 ± 0.0005 | 0.9028 ± 0.0031 |
| |  |  |  |  |  |
| scalar-dyn | 0.3958 ± 0.0036 | 0.1732 ± 0.0026 | – | 0.1488 ± 0.0010 | 0.6898 ± 0.0482 |
| **DLT** | 0.3938 ± 0.0034 | 0.1675 ± 0.0005 | – | 0.1787 ± 0.0000 | 0.8356 ± 0.0045 |
| **DLT-dyn** | 0.4078 ± 0.0045 | **0.1634 ± 0.0005** | – | **0.1326 ± 0.0023** | **0.5103 ± 0.0088** |
| |  |  |  |  |  |
| DLT-dyn + MoVE softmax gate | 0.3995 ± 0.0014 | <u>0.1637 ± 0.0022</u> | – | 0.1340 ± 0.0054 | <u>0.5258 ± 0.0072</u> |
| DLT-dyn + static gate (init 0.047) | 0.4053 ± 0.0030 | 0.1638 ± 0.0010 | – | <u>0.1329 ± 0.0011</u> | 0.5335 ± 0.0043 |

Bold: best mean; underlined: second best.

## Test MAE

| configuration | ETTh1_96 | Weather_96 | ECL_96 | PEMS08_24 | PEMS08_96 |
|---|---:|---:|---:|---:|---:|
| CI (no cross branch) | **0.3931 ± 0.0004** | 0.2340 ± 0.0008 | – | 0.3281 ± 0.0014 | 0.7980 ± 0.0036 |
| scalar (lag 0) | 0.4027 ± 0.0057 | 0.2213 ± 0.0004 | – | 0.2912 ± 0.0003 | 0.7266 ± 0.0056 |
| shift (CCF peak lag, LIFT-style) | 0.4044 ± 0.0055 | 0.2215 ± 0.0005 | – | 0.2955 ± 0.0003 | 0.7409 ± 0.0028 |
| tap3 (CrossLinear-style) | 0.3975 ± 0.0020 | 0.2196 ± 0.0012 | – | 0.2891 ± 0.0014 | 0.7098 ± 0.0112 |
| full (free, lags 0..48) | 0.4061 ± 0.0046 | 0.2166 ± 0.0008 | – | 0.2848 ± 0.0011 | 0.6571 ± 0.0027 |
| |  |  |  |  |  |
| top-5 lags, free weights | 0.3994 ± 0.0008 | 0.2150 ± 0.0002 | – | 0.2868 ± 0.0001 | 0.6860 ± 0.0053 |
| MillGNN-static (top-5 + decay) | <u>0.3955 ± 0.0007</u> | 0.2162 ± 0.0003 | – | 0.2949 ± 0.0004 | 0.7294 ± 0.0021 |
| MillGNN-dyn (top-5 + decay + attention) | 0.4007 ± 0.0041 | 0.2189 ± 0.0012 | – | 0.2928 ± 0.0006 | 0.6913 ± 0.0034 |
| |  |  |  |  |  |
| scalar-dyn | 0.4035 ± 0.0025 | 0.2198 ± 0.0025 | – | 0.2575 ± 0.0013 | 0.5793 ± 0.0131 |
| **DLT** | 0.4025 ± 0.0015 | 0.2160 ± 0.0006 | – | 0.2848 ± 0.0002 | 0.6627 ± 0.0034 |
| **DLT-dyn** | 0.4125 ± 0.0026 | **0.2098 ± 0.0012** | – | **0.2423 ± 0.0013** | **0.4919 ± 0.0054** |
| |  |  |  |  |  |
| DLT-dyn + MoVE softmax gate | 0.4090 ± 0.0019 | 0.2110 ± 0.0021 | – | 0.2450 ± 0.0061 | <u>0.4983 ± 0.0049</u> |
| DLT-dyn + static gate (init 0.047) | 0.4105 ± 0.0014 | <u>0.2106 ± 0.0010</u> | – | <u>0.2431 ± 0.0020</u> | 0.5035 ± 0.0027 |

Bold: best mean; underlined: second best.

## Relative test MSE against the channel-independent model (negative = better)

| configuration | ETTh1_96 | Weather_96 | ECL_96 | PEMS08_24 | PEMS08_96 |
|---|---:|---:|---:|---:|---:|
| scalar (lag 0) | +2.6% | -10.9% | – | -18.5% | -13.2% |
| shift (CCF peak lag, LIFT-style) | +2.6% | -10.6% | – | -16.8% | -11.5% |
| tap3 (CrossLinear-style) | +1.3% | -11.6% | – | -20.9% | -18.5% |
| full (free, lags 0..48) | +2.9% | -13.7% | – | -23.6% | -27.5% |
| top-5 lags, free weights | +1.8% | -14.1% | – | -22.1% | -22.6% |
| MillGNN-static (top-5 + decay) | +1.0% | -12.8% | – | -17.8% | -15.0% |
| MillGNN-dyn (top-5 + decay + attention) | +1.7% | -11.7% | – | -20.3% | -20.9% |
| scalar-dyn | +2.7% | -11.1% | – | -36.2% | -39.5% |
| **DLT** | +2.2% | -14.0% | – | -23.4% | -26.8% |
| **DLT-dyn** | +5.8% | -16.2% | – | -43.1% | -55.3% |
| DLT-dyn + MoVE softmax gate | +3.7% | -16.0% | – | -42.6% | -53.9% |
| DLT-dyn + static gate (init 0.047) | +5.2% | -15.9% | – | -43.0% | -53.3% |

## DLT-dyn against the other kernels (Welch t-test on the seed means)

Δ = MSE(other) / MSE(DLT-dyn) − 1; positive means DLT-dyn is better. p: two-sided Welch t-test, 3 vs 3 seeds.

| other kernel | ETTh1_96 | Weather_96 | ECL_96 | PEMS08_24 | PEMS08_96 |
|---|---|---|---|---|---|
| full (free, lags 0..48) | -2.8% (p=0.031) | +2.9% (p=0.002) | – | +34.3% (p=0.000) | +62.2% (p=0.000) |
| top-5 lags, free weights | -3.8% (p=0.022) | +2.4% (p=0.003) | – | +37.0% (p=0.001) | +73.2% (p=0.000) |
| MillGNN-static (top-5 + decay) | -4.6% (p=0.018) | +4.0% (p=0.000) | – | +44.5% (p=0.000) | +90.1% (p=0.000) |
| MillGNN-dyn (top-5 + decay + attention) | -3.9% (p=0.013) | +5.3% (p=0.000) | – | +40.2% (p=0.000) | +76.9% (p=0.000) |
| scalar-dyn | -3.0% (p=0.025) | +6.0% (p=0.019) | – | +12.2% (p=0.003) | +35.2% (p=0.020) |
| **DLT** | -3.4% (p=0.015) | +2.5% (p=0.001) | – | +34.8% (p=0.001) | +63.8% (p=0.000) |

## Gates on DLT-dyn

| gate | ETTh1_96 | Weather_96 | ECL_96 | PEMS08_24 | PEMS08_96 |
|---|---|---|---|---|---|
| CI (no cross branch) | 0.3853 ± 0.0004 (best ep. 5.3) | 0.1949 ± 0.0012 (best ep. 8.0) | – | 0.2332 ± 0.0002 (best ep. 7.0) | 1.1411 ± 0.0023 (best ep. 4.0) |
| **DLT-dyn** | 0.4078 ± 0.0045 (best ep. 1.3) | 0.1634 ± 0.0005 (best ep. 8.3) | – | 0.1326 ± 0.0023 (best ep. 9.3) | 0.5103 ± 0.0088 (best ep. 9.3) |
| DLT-dyn + MoVE softmax gate | 0.3995 ± 0.0014 (best ep. 1.7) | 0.1637 ± 0.0022 (best ep. 6.0) | – | 0.1340 ± 0.0054 (best ep. 9.0) | 0.5258 ± 0.0072 (best ep. 8.7) |
| DLT-dyn + static gate (init 0.047) | 0.4053 ± 0.0030 (best ep. 1.0) | 0.1638 ± 0.0010 (best ep. 3.0) | – | 0.1329 ± 0.0011 (best ep. 9.3) | 0.5335 ± 0.0043 (best ep. 8.7) |

## Training: epochs run / best epoch / seconds per epoch (1 CPU thread) / parameters

| configuration | ETTh1_96 | Weather_96 | ECL_96 | PEMS08_24 | PEMS08_96 |
|---|---|---|---|---|---|
| CI (no cross branch) | 8.3 / 5.3 / 0s / 9.3K | 10.0 / 8.0 / 3s / 9.3K | – | 9.3 / 7.0 / 4s / 2.3K | 7.0 / 4.0 / 5s / 9.3K |
| scalar (lag 0) | 6.3 / 3.3 / 1s / 18.7K | 9.3 / 7.7 / 6s / 18.8K | – | 9.7 / 6.7 / 9s / 6.0K | 9.0 / 7.3 / 13s / 20.0K |
| shift (CCF peak lag, LIFT-style) | 6.3 / 3.3 / 1s / 18.7K | 9.3 / 7.7 / 7s / 18.8K | – | 9.7 / 6.7 / 21s / 6.0K | 7.7 / 4.7 / 23s / 20.0K |
| tap3 (CrossLinear-style) | 8.3 / 5.3 / 1s / 18.8K | 9.3 / 7.3 / 8s / 19.1K | – | 9.7 / 7.0 / 22s / 8.7K | 10.0 / 9.0 / 25s / 22.7K |
| full (free, lags 0..48) | 5.3 / 2.3 / 2s / 20.7K | 9.7 / 7.7 / 28s / 26.9K | – | 9.7 / 8.0 / 72s / 71.3K | 9.3 / 7.3 / 73s / 85.3K |
| top-5 lags, free weights | 7.7 / 5.0 / 1s / 18.8K | 9.3 / 7.7 / 15s / 19.5K | – | 10.0 / 9.3 / 42s / 11.5K | 9.7 / 7.7 / 44s / 25.4K |
| MillGNN-static (top-5 + decay) | 9.7 / 8.3 / 1s / 18.8K | 9.7 / 8.0 / 16s / 19.3K | – | 9.7 / 7.7 / 41s / 10.1K | 8.3 / 5.7 / 45s / 24.1K |
| MillGNN-dyn (top-5 + decay + attention) | 7.0 / 4.0 / 1s / 20.8K | 10.0 / 9.0 / 19s / 21.2K | – | 10.0 / 9.7 / 48s / 10.8K | 9.3 / 7.0 / 50s / 24.8K |
| scalar-dyn | 5.7 / 2.7 / 1s / 21.3K | 7.7 / 5.7 / 8s / 21.4K | – | 10.0 / 10.0 / 19s / 8.7K | 10.0 / 9.7 / 19s / 22.6K |
| **DLT** | 7.3 / 4.3 / 2s / 19.0K | 9.0 / 6.7 / 28s / 19.5K | – | 10.0 / 9.3 / 70s / 10.3K | 8.7 / 6.3 / 72s / 24.3K |
| **DLT-dyn** | 4.3 / 1.3 / 2s / 21.7K | 10.0 / 8.3 / 30s / 22.2K | – | 10.0 / 9.3 / 65s / 13.0K | 10.0 / 9.3 / 67s / 27.0K |
| DLT-dyn + MoVE softmax gate | 4.7 / 1.7 / 2s / 21.9K | 8.0 / 6.0 / 29s / 22.4K | – | 10.0 / 9.0 / 65s / 13.2K | 10.0 / 8.7 / 67s / 27.2K |
| DLT-dyn + static gate (init 0.047) | 4.0 / 1.0 / 2s / 21.7K | 6.0 / 3.0 / 30s / 22.3K | – | 10.0 / 9.3 / 65s / 13.2K | 10.0 / 8.7 / 66s / 27.2K |

