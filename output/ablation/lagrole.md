# What the lags of DLT-dyn do

Suite `lagrole` (CPU, three seeds): every arm has the same model and neighbours (CCF search over lags 0..48);
only the maximum lag L of the kernels changes. Lag 0 is the lag-0 kernel with input-dependent weights (P0's
scalar-dyn); L 48 is DLT-dyn as in the paper.

## 1. Test MSE by maximum lag

| | Weather_96 MSE | vs lag 0 | p | share of the L-48 gain | PEMS08_96 MSE | vs lag 0 | p | share of the L-48 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CI (no cross branch) | 0.1949 ± 0.0012 (3) | +12.5 % | 0.001 |  | 1.1411 ± 0.0023 (3) | +65.4 % | 0.004 |  |
| lag 0 (scalar-dyn) | 0.1732 ± 0.0026 (3) |  |  |  | 0.6898 ± 0.0482 (3) |  |  |  |
| L 1 | 0.1650 ± 0.0010 (3) | -4.7 % | 0.021 | 83% | 0.5777 ± 0.0117 (3) | -16.3 % | 0.049 | 62% |
| L 2 | 0.1647 ± 0.0029 (3) | -4.9 % | 0.020 | 86% | 0.5583 ± 0.0093 (3) | -19.1 % | 0.038 | 73% |
| L 4 | 0.1635 ± 0.0009 (3) | -5.6 % | 0.015 | 99% | 0.5368 ± 0.0135 (3) | -22.2 % | 0.025 | 85% |
| L 8 | 0.1648 ± 0.0020 (3) | -4.8 % | 0.013 | 85% | 0.5300 ± 0.0134 (3) | -23.2 % | 0.022 | 89% |
| L 12 | 0.1630 ± 0.0007 (3) | -5.9 % | 0.015 | 104% | 0.5292 ± 0.0120 (3) | -23.3 % | 0.023 | 89% |
| L 48 (DLT-dyn) | 0.1634 ± 0.0005 (3) | -5.7 % | 0.019 | 100% | 0.5103 ± 0.0088 (3) | -26.0 % | 0.020 | 100% |

## 2. Change of the MSE against lag 0 by forecast step

Per-step test MSE averaged over the seeds, then over the steps of each bin; the CI row is CI against lag 0 (what
the lag-0 neighbours add, with the sign reversed). Last column: the share of the total MSE reduction against lag 0
that falls on steps 1 to 8 (8 % of the horizon).

### Weather_96

| vs lag 0 | step 1 | steps 2-4 | steps 5-8 | steps 9-16 | steps 17-32 | steps 33-64 | steps 65-96 | all | share of steps 1-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CI (no cross branch) | -1.3 % | +1.1 % | +2.9 % | +5.2 % | +8.7 % | +15.7 % | +13.0 % | +12.5 % |  |
| L 1 | -2.4 % | -2.0 % | -2.9 % | -4.3 % | -5.5 % | -5.1 % | -4.3 % | -4.7 % | 2% |
| L 2 | -0.9 % | -1.2 % | -2.1 % | -3.4 % | -4.7 % | -5.4 % | -5.0 % | -4.9 % | 1% |
| L 4 | -1.2 % | -1.6 % | -3.7 % | -5.3 % | -6.5 % | -6.2 % | -5.1 % | -5.6 % | 2% |
| L 8 | -1.7 % | -2.2 % | -3.9 % | -5.5 % | -5.9 % | -5.0 % | -4.3 % | -4.8 % | 2% |
| L 12 | -3.3 % | -4.3 % | -6.7 % | -8.2 % | -7.1 % | -6.0 % | -5.1 % | -5.9 % | 3% |
| L 48 (DLT-dyn) | -3.2 % | -4.4 % | -6.7 % | -8.2 % | -7.6 % | -5.6 % | -4.7 % | -5.7 % | 3% |

### PEMS08_96

| vs lag 0 | step 1 | steps 2-4 | steps 5-8 | steps 9-16 | steps 17-32 | steps 33-64 | steps 65-96 | all | share of steps 1-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CI (no cross branch) | +3.1 % | +6.0 % | +11.6 % | +22.1 % | +43.6 % | +77.7 % | +64.1 % | +65.4 % |  |
| L 1 | +0.6 % | +0.4 % | -0.6 % | -3.9 % | -11.2 % | -17.8 % | -16.9 % | -16.3 % | 0% |
| L 2 | -0.0 % | -0.5 % | -1.5 % | -5.0 % | -12.8 % | -20.9 % | -19.8 % | -19.1 % | 0% |
| L 4 | -0.3 % | -0.8 % | -2.1 % | -5.7 % | -14.5 % | -23.8 % | -23.4 % | -22.2 % | 0% |
| L 8 | +0.0 % | -1.1 % | -3.1 % | -7.7 % | -17.2 % | -24.4 % | -24.4 % | -23.2 % | 0% |
| L 12 | +0.3 % | -0.8 % | -3.1 % | -7.9 % | -17.5 % | -24.2 % | -24.6 % | -23.3 % | 0% |
| L 48 (DLT-dyn) | +0.9 % | +0.2 % | -1.3 % | -5.8 % | -16.5 % | -25.4 % | -29.1 % | -26.0 % | 0% |

## 3. Checks

- The arms ci, scalardyn and dltdyn against the same runs of P0: 18 runs, largest difference of the test MSE 0.0e+00.
- Weather_96 with L 12: 0.1630 ± 0.0007 with the neighbours of the search over lags 0..48, 0.1643 ± 0.0008 with those of the search over lags 0..12 (sensitivity study).
- PEMS08_96 with L 12: 0.5292 ± 0.0120 with the neighbours of the search over lags 0..48, 0.5244 ± 0.0197 with those of the search over lags 0..12 (sensitivity study).
