# Sensitivity of DLT-dyn and neighbour selection

Test MSE, mean ± std over the seeds; in brackets the change against the default (L 48, R 4, k 8, CCF top-8
neighbours: the P0 runs) and a Welch t-test over the seeds.

## One factor at a time (DLT-dyn)

| factor | value | ETTh1-96 | Weather-96 | PEMS08-96 |
|---|---|---:|---:|---:|
| maximum lag L | 12 | 0.4063 ± 0.0095 (-0.4 %, p 0.819) | 0.1643 ± 0.0008 (+0.5 %, p 0.196) | 0.5244 ± 0.0197 (+2.8 %, p 0.344) |
| maximum lag L | 24 | 0.4080 ± 0.0049 (+0.1 %, p 0.958) | 0.1636 ± 0.0004 (+0.1 %, p 0.564) | 0.5177 ± 0.0105 (+1.5 %, p 0.402) |
| maximum lag L | 48 (default) | 0.4078 ± 0.0045 | 0.1634 ± 0.0005 | 0.5103 ± 0.0088 |
| maximum lag L | 96 | 0.4065 ± 0.0029 (-0.3 %, p 0.707) | 0.1631 ± 0.0007 (-0.2 %, p 0.617) | 0.5126 ± 0.0049 (+0.5 %, p 0.712) |
| basis size R | 2 | 0.4013 ± 0.0042 (-1.6 %, p 0.141) | 0.1660 ± 0.0007 (+1.6 %, p 0.009) | 0.5690 ± 0.0251 (+11.5 %, p 0.044) |
| basis size R | 4 (default) | 0.4078 ± 0.0045 | 0.1634 ± 0.0005 | 0.5103 ± 0.0088 |
| basis size R | 8 | 0.4058 ± 0.0017 (-0.5 %, p 0.530) | 0.1622 ± 0.0013 (-0.8 %, p 0.234) | 0.4841 ± 0.0155 (-5.1 %, p 0.080) |
| neighbours k | 4 | 0.4009 ± 0.0045 (-1.7 %, p 0.136) | 0.1677 ± 0.0012 (+2.6 %, p 0.013) | 0.5707 ± 0.0058 (+11.8 %, p 0.001) |
| neighbours k | 8 (default) | 0.4078 ± 0.0045 | 0.1634 ± 0.0005 | 0.5103 ± 0.0088 |
| neighbours k | 16 | = k 8 (capped at 6) | 0.1598 ± 0.0030 (-2.2 %, p 0.172) | 0.4863 ± 0.0190 (-4.7 %, p 0.149) |

## Neighbour selection

| model | neighbours | Weather-96 | PEMS08-96 |
|---|---|---:|---:|
| DLT | CCF top-8 (default) | 0.1675 ± 0.0005 | 0.8356 ± 0.0045 |
| DLT | 8 random | 0.1710 ± 0.0012 (+2.1 %, p 0.021) | 0.7421 ± 0.0209 (-11.2 %, p 0.013) |
| DLT | all other variables | 0.1641 ± 0.0006 (-2.0 %, p 0.002) | 0.4958 ± 0.0021 (-40.7 %, p 0.000) |
| DLT-dyn | CCF top-8 (default) | 0.1634 ± 0.0005 | 0.5103 ± 0.0088 |
| DLT-dyn | 8 random | 0.1646 ± 0.0020 (+0.7 %, p 0.408) | 0.4984 ± 0.0064 (-2.3 %, p 0.139) |
| DLT-dyn | all other variables | 0.1612 ± 0.0030 (-1.4 %, p 0.334) | 0.4150 ± 0.0119 (-18.7 %, p 0.001) |
