# All other variables as neighbours

Test MSE, mean ± std over the seeds: the CCF top-8 (default) and all other variables as neighbours (dense
path); change of all against CCF with a Welch t-test over the seeds. Both arms from the same device.

| device | dataset | H | model | CCF top-8 | all neighbours | change |
|---|---|---:|---|---:|---:|---:|
| cpu | Weather | 96 | DLT | 0.1675 ± 0.0005 | 0.1641 ± 0.0006 | -2.0 % (p 0.002) |
| cpu | Weather | 96 | DLT-dyn | 0.1634 ± 0.0005 | 0.1612 ± 0.0030 | -1.4 % (p 0.334) |
| cpu | Weather | 192 | DLT | 0.2178 ± 0.0007 | 0.2138 ± 0.0004 | -1.8 % (p 0.002) |
| cpu | Weather | 192 | DLT-dyn | 0.2121 ± 0.0018 | 0.2110 ± 0.0009 | -0.5 % (p 0.424) |
| cpu | Weather | 336 | DLT | 0.2754 ± 0.0004 | 0.2709 ± 0.0011 | -1.6 % (p 0.011) |
| cpu | Weather | 336 | DLT-dyn | 0.2699 ± 0.0019 | 0.2671 ± 0.0032 | -1.1 % (p 0.267) |
| cpu | Weather | 720 | DLT | 0.3552 ± 0.0004 | 0.3513 ± 0.0001 | -1.1 % (p 0.002) |
| cpu | Weather | 720 | DLT-dyn | 0.3518 ± 0.0011 | 0.3485 ± 0.0002 | -0.9 % (p 0.033) |
| cpu | PEMS04 | 12 | DLT | 0.0979 ± 0.0002 | 0.0923 ± 0.0002 | -5.7 % (p 0.000) |
| cpu | PEMS04 | 12 | DLT-dyn | 0.0848 ± 0.0006 | 0.0812 ± 0.0014 | -4.2 % (p 0.032) |
| cpu | PEMS08 | 12 | DLT | 0.1016 ± 0.0001 | 0.0955 ± 0.0015 | -6.0 % (p 0.020) |
| cpu | PEMS08 | 12 | DLT-dyn | 0.0856 ± 0.0004 | 0.0886 ± 0.0020 | +3.6 % (p 0.108) |

## iTransformer + DLT-dyn (joint training, branch at 3e-3)

| device | dataset | H | alone | + DLT-dyn, CCF top-8 | vs alone | + DLT-dyn, all neighbours | vs alone | vs CCF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | PEMS08 | 12 | 0.0886 ± 0.0004 | 0.0809 ± 0.0003 | -8.7 % (p 0.000) | 0.0871 ± 0.0030 | -1.7 % (p 0.484) | +7.7 % (p 0.069) |
