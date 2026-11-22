# DLT-dyn in the efficiency setting of the S-Mamba paper: recipes ranked by validation loss

Batch 16, epochs of the official S-Mamba scripts, patience 3, no AMP. * = selected (lowest validation loss).
s / epoch: wall-clock time of the tuning run on a shared CPU (one thread per run, several runs in parallel);
it is not an efficiency measurement (see scripts/efficiency/run_efficiency.py --setting smamba).

## ETTm1, horizon 96 (published: S-Mamba 0.333 / 0.368, iTransformer 0.334 / 0.368)

| recipe | seeds | validation loss | test MSE | test MAE | epochs | best epoch | s / epoch | change vs S-Mamba |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1 * | 1 | 0.3824 | 0.3229 | 0.3598 | 10.0 | 7.0 | 12 | -3.0 % |
| r4-L96 | 1 | 0.3858 | 0.3198 | 0.3584 | 10.0 | 7.0 | 9 | -4.0 % |
| r2-lr1e-3 | 1 | 0.3859 | 0.3306 | 0.3653 | 6.0 | 3.0 | 9 | -0.7 % |
| r3-R8 | 1 | 0.3883 | 0.3275 | 0.3643 | 8.0 | 5.0 | 10 | -1.6 % |
| r5-k4 | 1 | 0.3917 | 0.3256 | 0.3628 | 10.0 | 10.0 | 8 | -2.2 % |

Selected recipe: `r1`: `--dlt_variant dltdyn --dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl basis`, learning rate 0.003.

## ECL, horizon 96 (published: S-Mamba 0.139 / 0.235, iTransformer 0.148 / 0.240)

| recipe | seeds | validation loss | test MSE | test MAE | epochs | best epoch | s / epoch | change vs S-Mamba |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r5-R8k16 * | 1 | 0.1450 | 0.1697 | 0.2643 | 5.0 | 5.0 | 949 | +22.1 % |
| r3-R8 | 1 | 0.1459 | 0.1681 | 0.2614 | 5.0 | 5.0 | 337 | +20.9 % |
| r4-k16 | 1 | 0.1487 | 0.1710 | 0.2642 | 5.0 | 5.0 | 526 | +23.0 % |
| r1 | 1 | 0.1503 | 0.1716 | 0.2624 | 5.0 | 5.0 | 207 | +23.4 % |
| r2-lr1e-3 | 1 | 0.1524 | 0.1738 | 0.2642 | 5.0 | 5.0 | 199 | +25.1 % |

Selected recipe: `r5-R8k16`: `--dlt_variant dltdyn --dlt_k 16 --dlt_L 48 --dlt_R 8 --dlt_impl basis`, learning rate 0.003.

## Traffic, horizon 96 (published: S-Mamba 0.382 / 0.261, iTransformer 0.395 / 0.268)

| recipe | seeds | validation loss | test MSE | test MAE | epochs | best epoch | s / epoch | change vs S-Mamba |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r4-k16 * | 1 | 0.4315 | 0.4988 | 0.3423 | 9.0 | 6.0 | 997 | +30.6 % |
| r3-R8 | 1 | 0.4357 | 0.5104 | 0.3445 | 10.0 | 8.0 | 957 | +33.6 % |
| r1 | 1 | 0.4394 | 0.5092 | 0.3437 | 10.0 | 10.0 | 889 | +33.3 % |

Selected recipe: `r4-k16`: `--dlt_variant dltdyn --dlt_k 16 --dlt_L 48 --dlt_R 4 --dlt_impl basis`, learning rate 0.003.

## PEMS07, horizon 12 (published: S-Mamba 0.063 / 0.159, iTransformer 0.067 / 0.165)

| recipe | seeds | validation loss | test MSE | test MAE | epochs | best epoch | s / epoch | change vs S-Mamba |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r4-k16 * | 1 | 0.0678 | 0.0694 | 0.1726 | 10.0 | 8.0 | 1196 | +10.2 % |
| r3-R8 | 1 | 0.0692 | 0.0701 | 0.1732 | 10.0 | 9.0 | 1199 | +11.3 % |
| r1 | 1 | 0.0725 | 0.0732 | 0.1783 | 10.0 | 10.0 | 1084 | +16.2 % |

Selected recipe: `r4-k16`: `--dlt_variant dltdyn --dlt_k 16 --dlt_L 48 --dlt_R 4 --dlt_impl basis`, learning rate 0.003.

`eff_tune_superseded.tsv`: the first run of r1 on ETTm1, on the kernel path; it was repeated on
the basis path used by all other runs and gave the same validation loss and test MSE to four decimals.

Time and memory of the selected recipes and of the baselines in this setting (one command; the recipes are
read from the tuning records):

    python scripts/efficiency/run_efficiency.py --setting smamba --device gpu --tuned

