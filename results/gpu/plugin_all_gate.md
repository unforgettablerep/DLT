# Plug-in study: gate probe, validation rule, first Electricity / Traffic runs (2026-11-15)

**Source.** [`plugin_all.jsonl`](plugin_all.jsonl): 663 records from the RTX 4090 (PyTorch 2.10.0+cu130), seed 2021.
It contains every record of [`plugin_all_stage1.jsonl`](plugin_all_stage1.jsonl) unchanged, plus 148 new ones:
- 128 runs of the gate probe: the + DLT arm with a static gate (`dlt_gate-static`; the branch starts at 4.7 %
  strength) on ETTh1, ETTh2, Exchange and Weather;
- 20 runs of stage 3: Electricity and Traffic (TimesNet complete, PatchTST and TimeXer + DLT arms so far).

The new S-Mamba records name the Mamba block: `mamba_ssm` (16 of 16). Tables:
`python scripts/p1/plugin_table.py --files results/gpu/plugin_all.jsonl [--arm dlt_gate-static]`.

## 1. The static gate does not remove the harm
Change of test MSE against the backbone alone, median over 8 backbones × 4 horizons (settings where the branch
helps); best epoch, mean:

| dataset | default branch | gated branch | gate better than default | best epoch: alone / default / gated |
|---|---:|---:|---:|---:|
| ETTh1 | +2.0 % (10 of 32) | +2.3 % (10 of 32) | 19 of 32 | 4.4 / 1.6 / 2.1 |
| ETTh2 | +9.5 % (6 of 32) | +10.0 % (5 of 32) | 12 of 32 | 4.5 / 2.4 / 2.3 |
| Exchange | +14.0 % (2 of 32) | +11.0 % (5 of 32) | 20 of 32 | 3.4 / 1.8 / 2.7 |
| Weather | −3.6 % (23 of 32) | −3.9 % (25 of 32) | 18 of 32 | 4.3 / 4.3 / 4.0 |

The gated arm is better than the default arm in about half of the settings, as chance would have it. It still
peaks one to two epochs after the start. So the harm does not come from the full strength of the branch at the
first step. A likely reason is that the gate parameter learns at the branch's rate (3e-3) and opens within the
first epoch. The gate keeps the gains on Weather (except Informer, +8.6 %) but brings nothing elsewhere; the
default recipe stays.

The remaining suspect in the recipe is the branch's learning rate, 30 times that of most backbones. The
learning-rate probe (`--dlt "--dlt_lr 0"`: the branch trains at the backbone's rate) tests it. If it fails as
well, the cause is the data alone: no stable lead-lag structure across variables.

## 2. A validation rule keeps the gains and avoids most of the harm
The rule keeps the branch only where it lowers the best validation loss of the backbone alone. It decides on
validation data only and needs no new run. Default branch, change of test MSE, mean, and settings made worse by
more than 2 %:

| dataset | settings | branch kept | always + DLT | with the rule |
|---|---:|---:|---:|---:|
| ETTh1 | 32 | 8 | −0.8 %, 16 worse | −2.9 %, 2 worse |
| ETTh2 | 32 | 6 | +6.8 %, 24 worse | −1.5 %, 1 worse |
| ETTm1 | 32 | 25 | −0.7 %, 12 worse | −2.1 %, 6 worse |
| ETTm2 | 32 | 15 | +3.8 %, 10 worse | −1.0 %, 3 worse |
| Exchange | 32 | 10 | +18.7 %, 29 worse | +7.2 %, 8 worse |
| Weather | 32 | 26 | −4.0 %, 2 worse | −4.9 %, 0 worse |
| PEMS04 | 32 | 26 | −15.7 %, 1 worse | −15.7 %, 0 worse |
| PEMS08 | 32 | 30 | −19.2 %, 1 worse | −19.4 %, 0 worse |
| all (incl. 4 Electricity) | 260 | 149 | −1.3 %, 96 worse | −5.0 %, 20 worse |

Exchange stays harmed. On Exchange the validation and test periods disagree: in 9 of 32 settings the branch
lowers the validation loss and raises the test loss. In a paper the rule is a reported model-selection step
("+ DLT, kept if it lowers the validation loss"), like the choice of any hyper-parameter on validation data.

## 3. Electricity and Traffic so far
- **Electricity, TimesNet:** −1.8 %, −3.8 % and −2.4 % at horizons 96, 192 and 336; +19.2 % at 720, where the
  validation loss is worse as well (0.211 against 0.184), so the rule would drop it. The PatchTST and TimeXer
  pairs are not complete yet.
- **Traffic, TimesNet: every test MSE is NaN**, in both arms and at all four horizons; the validation losses are
  normal (0.30 to 0.42). The cause is not in the data and not in the plug-in:
  - `traffic.csv` has no missing or infinite value;
  - the backbone alone fails too;
  - a small untrained TimesNet run over the same test windows on the CPU gives finite forecasts.

  One difference between the splits: 1224 of the 3509 test windows contain a sensor that is constant over the
  whole input window; no validation window does. `scripts/gpu/diag_nonfinite.py Traffic_96_TimesNet_alone`
  re-tests the saved checkpoint on the GPU. It finds the windows and the first layer with a non-finite value, and
  re-runs the batch with TF32 off, with cuDNN off and on the CPU. New records count non-finite forecasts
  (`test_nonfinite`), and the run log warns about them.

## 4. PEMS: some backbones alone fall far behind their published numbers
Compared with the iTransformer paper (Table 9: input 96, MSE), backbone alone at horizons 12 / 24 / 48 / 96:

| | PEMS04 | PEMS08 |
|---|---|---|
| DLinear | −1 / −1 / −1 / −1 % | 0 / 0 / +1 / +3 % |
| TimesNet | +3 / +5 / +9 / +3 % | +7 / +7 / +8 / +5 % |
| iTransformer | +4 / +5 / +9 / +16 % | +10 / +15 / **+41 / +123 %** |
| PatchTST | +34 / +45 / +84 / +173 % | −17 / −6 / +29 / +114 % |

- **The pipeline is right.** DLinear reproduces the published table to 1–3 %, so the data, the split and the
  metric are not at fault.
- **Two of these comparisons are not valid.** iTransformer on PEMS08 at horizons 48 and 96 fails to train: its
  best epoch is 1 at horizon 96, with test MSE 0.493 against 0.221 published. PatchTST has no official PEMS setting
  and stops after three epochs at horizon 96. The "+ DLT" gains there mostly repair a broken baseline.
  iTransformer + DLT on PEMS08-96 (0.249) is still 13 % worse than the published iTransformer alone.
- **The other gains hold** when the published numbers are the reference instead of our runs:

  | + DLT against the published backbone alone | PEMS04 | PEMS08 |
  |---|---:|---:|
  | DLinear | −43 to −50 % | −44 to −51 % |
  | TimesNet | −8 to −15 % | −1 to −27 % |
  | iTransformer | −2 to −7 % | not valid (see above) |
- **One cause has been found and fixed.** The iTransformer and S-Mamba code, where the official PEMS and Solar
  settings come from, gives the model no time marks there (`x_mark = None`); our runs gave five. A CPU control run
  of iTransformer PEMS08-96 (official setting) reached a validation loss of 0.576 after epoch 1 without time marks
  and 0.695 with them, and 0.526 after epoch 2 without them (training continues). plugin_all now passes
  `--no_time_marks` on PEMS and Solar; the PEMS04/08 runs are to be repeated with it.
- **PatchTST needs a PEMS setting of its own**, for example chosen on the validation loss, before its PEMS numbers
  can be compared.

## Next
1. `diag_nonfinite.py` on one Traffic TimesNet run, before the remaining Traffic runs are evaluated.
2. Learning-rate probe on ETTh2, Exchange and Weather (96 runs, separate list, next to stage 3).
3. Remaining datasets as planned: the rest of Electricity and Traffic, Solar, PEMS03, PEMS07, and PEMS04/08 again
   without time marks (all their records removed first; see GPU_RUNS.md).
