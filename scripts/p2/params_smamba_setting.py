"""Parameters of the six baselines and DLT-dyn recipes in the efficiency setting of the S-Mamba paper (no training).

    python scripts/p2/params_smamba_setting.py      # -> results/p2/params_smamba_setting.json
"""
import json, os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/scripts'); os.chdir(REPO)
import torch, run, baselines
from importlib import import_module
torch.set_num_threads(1)
SETS = {'PEMS07': (883, 12), 'ECL': (321, 96), 'Traffic': (862, 96), 'ETTm1': (7, 96)}
out = []
for ds, (C, H) in SETS.items():
    for name, model, extra in [('S-Mamba', 'SMamba', ''), ('iTransformer', 'iTransformer', ''), ('PatchTST', 'PatchTST', ''),
                               ('RLinear', 'RLinear', ''), ('DLinear', 'DLinear', ''), ('TiDE', 'TiDE', ''),
                               ('DLT-dyn k8 R4', 'DLT', '--dlt_variant dltdyn --dlt_k 8 --dlt_R 4'),
                               ('DLT-dyn k8 R8', 'DLT', '--dlt_variant dltdyn --dlt_k 8 --dlt_R 8'),
                               ('DLT-dyn k16 R4', 'DLT', '--dlt_variant dltdyn --dlt_k 16 --dlt_R 4'),
                               ('DLT-dyn k16 R8', 'DLT', '--dlt_variant dltdyn --dlt_k 16 --dlt_R 8')]:
        bargs = baselines.args(model, ds, H)[0] if model != 'DLT' else '--dlt_L 48'
        argv = (f'--task_name long_term_forecast --is_training 1 --model_id p --model {model} --data custom '
                f'--seq_len 96 --label_len 48 --pred_len {H} --enc_in {C} --dec_in {C} --c_out {C} {bargs} {extra}').split()
        a = run.get_parser().parse_args(argv)
        if model == 'SMamba':
            a.mamba_impl = 'torch'
        m = import_module(f'models.{model}').Model(a)
        n = sum(p.numel() for p in m.parameters())
        out.append({'dataset': ds, 'model': name, 'params': n})
        print(ds, name, n, flush=True)
json.dump(out, open(os.path.join(REPO, 'results', 'p2', 'params_smamba_setting.json'), 'w'), indent=1)
