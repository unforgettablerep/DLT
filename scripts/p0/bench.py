"""Per-step training cost of every configuration (1 CPU thread, B = 32, random data), used to plan the grid.

    OMP_NUM_THREADS=1 python scripts/p0/bench.py {ETTh1|Weather|ECL|PEMS08_24|PEMS08_96} > bench_<setting>.jsonl
epoch_s = train step x training batches + forward x validation batches (model compute only, no data loading).
"""
import sys, time, types, json
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tslib'))
from models import DLT
torch.set_num_threads(1)
setting = sys.argv[1]
dims = {'ETTh1': (7, 96, 264, 88), 'Weather': (21, 96, 1147, 162), 'ECL': (321, 96, 570, 80),
        'PEMS08_24': (170, 24, 333, 112), 'PEMS08_96': (170, 96, 332, 112)}
C, H, n_tr, n_va = dims[setting]
configs = [(v, 'none', 'kernel') for v in DLT.VARIANTS] + [('dlt', 'none', 'basis'), ('dltdyn', 'none', 'basis'),
           ('dltdyn', 'moe', 'kernel'), ('dltdyn', 'static', 'kernel')]
Xtr = torch.randn(4000, C).cumsum(0).numpy()
for v, gate, impl in configs:
    cfg = types.SimpleNamespace(task_name='long_term_forecast', seq_len=96, pred_len=H, enc_in=C, dlt_variant=v,
                                dlt_k=8, dlt_L=48, dlt_R=4, dlt_dyn_dim=16, dlt_ccf_len=8000, dlt_impl=impl,
                                dlt_topk=5, dlt_gate=gate)
    torch.manual_seed(0); m = DLT.Model(cfg); m.prepare(Xtr) if v != 'ci' else None
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    x = torch.randn(32, 96, C); y = torch.randn(32, H, C)
    def step():
        opt.zero_grad(); loss = (m(x, None, None, None) - y).pow(2).mean(); loss.backward(); opt.step()
    for _ in range(2): step()
    t = time.time(); n = 6
    for _ in range(n): step()
    ts = (time.time() - t) / n
    with torch.no_grad():
        t = time.time()
        for _ in range(n): m(x, None, None, None)
        te = (time.time() - t) / n
    ep = ts * n_tr + te * n_va
    print(json.dumps({'setting': setting, 'variant': v, 'gate': gate, 'impl': impl, 'train_ms': round(ts * 1e3, 1),
                      'eval_ms': round(te * 1e3, 1), 'epoch_s': round(ep, 1)}), flush=True)
