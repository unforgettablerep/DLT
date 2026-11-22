"""Plain-text summary of the feasibility runs:  python scripts/prototype/agg.py [output/prototype/runs.jsonl]."""
import json, sys, collections
from pathlib import Path

src = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "output" / "prototype" / "runs.jsonl"
rows = [json.loads(l) for l in open(src) if l.strip()]
order = ["ci", "scalar", "shift", "tap3", "dlt", "full", "scalardyn", "dltdyn"]
g = collections.defaultdict(list)
for r in rows:
    g[(r["data"], r["H"], r["variant"])].append(r)
keys = sorted({(d, h) for d, h, _ in g})
for d, h in keys:
    base = g.get((d, h, "ci"))
    bm = sum(x["mse"] for x in base) / len(base) if base else None
    print(f"\n== {d}  H={h}   (lag0 frac of top-k pairs: {next(iter(g[(d,h,v)] for v in order if (d,h,v) in g))[0]['frac_lag0']:.2f})")
    print(f"{'variant':8s} {'n':>2s} {'MSE':>8s} {'MAE':>8s} {'vsCI%':>7s} {'params':>8s} {'s/run':>6s} {'ep':>3s}")
    for v in order:
        rs = g.get((d, h, v))
        if not rs: continue
        m = sum(x["mse"] for x in rs) / len(rs); a = sum(x["mae"] for x in rs) / len(rs)
        rel = f"{100*(m-bm)/bm:+.1f}" if bm else "  -"
        sp = "/".join(f"{x['mse']:.4f}" for x in rs) if len(rs) > 1 else ""
        print(f"{v:8s} {len(rs):2d} {m:8.4f} {a:8.4f} {rel:>7s} {rs[0]['params']:8d} {sum(x['secs'] for x in rs)/len(rs):6.0f} {rs[0]['epochs']:3d}  {sp}")
