"""Markdown summary of the feasibility runs: mean MSE/MAE over seeds, change vs. CI."""
import json, sys, collections

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
order = ["ci", "scalar", "shift", "tap3", "dlt", "full", "scalardyn", "dltdyn"]
names = {"ci": "CI (no cross)", "scalar": "scalar mixing", "shift": "single-lag shift",
         "tap3": "3-tap kernel", "dlt": "DLT (static)", "full": "free kernel",
         "scalardyn": "scalar + dynamic (control)", "dltdyn": "DLT-dyn"}
g = collections.defaultdict(list)
for r in rows:
    g[(r["data"], r["H"], r["variant"])].append(r)
sets = [("ETTh1", 96), ("weather", 96), ("electricity", 96), ("PEMS08", 24), ("PEMS08", 96)]
sets = [s for s in sets if any((s[0], s[1], v) in g for v in order)]

out = ["| variant | " + " | ".join(f"{d} H={h}" for d, h in sets) + " |",
       "|---|" + "---|" * len(sets)]
best = {}
for d, h in sets:
    ms = [(sum(x["mse"] for x in g[(d, h, v)]) / len(g[(d, h, v)]), v) for v in order if (d, h, v) in g]
    best[(d, h)] = min(ms)[1]
for v in order:
    cells = []
    for d, h in sets:
        rs = g.get((d, h, v))
        if not rs:
            cells.append("–"); continue
        m = sum(x["mse"] for x in rs) / len(rs)
        base = g[(d, h, "ci")]
        b = sum(x["mse"] for x in base) / len(base)
        s = f"{m:.4f} ({100*(m-b)/b:+.1f}%)" + ("" if len(rs) > 1 else "¹")
        cells.append(f"**{s}**" if best[(d, h)] == v else s)
    out.append(f"| {names[v]} | " + " | ".join(cells) + " |")
out.append("")
out.append("MSE on z-scored test set, mean over seeds (¹ = single seed); % = change vs. CI; bold = best per column.")
out.append("")
lag = {d: next(r for r in rows if r["data"] == d)["frac_lag0"] for d, _ in sets}
out.append("Share of top-k neighbour pairs whose CCF peak is at lag 0: " +
           ", ".join(f"{d} {v:.0%}" for d, v in lag.items()))
seeds = collections.Counter(len(g[k]) for k in g)
out.append(f"Runs: {len(rows)} (cells with 2 seeds: {seeds.get(2, 0)}, 1 seed: {seeds.get(1, 0)})")
print("\n".join(out))
