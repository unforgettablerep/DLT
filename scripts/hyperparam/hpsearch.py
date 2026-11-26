"""Random validation search for DLT-dyn (20 trials, seed 2021; the test split is never used for selection).

    python scripts/hyperparam/hpsearch.py make [--datasets ETTh1 ...] [--H 96] [--trials 20] [--device cpu]
    bash scripts/utils/run_queue.sh hp
    python scripts/hyperparam/hpsearch.py select     # -> output/hyperparam/tuned.json
    python scripts/hyperparam/hpsearch.py tuned [--horizons 96 192 336 720] [--seeds ...]
    bash scripts/utils/run_queue.sh tuned
--tag <suffix> keeps a second search next to the first (e.g. PEMS at horizon 12: --tag _pems).
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import jobs  # noqa: E402
from src.utils import paths  # noqa: E402

SPACE = {
    "learning_rate": [0.001, 0.003, 0.01],
    "batch_size": [16, 32, 64],
    "dlt_k": [4, 8, 16],
    "dlt_L": [24, 48, 96],
    "dlt_R": [2, 4, 8],
    "dlt_dyn_dim": [8, 16, 32],
    "dlt_gate": ["none", "static", "moe"],
}
DEFAULT = {"learning_rate": 0.003, "batch_size": 32, "dlt_k": 8, "dlt_L": 48, "dlt_R": 4, "dlt_dyn_dim": 16, "dlt_gate": "none"}


def trials(n, seed=0):
    rng = random.Random(seed)
    out, seen = [dict(DEFAULT)], {tuple(DEFAULT.values())}
    while len(out) < n:
        c = {k: rng.choice(v) for k, v in SPACE.items()}
        if tuple(c.values()) not in seen:
            seen.add(tuple(c.values()))
            out.append(c)
    return out


def cfg_args(c):
    return "--dlt_variant dltdyn --dlt_impl basis --learning_rate {} --batch_size {} --dlt_k {} --dlt_L {} --dlt_R {} --dlt_dyn_dim {} --dlt_gate {} --train_epochs 10 --patience 3".format(
        c["learning_rate"], c["batch_size"], c["dlt_k"], c["dlt_L"], c["dlt_R"], c["dlt_dyn_dim"], c["dlt_gate"]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["make", "select", "tuned"])
    p.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"])
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--horizons", type=int, nargs="+", default=list(jobs.LONG_HORIZONS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(jobs.SEEDS))
    p.add_argument("--tag", default="", help="suffix of the suite / file names (hp<tag>, tuned<tag>)")
    a = p.parse_args()
    HP, TU = "hp" + a.tag, "tuned" + a.tag
    tuned_json = paths.OUTPUT / "hyperparam" / "{}.json".format(TU)
    if a.cmd == "make":
        g = jobs.JobList(HP, "hyperparam", a.device)
        for ds in a.datasets:
            for t, c in enumerate(trials(a.trials)):
                g.add("{}_{}_t{:02d}".format(ds, a.H, t), "DLT", ds, a.H, 2021, "{} --dlt_hp_trial {}".format(cfg_args(c), t), jobs.cost(ds, c["dlt_k"] / 8 * c["dlt_L"] / 48))
        g.write()
    elif a.cmd == "select":
        rows = jobs.records("hyperparam", HP)
        best = {}
        name = lambda r: r["des"].split("_")[0]
        for r in rows:
            if name(r) not in best or r["best_vali"] < best[name(r)]["best_vali"]:
                best[name(r)] = r
        tuned = {}
        print("| dataset | trials | best val | default val | lr | batch | k | L | R | d | gate |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, r in sorted(best.items()):
            n = sum(1 for x in rows if name(x) == ds)
            dflt = [x for x in rows if name(x) == ds and x.get("dlt_hp_trial") == 0]
            c = {
                "learning_rate": r["learning_rate"],
                "batch_size": r["batch_size"],
                "dlt_k": r["dlt_k"],
                "dlt_L": r["dlt_L"],
                "dlt_R": r["dlt_R"],
                "dlt_dyn_dim": r.get("dlt_dyn_dim") or 16,
                "dlt_gate": r["dlt_gate"],
            }
            tuned[ds] = {"config": c, "searched_at_H": r["pred_len"], "trials": n, "best_vali": r["best_vali"]}
            print("| {} | {} | {:.4f} | {:.4f} | ".format(ds, n, r["best_vali"], dflt[0]["best_vali"] if dflt else float("nan")) + " | ".join(str(v) for v in c.values()) + " |")
        tuned_json.write_text(json.dumps(tuned, indent=1))
    else:
        tuned = json.loads(tuned_json.read_text())
        g = jobs.JobList(TU, "hyperparam", a.device)
        for ds, t in tuned.items():
            for H, seed in itertools.product(a.horizons, a.seeds):
                g.add("{}_{}_tuned".format(ds, H), "DLT", ds, H, seed, cfg_args(t["config"]), jobs.cost(ds))
        g.write()


if __name__ == "__main__":
    main()
