from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import random
from collections import defaultdict

from src.fairseq.io import load_instance
from src.fairseq.algorithms import solve

ALGOS = ["ls", "sa", "tabu", "ga"]
INITS = ["round_robin", "random", "greedy_fair"]

def find_instances(data_root: Path):
    inst_dirs = [p for p in data_root.rglob("*") if p.is_dir() and list(p.glob("*_size.csv"))]
    inst_dirs.sort()
    return inst_dirs

def main():
    data_root = Path("data")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "smoke_grid.csv"

    inst_dirs = find_instances(data_root)

    # --- SMOKE SETTINGS (tienili bassi) ---
    time_limit_s = 0.2          # <<<<<< smoke vero
    seeds = [1]                 # <<<<<< 1 seed basta per smoke
    max_instances = 20          # <<<<<< prova su 20 istanze random prima
    random_subset = True        # <<<<<< cambia a False per le prime 20 in ordine

    if random_subset:
        rng = random.Random(1)
        rng.shuffle(inst_dirs)
    inst_dirs = inst_dirs[:max_instances]

    total = len(inst_dirs) * len(seeds) * len(INITS) * len(ALGOS)
    print(f"Running smoke grid: instances={len(inst_dirs)} seeds={len(seeds)} inits={len(INITS)} algos={len(ALGOS)} => runs={total}")
    print(f"time_limit_s={time_limit_s}\n")

    # Write header immediately (so you see file created)
    fieldnames = ["instance", "family", "seed", "init", "algo",
                  "feasible", "violation", "best_max_avg", "best_sum_avg", "elapsed_s", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    agg = defaultdict(lambda: {"n": 0, "feas": 0})

    run_id = 0
    for inst_dir in inst_dirs:
        inst = load_instance(inst_dir)
        family = inst_dir.parts[1] if len(inst_dir.parts) > 1 else ""

        for seed in seeds:
            for init in INITS:
                for algo in ALGOS:
                    run_id += 1
                    print(f"[{run_id:4d}/{total}] {family}/{inst_dir.name}  init={init:11s} algo={algo:4s} seed={seed}")

                    row = {
                        "instance": str(inst_dir),
                        "family": family,
                        "seed": seed,
                        "init": init,
                        "algo": algo,
                        "feasible": False,
                        "violation": float("inf"),
                        "best_max_avg": float("inf"),
                        "best_sum_avg": float("inf"),
                        "elapsed_s": 0.0,
                        "error": "",
                    }

                    try:
                        _, res = solve(inst, algo=algo, init_method=init, seed=seed, time_limit_s=time_limit_s, verbose=False)
                        row.update({
                            "feasible": bool(res.feasible),
                            "violation": float(res.violation),
                            "best_max_avg": float(res.best_max_avg),
                            "best_sum_avg": float(res.best_sum_avg),
                            "elapsed_s": float(res.elapsed_s),
                        })
                    except Exception as e:
                        row["error"] = repr(e)

                    # append immediately
                    with open(csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writerow(row)

                    key = (family, algo, init)
                    agg[key]["n"] += 1
                    agg[key]["feas"] += 1 if row["feasible"] else 0

                    # every 25 runs, print a short summary snapshot
                    if run_id % 25 == 0:
                        print("\nSnapshot feasible-rate (family, algo, init):")
                        for (fam, a, ini), v in sorted(agg.items()):
                            rate = v["feas"] / max(1, v["n"])
                            print(f"  {fam:5s} {a:4s} {ini:11s}  {v['feas']:3d}/{v['n']:3d} rate={rate:.2f}")
                        print("")

    print(f"\nDone. Wrote: {csv_path}")

if __name__ == "__main__":
    main()