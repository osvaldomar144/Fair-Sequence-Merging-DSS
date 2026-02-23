from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from collections import defaultdict
from src.fairseq.io import load_instance

def main():
    data_root = Path("data")
    inst_dirs = [p for p in data_root.rglob("*") if p.is_dir() and list(p.glob("*_size.csv"))]
    inst_dirs.sort()

    stats = defaultdict(lambda: {"n":0, "nec_infeas":0})

    for d in inst_dirs:
        fam = d.parts[1] if len(d.parts) > 1 else ""
        inst = load_instance(d)
        caps = inst.capacities
        items = np.concatenate(inst.requirements) if inst.requirements else np.array([])
        sum_cap = float(np.sum(caps))
        sum_req = float(np.sum(items)) if len(items) else 0.0
        max_cap = float(np.max(caps)) if len(caps) else 0.0
        max_req = float(np.max(items)) if len(items) else 0.0

        ok = (max_req <= max_cap + 1e-9) and (sum_req <= sum_cap + 1e-9)

        stats[fam]["n"] += 1
        if not ok:
            stats[fam]["nec_infeas"] += 1

    print("Necessary-infeasible by family:")
    for fam, v in sorted(stats.items()):
        print(f"{fam:5s}  {v['nec_infeas']:2d}/{v['n']:2d}")

if __name__ == "__main__":
    main()