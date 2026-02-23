from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.fairseq.io import load_instance

def necessary_conditions(inst) -> tuple[bool, float, float, float, float]:
    caps = inst.capacities
    items = np.concatenate(inst.requirements) if inst.requirements else np.array([])
    sum_cap = float(np.sum(caps))
    sum_req = float(np.sum(items)) if len(items) else 0.0
    max_cap = float(np.max(caps)) if len(caps) else 0.0
    max_req = float(np.max(items)) if len(items) else 0.0
    ok = (max_req <= max_cap + 1e-9) and (sum_req <= sum_cap + 1e-9)
    return ok, sum_req, sum_cap, max_req, max_cap

def main():
    data_root = Path("data")  # cambia se serve
    inst_dirs = [p for p in data_root.rglob("*") if p.is_dir() and list(p.glob("*_size.csv"))]
    inst_dirs.sort()

    bad = 0
    for d in inst_dirs:
        inst = load_instance(d)
        ok, sum_req, sum_cap, max_req, max_cap = necessary_conditions(inst)
        if not ok:
            bad += 1
            print(f"[INFEASIBLE (necessary fails)] {d}  sum_req={sum_req:.1f} sum_cap={sum_cap:.1f}  max_req={max_req:.1f} max_cap={max_cap:.1f}")
    print(f"\nChecked {len(inst_dirs)} instances. Necessary-infeasible: {bad}")

if __name__ == "__main__":
    main()