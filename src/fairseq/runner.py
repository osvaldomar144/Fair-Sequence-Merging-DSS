from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import pandas as pd
import numpy as np

from src.fairseq.io import load_instance
from src.fairseq.algorithms import solve, AlgoResult
from src.fairseq.evaluate import evaluate


def _iter_instance_dirs(data_root: Path):
    data_root = Path(data_root)
    for class_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for inst_dir in sorted(
            [p for p in class_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else p.name
        ):
            yield class_dir.name, inst_dir


def run_instance(instance_dir: Path, algo: str, time_limit_s: int, seed: int, init_method: str, verbose: bool = False) -> AlgoResult:
    inst = load_instance(instance_dir)
    _, res = solve(inst, algo=algo, init_method=init_method, seed=seed, time_limit_s=time_limit_s, verbose=verbose)
    return res


def _build_diagnostics(inst, sol, ev) -> Dict[str, Any]:
    """Extra info for human-friendly interpretation (does not affect algorithms)."""
    avg_by_agent = ev.avg_by_agent
    worst_idx = int(np.argmax(avg_by_agent)) if len(avg_by_agent) else -1
    worst_avg = float(avg_by_agent[worst_idx]) if worst_idx >= 0 else 0.0

    # counts distribution of worst agent over slots
    worst_counts = sol.counts[worst_idx].astype(int).tolist() if worst_idx >= 0 else []
    # slot usage / slack
    slot_loads = ev.slot_loads.astype(float).tolist()
    capacities = inst.capacities.astype(float).tolist()
    slot_slack = (inst.capacities - ev.slot_loads).astype(float).tolist()

    return {
        "avg_by_agent": [float(x) for x in avg_by_agent.tolist()],
        "worst_agent_idx": worst_idx,
        "worst_agent_avg_completion": worst_avg,
        "worst_agent_counts_by_slot": worst_counts,
        "slot_loads": slot_loads,
        "slot_capacities": capacities,
        "slot_slack": slot_slack,
    }


def run_instance_trace(
    instance_dir: Path,
    algo: str,
    time_limit_s: int,
    seed: int,
    init_method: str,
    log_every_s: float = 1.0,
    verbose: bool = False,
    external_recorder: Optional[Callable[[float, float, float, float], None]] = None,
    return_diagnostics: bool = False,
) -> Tuple:
    """
    Run a single instance and record best-so-far trace for dashboard/plots.

    Returns:
      if return_diagnostics=False:
        (AlgoResult, trace)
      else:
        (AlgoResult, trace, diagnostics)

    trace item keys: t, max_avg_completion, sum_avg_completion, obj
    """
    inst = load_instance(instance_dir)
    trace: List[Dict[str, float]] = []

    def recorder(elapsed, max_avg, sum_avg, obj):
        p = {
            "t": float(elapsed),
            "max_avg_completion": float(max_avg),
            "sum_avg_completion": float(sum_avg),
            "obj": float(obj),
        }
        trace.append(p)
        if external_recorder is not None:
            external_recorder(p["t"], p["max_avg_completion"], p["sum_avg_completion"], p["obj"])

    sol, res = solve(
        inst,
        algo=algo,
        init_method=init_method,
        seed=seed,
        time_limit_s=time_limit_s,
        verbose=verbose,
        recorder=recorder,
        log_every_s=log_every_s,
    )

    if not return_diagnostics:
        return res, trace

    ev = evaluate(inst, sol)
    diagnostics = _build_diagnostics(inst, sol, ev)
    return res, trace, diagnostics


def run_batch(data_root: Path, out_dir: Path, algos: List[str], time_limits: List[int], seeds: List[int], init_method: str, verbose: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for cls, inst_dir in _iter_instance_dirs(data_root):
        inst = load_instance(inst_dir)
        for tl in time_limits:
            for seed in seeds:
                for algo in algos:
                    sol, res = solve(inst, algo=algo, init_method=init_method, seed=seed, time_limit_s=tl, verbose=verbose)
                    ev = evaluate(inst, sol)
                    row = {
                        "class": cls,
                        "instance": inst_dir.as_posix(),
                        "algo": algo,
                        "init": init_method,
                        "seed": seed,
                        "time_limit_s": tl,
                        "feasible": ev.feasible,
                        "violation": ev.violation,
                        "max_avg_completion": ev.max_avg_completion,
                        "sum_avg_completion": ev.sum_avg_completion,
                        "iters": res.n_iters,
                        "elapsed_s": res.elapsed_s,
                    }
                    rows.append(row)
                    if verbose:
                        print(row)

    raw_path = out_dir / "raw_results.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["class", "algo", "time_limit_s"]).agg(
        runs=("max_avg_completion", "count"),
        feasible_rate=("feasible", "mean"),
        maxavg_mean=("max_avg_completion", "mean"),
        maxavg_median=("max_avg_completion", "median"),
        sumavg_mean=("sum_avg_completion", "mean"),
        sumavg_median=("sum_avg_completion", "median"),
    ).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    best_rows = []
    for (cls, inst_path, tl), g in df.groupby(["class", "instance", "time_limit_s"]):
        g2 = g.groupby("algo")["max_avg_completion"].median().reset_index().sort_values("max_avg_completion")
        best_algo = g2.iloc[0]["algo"]
        best_val = float(g2.iloc[0]["max_avg_completion"])
        best_rows.append({"class": cls, "instance": inst_path, "time_limit_s": tl, "best_algo": best_algo, "best_median_maxavg": best_val})
    pd.DataFrame(best_rows).to_csv(out_dir / "best_by_instance.csv", index=False)

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {out_dir/'summary.csv'}")
    print(f"Wrote: {out_dir/'best_by_instance.csv'}")