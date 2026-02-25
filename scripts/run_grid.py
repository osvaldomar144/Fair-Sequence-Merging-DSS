from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

# Make "src" importable when launching as: python scripts/run_grid.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fairseq.io import load_instance
from src.fairseq.algorithms import solve
from src.fairseq.evaluate import evaluate


INITS_DEFAULT = ["round_robin", "random", "greedy_fair"]
ALGOS_DEFAULT = ["ls", "sa", "tabu", "ga"]


def iter_instance_dirs(data_root: Path):
    data_root = Path(data_root)
    for class_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for inst_dir in sorted(
            [p for p in class_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        ):
            yield class_dir.name, inst_dir


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="Root folder with class subfolders (e.g., data/FLB/1)")
    ap.add_argument("--out-dir", default="results/grid", help="Output directory")
    ap.add_argument("--algos", default=",".join(ALGOS_DEFAULT), help="Comma-separated algos: ls,sa,tabu,ga")
    ap.add_argument("--inits", default=",".join(INITS_DEFAULT), help="Comma-separated init methods")
    ap.add_argument("--time-limits", default="180,300,600", help="Comma-separated time limits in seconds")
    ap.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated seeds")
    ap.add_argument("--exclude-families", default="FSE", help="Comma-separated family names to exclude (e.g., FSE)")
    ap.add_argument("--max-instances", type=int, default=0, help="If >0, run only first N instances (quick test)")
    ap.add_argument("--resume", action="store_true", help="Resume if raw_results.csv exists (skip completed rows)")

    # NEW: verbosity / heartbeat
    ap.add_argument("--verbose", action="store_true", help="Print START/DONE logs for each run")
    ap.add_argument("--heartbeat-s", type=float, default=30.0, help="Heartbeat every N seconds (even during long runs)")
    ap.add_argument("--flush-every", type=int, default=20, help="Flush CSV buffer every N rows (smaller = safer)")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    algos = parse_csv_list(args.algos)
    inits = parse_csv_list(args.inits)
    time_limits = parse_int_list(args.time_limits)
    seeds = parse_int_list(args.seeds)
    exclude = set(parse_csv_list(args.exclude_families)) if args.exclude_families else set()

    raw_path = out_dir / "raw_results.csv"

    done_keys = set()
    if args.resume and raw_path.exists():
        df_old = pd.read_csv(raw_path)
        for _, r in df_old.iterrows():
            done_keys.add((r["class"], r["instance_name"], r["algo"], r["init"], int(r["time_limit_s"]), int(r["seed"])))
        print(f"[resume] loaded {len(done_keys)} completed runs from {raw_path}")

    # Build instance list
    instances: List[Tuple[str, Path]] = []
    for fam, inst_dir in iter_instance_dirs(data_root):
        if fam in exclude:
            continue
        instances.append((fam, inst_dir))
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]

    total_runs = len(instances) * len(algos) * len(inits) * len(time_limits) * len(seeds)
    print(f"Grid: instances={len(instances)} algos={len(algos)} inits={len(inits)} TLs={len(time_limits)} seeds={len(seeds)} => runs={total_runs}")
    if exclude:
        print(f"Excluded families: {sorted(exclude)}")

    rows_buffer: List[Dict[str, Any]] = []
    started = time.time()
    k = 0
    skipped = 0

    # Heartbeat state
    last_heartbeat = time.time()
    current_run_desc = "N/A"
    completed_times: List[float] = []  # durations of completed runs (for ETA)

    # write header if new
    if not raw_path.exists():
        pd.DataFrame([], columns=[
            "class", "instance", "instance_name",
            "algo", "init", "seed", "time_limit_s",
            "feasible", "violation",
            "max_avg_completion", "sum_avg_completion",
            "iters", "elapsed_s"
        ]).to_csv(raw_path, index=False)

    def flush():
        nonlocal rows_buffer
        if rows_buffer:
            pd.DataFrame(rows_buffer).to_csv(raw_path, mode="a", header=False, index=False)
            rows_buffer.clear()

    for fam, inst_dir in instances:
        inst = load_instance(inst_dir)
        inst_name = inst_dir.name

        for tl in time_limits:
            for seed in seeds:
                for algo in algos:
                    for init_method in inits:
                        key = (fam, inst_name, algo, init_method, tl, seed)
                        if key in done_keys:
                            skipped += 1
                            continue

                        # heartbeat even before long solve()
                        now = time.time()
                        if now - last_heartbeat >= args.heartbeat_s:
                            done = skipped + k
                            avg = (sum(completed_times) / len(completed_times)) if completed_times else None
                            if avg is not None:
                                remaining = max(0, total_runs - done)
                                eta = remaining * avg
                                eta_str = f"{eta/60:.1f} min"
                            else:
                                eta_str = "n/a"
                            print(f"[heartbeat] done={done}/{total_runs}  current={current_run_desc}  ETA~{eta_str}")
                            last_heartbeat = now
                            flush()

                        current_run_desc = f"{fam}/{inst_name} algo={algo} init={init_method} TL={tl}s seed={seed}"

                        if args.verbose:
                            print(f"[START] {current_run_desc}")

                        k += 1
                        t0 = time.time()

                        sol, res = solve(
                            inst,
                            algo=algo,
                            init_method=init_method,
                            seed=seed,
                            time_limit_s=tl,
                            verbose=False
                        )

                        ev = evaluate(inst, sol)

                        dur = time.time() - t0
                        completed_times.append(dur)

                        row = {
                            "class": fam,
                            "instance": inst_dir.as_posix(),
                            "instance_name": inst_name,
                            "algo": algo,
                            "init": init_method,
                            "seed": seed,
                            "time_limit_s": tl,
                            "feasible": bool(ev.feasible),
                            "violation": float(ev.violation),
                            "max_avg_completion": float(ev.max_avg_completion),
                            "sum_avg_completion": float(ev.sum_avg_completion),
                            "iters": int(res.n_iters),
                            "elapsed_s": float(res.elapsed_s),
                        }
                        rows_buffer.append(row)

                        # flush periodically
                        if len(rows_buffer) >= args.flush_every:
                            flush()

                        if args.verbose:
                            print(
                                f"[DONE ] {current_run_desc}  "
                                f"dur={dur:.1f}s  feasible={ev.feasible}  viol={ev.violation:.3f}  "
                                f"maxavg={ev.max_avg_completion:.4f}"
                            )

                        # progress every 10 completed runs
                        if k % 10 == 0:
                            elapsed = time.time() - started
                            done = skipped + k
                            avg = sum(completed_times) / max(1, len(completed_times))
                            remaining = max(0, total_runs - done)
                            eta = remaining * avg
                            print(
                                f"[{done}/{total_runs}] last={fam}/{inst_name} algo={algo} init={init_method} TL={tl} seed={seed}  "
                                f"avg_run={avg:.1f}s  ETA~{eta/60:.1f} min"
                            )

    # final flush
    flush()

    print(f"Done. Wrote raw: {raw_path}")

    # summaries
    df = pd.read_csv(raw_path)

    combo = df.groupby(["algo", "init", "time_limit_s"]).agg(
        runs=("max_avg_completion", "count"),
        feasible_rate=("feasible", "mean"),
        maxavg_mean=("max_avg_completion", "mean"),
        maxavg_median=("max_avg_completion", "median"),
        sumavg_mean=("sum_avg_completion", "mean"),
        sumavg_median=("sum_avg_completion", "median"),
        viol_mean=("violation", "mean"),
        viol_median=("violation", "median"),
    ).reset_index().sort_values(["time_limit_s", "maxavg_median"])

    combo_path = out_dir / "summary_by_combo.csv"
    combo.to_csv(combo_path, index=False)
    print(f"Wrote: {combo_path}")

    best_rows = []
    for (fam, inst_name, tl), g in df.groupby(["class", "instance_name", "time_limit_s"]):
        g2 = g.groupby(["algo", "init"]).agg(
            med=("max_avg_completion", "median"),
            feas=("feasible", "mean"),
            vmed=("violation", "median"),
        ).reset_index().sort_values(["feas", "vmed", "med"], ascending=[False, True, True])

        top = g2.iloc[0]
        best_rows.append({
            "class": fam,
            "instance_name": inst_name,
            "time_limit_s": int(tl),
            "best_algo": top["algo"],
            "best_init": top["init"],
            "best_median_maxavg": float(top["med"]),
            "feasible_rate": float(top["feas"]),
            "median_violation": float(top["vmed"]),
        })

    best_df = pd.DataFrame(best_rows).sort_values(["time_limit_s", "best_median_maxavg"])
    best_path = out_dir / "best_by_instance.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Wrote: {best_path}")

    overview = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "excluded_families": sorted(exclude),
        "n_rows": int(len(df)),
        "combos": combo.head(50).to_dict(orient="records"),
        "best_by_instance": best_df.head(200).to_dict(orient="records"),
    }
    overview_path = out_dir / "overview.json"
    import json
    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(f"Wrote: {overview_path}")


if __name__ == "__main__":
    main()