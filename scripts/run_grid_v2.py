from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

# Make "src" importable when launching as: python scripts/run_grid.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fairseq.io import load_instance
from src.fairseq.algorithms import solve
from src.fairseq.evaluate import evaluate


INITS_DEFAULT = ["round_robin", "random", "greedy_fair"]
ALGOS_DEFAULT = ["ls", "sa", "tabu", "ga"]


# -----------------------------
# Compatibility rules
# -----------------------------
def is_init_allowed(algo: str, init_method: str) -> Tuple[bool, Optional[str]]:
    # Requirement: GA cannot use round_robin
    if algo == "ga" and init_method == "round_robin":
        return False, "ga does not support round_robin"
    return True, None


def combo_key(algo: str, init_method: str) -> str:
    # stable key used in files
    return f"{init_method}__{algo}"


def combo_label(algo: str, init_method: str) -> str:
    # label used as column header in the pivot (like screenshot)
    meta_name = {
        "ls": "Local Search",
        "sa": "Simulated Annealing",
        "tabu": "Tabu Search",
        "ga": "Genetic Algorithm",
    }.get(algo, algo)

    init_name = {
        "round_robin": "Round Robin",
        "random": "Random",
        "greedy_fair": "Greedy Fair",
    }.get(init_method, init_method)

    return f"Inizializzazione: {init_name} | Meta-euristica: {meta_name}"


# -----------------------------
# Instance listing / selection
# -----------------------------
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


def select_instances(
    data_root: Path,
    classes: Optional[List[str]],
    instance_ids: List[int],
    exclude_families: Optional[List[str]],
) -> List[Tuple[str, Path]]:
    data_root = Path(data_root)
    exclude = set(exclude_families or [])
    wanted_classes = set(classes) if classes else None
    wanted_ids = set(instance_ids)

    out: List[Tuple[str, Path]] = []
    for fam, inst_dir in iter_instance_dirs(data_root):
        if fam in exclude:
            continue
        if wanted_classes is not None and fam not in wanted_classes:
            continue
        if not inst_dir.name.isdigit():
            continue
        if int(inst_dir.name) not in wanted_ids:
            continue
        out.append((fam, inst_dir))

    # keep stable ordering: by class then instance number
    out.sort(key=lambda x: (x[0], int(x[1].name) if x[1].name.isdigit() else x[1].name))
    return out


# -----------------------------
# Worker (picklable): one job -> one row
# -----------------------------
def _run_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job keys:
      class, instance_dir, instance_name, algo, init, seed, time_limit_s
    returns:
      row dict compatible with raw_results.csv
    """
    fam = job["class"]
    inst_dir = Path(job["instance_dir"])
    inst_name = job["instance_name"]
    algo = job["algo"]
    init_m = job["init"]
    tl = int(job["time_limit_s"])
    seed = int(job["seed"])

    inst = load_instance(inst_dir)

    t0 = time.time()
    sol, res = solve(
        inst,
        algo=algo,
        init_method=init_m,
        seed=seed,
        time_limit_s=tl,
        verbose=False,
    )
    ev = evaluate(inst, sol)
    dur = time.time() - t0

    return {
        "class": fam,
        "instance": inst_dir.as_posix(),
        "instance_name": inst_name,
        "algo": algo,
        "init": init_m,
        "seed": seed,
        "time_limit_s": tl,
        "feasible": bool(ev.feasible),
        "violation": float(ev.violation),
        "max_avg_completion": float(ev.max_avg_completion),
        "sum_avg_completion": float(ev.sum_avg_completion),
        "iters": int(res.n_iters),
        "elapsed_s": float(res.elapsed_s),
        "_wall_s": float(dur),  # internal for ETA (not written if you don't want it)
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="Root folder with class subfolders (e.g., data/FLE/1)")
    ap.add_argument("--out-dir", default="results/specific_summary_run", help="Output directory")

    # selection
    ap.add_argument("--classes", default="", help="Comma-separated classes to include (e.g., FLE,FSE,MLE). Empty=all")
    ap.add_argument("--instance-ids", default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated instance ids to include (default 1..10)")
    ap.add_argument("--exclude-families", default="", help="Comma-separated family names to exclude (optional)")

    # run parameters
    ap.add_argument("--algos", default=",".join(ALGOS_DEFAULT), help="Comma-separated algos: ls,sa,tabu,ga")
    ap.add_argument("--inits", default=",".join(INITS_DEFAULT), help="Comma-separated init methods")
    ap.add_argument("--time-limit-s", type=int, default=180, help="Single time limit in seconds (e.g., 180 for 3min)")
    ap.add_argument("--seed", type=int, default=1, help="Single seed")

    # NEW: parallelism
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (processes). 1=sequential")

    # resume & logs
    ap.add_argument("--resume", action="store_true", help="Resume if raw_results.csv exists (skip completed rows)")
    ap.add_argument("--verbose", action="store_true", help="Print START/DONE logs for each completed run")
    ap.add_argument("--heartbeat-s", type=float, default=30.0, help="Heartbeat every N seconds")
    ap.add_argument("--flush-every", type=int, default=20, help="Flush CSV buffer every N rows")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = parse_csv_list(args.classes) if args.classes else None
    instance_ids = parse_int_list(args.instance_ids)
    exclude = parse_csv_list(args.exclude_families) if args.exclude_families else []

    algos = parse_csv_list(args.algos)
    inits = parse_csv_list(args.inits)
    tl = int(args.time_limit_s)
    seed = int(args.seed)

    # build allowed combos
    combos: List[Tuple[str, str]] = []
    for algo in algos:
        for init_m in inits:
            ok, _ = is_init_allowed(algo, init_m)
            if ok:
                combos.append((algo, init_m))
    # stable column order: by init then algo
    combos.sort(key=lambda x: (x[1], x[0]))

    instances = select_instances(
        data_root=data_root,
        classes=classes,
        instance_ids=instance_ids,
        exclude_families=exclude,
    )

    raw_path = out_dir / "raw_results.csv"
    summary_csv_path = out_dir / "specific_summary.csv"
    summary_json_path = out_dir / "specific_summary.json"

    print(f"Specific summary run")
    print(f" - data_root: {data_root}")
    print(f" - out_dir:   {out_dir}")
    print(f" - time_limit_s: {tl}  seed: {seed}")
    print(f" - selected instances: {len(instances)}  (ids={instance_ids})")
    if classes:
        print(f" - selected classes: {classes}")
    if exclude:
        print(f" - excluded families: {exclude}")
    print(f" - combos: {len(combos)}  (GA excludes round_robin)")
    print(f" - workers: {args.workers}")

    # resume
    done_keys = set()
    if args.resume and raw_path.exists():
        try:
            df_old = pd.read_csv(raw_path)
            for _, r in df_old.iterrows():
                done_keys.add((r["class"], r["instance_name"], r["algo"], r["init"], int(r["time_limit_s"]), int(r["seed"])))
            print(f"[resume] loaded {len(done_keys)} completed runs from {raw_path}")
        except Exception as e:
            print(f"[resume] failed reading raw_results.csv: {e!r} (resume disabled)")

    # header
    if not raw_path.exists():
        pd.DataFrame([], columns=[
            "class", "instance", "instance_name",
            "algo", "init",
            "seed", "time_limit_s",
            "feasible", "violation",
            "max_avg_completion", "sum_avg_completion",
            "iters", "elapsed_s"
        ]).to_csv(raw_path, index=False)

    rows_buffer: List[Dict[str, Any]] = []
    started = time.time()
    last_heartbeat = time.time()
    completed_wall: List[float] = []
    skipped = 0
    done = 0

    # build jobs list (skip already done)
    jobs: List[Dict[str, Any]] = []
    for fam, inst_dir in instances:
        inst_name = inst_dir.name
        for algo, init_m in combos:
            key = (fam, inst_name, algo, init_m, tl, seed)
            if key in done_keys:
                skipped += 1
                continue
            jobs.append({
                "class": fam,
                "instance_dir": inst_dir.as_posix(),
                "instance_name": inst_name,
                "algo": algo,
                "init": init_m,
                "seed": seed,
                "time_limit_s": tl,
            })

    total_runs = skipped + len(jobs)
    print(f" - scheduled jobs: {len(jobs)}  skipped: {skipped}  total: {total_runs}")

    def flush():
        nonlocal rows_buffer
        if rows_buffer:
            # drop internal fields
            dfw = pd.DataFrame(rows_buffer)
            if "_wall_s" in dfw.columns:
                dfw = dfw.drop(columns=["_wall_s"])
            dfw.to_csv(raw_path, mode="a", header=False, index=False)
            rows_buffer.clear()

    # sequential path
    if args.workers <= 1:
        for job in jobs:
            current_desc = f'{job["class"]}/{job["instance_name"]} algo={job["algo"]} init={job["init"]} TL={tl}s seed={seed}'
            now = time.time()
            if now - last_heartbeat >= args.heartbeat_s:
                done_all = skipped + done
                avg = (sum(completed_wall) / len(completed_wall)) if completed_wall else None
                eta_str = "n/a"
                if avg is not None:
                    remaining = max(0, total_runs - done_all)
                    eta_str = f"{(remaining * avg) / 60:.1f} min"
                print(f"[heartbeat] done={done_all}/{total_runs} current={current_desc} ETA~{eta_str}")
                last_heartbeat = now
                flush()

            if args.verbose:
                print(f"[START] {current_desc}")

            row = _run_one(job)
            completed_wall.append(float(row.get("_wall_s", tl)))
            done += 1
            rows_buffer.append(row)

            if len(rows_buffer) >= args.flush_every:
                flush()

            if args.verbose:
                print(
                    f"[DONE ] {current_desc} "
                    f"wall={row.get('_wall_s', 0.0):.1f}s feasible={row['feasible']} "
                    f"viol={row['violation']:.3f} maxavg={row['max_avg_completion']:.4f}"
                )

            if done % 10 == 0:
                elapsed = time.time() - started
                avg = sum(completed_wall) / max(1, len(completed_wall))
                remaining = max(0, total_runs - (skipped + done))
                eta_s = remaining * avg
                print(f"[{skipped+done}/{total_runs}] avg_run={avg:.1f}s ETA~{eta_s/60:.1f} min (elapsed {elapsed/60:.1f} min)")

        flush()

    # parallel path
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(_run_one, j) for j in jobs]

            for fut in as_completed(futures):
                row = fut.result()
                done += 1
                completed_wall.append(float(row.get("_wall_s", tl)))
                rows_buffer.append(row)

                if args.verbose:
                    desc = f'{row["class"]}/{row["instance_name"]} algo={row["algo"]} init={row["init"]} TL={row["time_limit_s"]}s seed={row["seed"]}'
                    print(
                        f"[DONE ] {desc} "
                        f"wall={row.get('_wall_s', 0.0):.1f}s feasible={row['feasible']} "
                        f"viol={row['violation']:.3f} maxavg={row['max_avg_completion']:.4f}"
                    )

                if len(rows_buffer) >= args.flush_every:
                    flush()

                now = time.time()
                if now - last_heartbeat >= args.heartbeat_s:
                    done_all = skipped + done
                    avg = (sum(completed_wall) / len(completed_wall)) if completed_wall else None
                    eta_str = "n/a"
                    if avg is not None:
                        remaining = max(0, total_runs - done_all)
                        eta_str = f"{(remaining * avg) / 60:.1f} min"
                    print(f"[heartbeat] done={done_all}/{total_runs} workers={args.workers} ETA~{eta_str}")
                    last_heartbeat = now
                    flush()

        flush()

    print(f"Done. Wrote raw: {raw_path}")

    # -----------------------------
    # Build the screenshot-like pivot
    # -----------------------------
    df = pd.read_csv(raw_path)

    # keep only this run's TL/seed (so the summary is clean)
    df = df[(df["time_limit_s"] == tl) & (df["seed"] == seed)].copy()

    # only selected instances (safety)
    if classes:
        df = df[df["class"].isin(classes)]
    df = df[df["instance_name"].astype(str).isin([str(i) for i in instance_ids])]

    # Create "combo" column label
    df["combo_key"] = df.apply(lambda r: combo_key(r["algo"], r["init"]), axis=1)
    df["combo_label"] = df.apply(lambda r: combo_label(r["algo"], r["init"]), axis=1)

    # mean max_avg_completion over instances per class+combo
    grouped = df.groupby(["class", "combo_key", "combo_label"]).agg(
        mean_maxavg=("max_avg_completion", "mean"),
        n=("max_avg_completion", "count"),
        feasible_rate=("feasible", "mean"),
        viol_mean=("violation", "mean"),
    ).reset_index()

    # pivot: rows=class, cols=combo_label, values=mean_maxavg
    pivot = grouped.pivot_table(
        index="class",
        columns="combo_label",
        values="mean_maxavg",
        aggfunc="mean"
    )

    # enforce stable column ordering according to combos list
    ordered_labels = [combo_label(algo, init_m) for (algo, init_m) in combos]
    pivot = pivot.reindex(columns=[c for c in ordered_labels if c in pivot.columns])

    pivot_out = pivot.reset_index()
    pivot_out.to_csv(summary_csv_path, index=False)
    print(f"Wrote: {summary_csv_path}")

    summary_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "time_limit_s": tl,
        "seed": seed,
        "selected_classes": classes or "ALL",
        "selected_instance_ids": instance_ids,
        "workers": int(args.workers),
        "combos": [{"algo": a, "init": i, "key": combo_key(a, i), "label": combo_label(a, i)} for (a, i) in combos],
        "kpi": {
            "n_rows_raw": int(len(df)),
            "n_classes": int(df["class"].nunique()) if len(df) else 0,
            "n_instances": int(df[["class", "instance_name"]].drop_duplicates().shape[0]) if len(df) else 0,
            "feasible_rate": float(df["feasible"].mean()) if len(df) else 0.0,
            "viol_mean": float(df["violation"].mean()) if len(df) else 0.0,
        },
        "table": {
            "columns": ["class"] + [c for c in pivot.columns.tolist()],
            "rows": pivot_out.to_dict(orient="records"),
        },
        "raw_results_csv": str(raw_path),
        "summary_csv": str(summary_csv_path),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote: {summary_json_path}")


if __name__ == "__main__":
    main()