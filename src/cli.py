import argparse
from pathlib import Path
from src.fairseq.runner import run_instance, run_batch

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fairseq-metaheuristics")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-instance", help="Run one algorithm on one instance folder")
    p1.add_argument("--instance-dir", type=Path, required=True)
    p1.add_argument("--algo", choices=["ls","sa","tabu","ga"], required=True)
    p1.add_argument("--time-limit", type=int, default=180)
    p1.add_argument("--seed", type=int, default=1)
    p1.add_argument("--init", choices=["round_robin","random","greedy_fair"], default="round_robin")
    p1.add_argument("--verbose", action="store_true")

    p2 = sub.add_parser("run-batch", help="Run multiple algos on all instances in a data root")
    p2.add_argument("--data-root", type=Path, required=True)
    p2.add_argument("--time-limits", type=int, nargs="+", default=[180,300,600])
    p2.add_argument("--seeds", type=int, nargs="+", default=[1,2,3])
    p2.add_argument("--algos", nargs="+", default=["ls","sa","tabu","ga"])
    p2.add_argument("--init", choices=["round_robin","random","greedy_fair"], default="round_robin")
    p2.add_argument("--out-dir", type=Path, default=Path("results"))
    p2.add_argument("--verbose", action="store_true")

    return p

def main():
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "run-instance":
        res = run_instance(
            instance_dir=args.instance_dir,
            algo=args.algo,
            time_limit_s=args.time_limit,
            seed=args.seed,
            init_method=args.init,
            verbose=args.verbose,
        )
        print(res.to_json(indent=2))
    elif args.cmd == "run-batch":
        run_batch(
            data_root=args.data_root,
            out_dir=args.out_dir,
            algos=args.algos,
            time_limits=args.time_limits,
            seeds=args.seeds,
            init_method=args.init,
            verbose=args.verbose,
        )

if __name__ == "__main__":
    main()
