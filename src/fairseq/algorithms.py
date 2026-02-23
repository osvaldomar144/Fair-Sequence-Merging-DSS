from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import math
import random
import numpy as np

from src.fairseq.types import Instance, Solution
from src.fairseq.evaluate import evaluate, objective_with_penalty
from src.fairseq.neighborhood import random_move, guided_move, apply_move, Move
from src.fairseq.constructive import build_initial
from src.fairseq.repair import repair_capacity_forward


@dataclass
class AlgoResult:
    algo: str
    init: str
    seed: int
    time_limit_s: int
    best_obj: float
    best_max_avg: float
    best_sum_avg: float
    feasible: bool
    violation: float
    n_iters: int
    elapsed_s: float

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.__dict__, indent=indent)


def _maybe_record(recorder, start: float, last_log: float, log_every_s: float, best_ev, best_obj):
    if recorder is None:
        return last_log
    elapsed = time.time() - start
    if elapsed - last_log >= log_every_s:
        recorder(elapsed, best_ev.max_avg_completion, best_ev.sum_avg_completion, best_obj)
        return elapsed
    return last_log


def _init_best_tracking(inst: Instance, sol: Solution):
    ev = evaluate(inst, sol)
    obj = objective_with_penalty(ev)

    best_any_sol = sol
    best_any_ev = ev
    best_any_obj = obj

    best_feas_sol = None
    best_feas_ev = None
    best_feas_obj = float("inf")
    if ev.feasible:
        best_feas_sol = sol
        best_feas_ev = ev
        best_feas_obj = float(ev.max_avg_completion)

    return (ev, obj,
            best_any_sol, best_any_ev, best_any_obj,
            best_feas_sol, best_feas_ev, best_feas_obj)


def _update_best(inst: Instance, sol: Solution, ev, obj,
                 best_any_sol, best_any_ev, best_any_obj,
                 best_feas_sol, best_feas_ev, best_feas_obj):
    if obj < best_any_obj - 1e-12:
        best_any_sol, best_any_ev, best_any_obj = sol, ev, obj

    if ev.feasible:
        feas_obj = float(ev.max_avg_completion)
        if feas_obj < best_feas_obj - 1e-12:
            best_feas_sol, best_feas_ev, best_feas_obj = sol, ev, feas_obj

    return (best_any_sol, best_any_ev, best_any_obj,
            best_feas_sol, best_feas_ev, best_feas_obj)


# -------------------- Local Search --------------------

def local_search(inst: Instance, init_sol: Solution, time_limit_s: int, seed: int,
                 verbose: bool = False, recorder=None, log_every_s: float = 1.0) -> Tuple[Solution, AlgoResult]:
    rng = random.Random(seed)
    start = time.time()
    last_log = 0.0

    cur = init_sol
    (cur_ev, cur_obj,
     best_any_sol, best_any_ev, best_any_obj,
     best_feas_sol, best_feas_ev, best_feas_obj) = _init_best_tracking(inst, cur)

    if recorder is not None:
        ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
        ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
        recorder(0.0, ref_ev.max_avg_completion, ref_ev.sum_avg_completion, ref_obj)

    iters = 0
    while time.time() - start < time_limit_s:
        iters += 1
        ev_cur = evaluate(inst, cur)
        if ev_cur.violation > 1e-9 and rng.random() < 0.8:
            mv = guided_move(inst, cur, rng)
        else:
            mv = random_move(inst, cur, rng)
        nxt = apply_move(inst, cur, mv)
        if nxt is None:
            continue

        ev = evaluate(inst, nxt)
        obj = objective_with_penalty(ev)

        if obj < cur_obj - 1e-12:
            cur, cur_ev, cur_obj = nxt, ev, obj

        (best_any_sol, best_any_ev, best_any_obj,
         best_feas_sol, best_feas_ev, best_feas_obj) = _update_best(
            inst, nxt, ev, obj,
            best_any_sol, best_any_ev, best_any_obj,
            best_feas_sol, best_feas_ev, best_feas_obj
        )

        if recorder is not None:
            ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
            ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
            last_log = _maybe_record(recorder, start, last_log, log_every_s, ref_ev, ref_obj)

    if best_feas_sol is not None:
        final_sol = best_feas_sol
        final_ev = best_feas_ev
        final_obj = best_feas_obj
    else:
        final_sol = best_any_sol
        final_ev = best_any_ev
        final_obj = best_any_obj

    res = AlgoResult(
        algo="ls",
        init="custom",
        seed=seed,
        time_limit_s=time_limit_s,
        best_obj=float(final_obj),
        best_max_avg=float(final_ev.max_avg_completion),
        best_sum_avg=float(final_ev.sum_avg_completion),
        feasible=bool(final_ev.feasible),
        violation=float(final_ev.violation),
        n_iters=iters,
        elapsed_s=float(time.time() - start),
    )
    return final_sol, res


# -------------------- Simulated Annealing --------------------

def simulated_annealing(inst: Instance, init_sol: Solution, time_limit_s: int, seed: int,
                        verbose: bool = False, recorder=None, log_every_s: float = 1.0) -> Tuple[Solution, AlgoResult]:
    rng = random.Random(seed)
    start = time.time()
    last_log = 0.0

    cur = init_sol
    (cur_ev, cur_obj,
     best_any_sol, best_any_ev, best_any_obj,
     best_feas_sol, best_feas_ev, best_feas_obj) = _init_best_tracking(inst, cur)

    if recorder is not None:
        ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
        ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
        recorder(0.0, ref_ev.max_avg_completion, ref_ev.sum_avg_completion, ref_obj)

    T0 = 1.0
    Tf = 1e-3
    iters = 0

    while time.time() - start < time_limit_s:
        iters += 1
        frac = (time.time() - start) / max(1e-9, float(time_limit_s))
        temp = T0 * (Tf / T0) ** frac

        ev_cur = evaluate(inst, cur)
        if ev_cur.violation > 1e-9 and rng.random() < 0.8:
            mv = guided_move(inst, cur, rng)
        else:
            mv = random_move(inst, cur, rng)
        nxt = apply_move(inst, cur, mv)
        if nxt is None:
            continue

        ev = evaluate(inst, nxt)
        obj = objective_with_penalty(ev)

        delta = obj - cur_obj
        accept = False
        if delta <= 0:
            accept = True
        else:
            if temp > 0:
                p = math.exp(-delta / temp)
                accept = (rng.random() < p)

        if accept:
            cur, cur_ev, cur_obj = nxt, ev, obj

        (best_any_sol, best_any_ev, best_any_obj,
         best_feas_sol, best_feas_ev, best_feas_obj) = _update_best(
            inst, nxt, ev, obj,
            best_any_sol, best_any_ev, best_any_obj,
            best_feas_sol, best_feas_ev, best_feas_obj
        )

        if recorder is not None:
            ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
            ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
            last_log = _maybe_record(recorder, start, last_log, log_every_s, ref_ev, ref_obj)

    if best_feas_sol is not None:
        final_sol = best_feas_sol
        final_ev = best_feas_ev
        final_obj = best_feas_obj
    else:
        final_sol = best_any_sol
        final_ev = best_any_ev
        final_obj = best_any_obj

    res = AlgoResult(
        algo="sa",
        init="custom",
        seed=seed,
        time_limit_s=time_limit_s,
        best_obj=float(final_obj),
        best_max_avg=float(final_ev.max_avg_completion),
        best_sum_avg=float(final_ev.sum_avg_completion),
        feasible=bool(final_ev.feasible),
        violation=float(final_ev.violation),
        n_iters=iters,
        elapsed_s=float(time.time() - start),
    )
    return final_sol, res


# -------------------- Tabu Search --------------------

def tabu_search(inst: Instance, init_sol: Solution, time_limit_s: int, seed: int,
                verbose: bool = False, recorder=None, log_every_s: float = 1.0) -> Tuple[Solution, AlgoResult]:
    rng = random.Random(seed)
    start = time.time()
    last_log = 0.0
    end_time = start + time_limit_s

    cur = init_sol
    (cur_ev, cur_obj,
     best_any_sol, best_any_ev, best_any_obj,
     best_feas_sol, best_feas_ev, best_feas_obj) = _init_best_tracking(inst, cur)

    tabu_tenure = 50
    tabu = {}
    iters = 0

    def signature(mv: Move) -> tuple:
        return (mv.kind, mv.a, mv.t, mv.direction, mv.b)

    if recorder is not None:
        ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
        ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
        recorder(0.0, ref_ev.max_avg_completion, ref_ev.sum_avg_completion, ref_obj)

    while time.time() < end_time:
        iters += 1
        candidates = []

        ev_cur = evaluate(inst, cur)
        cur_infeasible = ev_cur.violation > 1e-9

        for _ in range(60):
            if time.time() >= end_time:
                break

            if cur_infeasible and rng.random() < 0.8:
                mv = guided_move(inst, cur, rng, ev=ev_cur)
            else:
                mv = random_move(inst, cur, rng)

            nxt = apply_move(inst, cur, mv)
            if nxt is None:
                continue

            ev = evaluate(inst, nxt)
            obj = objective_with_penalty(ev)
            candidates.append((obj, mv, nxt, ev))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])

        chosen = None
        for obj, mv, nxt, ev in candidates:
            sig = signature(mv)
            is_tabu = sig in tabu and tabu[sig] > iters
            if (not is_tabu) or (obj < best_any_obj - 1e-12):
                chosen = (obj, mv, nxt, ev)
                break

        if chosen is None:
            chosen = candidates[0]

        obj, mv, nxt, ev = chosen
        cur, cur_ev, cur_obj = nxt, ev, obj
        tabu[signature(mv)] = iters + tabu_tenure

        (best_any_sol, best_any_ev, best_any_obj,
         best_feas_sol, best_feas_ev, best_feas_obj) = _update_best(
            inst, nxt, ev, obj,
            best_any_sol, best_any_ev, best_any_obj,
            best_feas_sol, best_feas_ev, best_feas_obj
        )

        if recorder is not None:
            ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
            ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
            last_log = _maybe_record(recorder, start, last_log, log_every_s, ref_ev, ref_obj)

    if best_feas_sol is not None:
        final_sol = best_feas_sol
        final_ev = best_feas_ev
        final_obj = best_feas_obj
    else:
        final_sol = best_any_sol
        final_ev = best_any_ev
        final_obj = best_any_obj

    res = AlgoResult(
        algo="tabu",
        init="custom",
        seed=seed,
        time_limit_s=time_limit_s,
        best_obj=float(final_obj),
        best_max_avg=float(final_ev.max_avg_completion),
        best_sum_avg=float(final_ev.sum_avg_completion),
        feasible=bool(final_ev.feasible),
        violation=float(final_ev.violation),
        n_iters=iters,
        elapsed_s=float(time.time() - start),
    )
    return final_sol, res

# -------------------- Genetic Algorithm (TIME-SAFE) --------------------

def _random_counts_for_agent(qj: int, T: int, rng: random.Random) -> np.ndarray:
    cuts = [0] * T
    for _ in range(qj):
        cuts[rng.randrange(T)] += 1
    return np.array(cuts, dtype=int)


def genetic_algorithm(inst: Instance, init_sol: Solution, time_limit_s: int, seed: int,
                      verbose: bool = False, recorder=None, log_every_s: float = 1.0) -> Tuple[Solution, AlgoResult]:
    rng = random.Random(seed)
    start = time.time()
    end_time = start + time_limit_s
    last_log = 0.0

    pop_size = 40
    elite = 6
    mutation_rate = 0.35

    # Helper: adapt repair budget near deadline
    def repair_budget() -> int:
        remaining = max(0.0, end_time - time.time())
        if remaining < 0.5:
            return 200
        if remaining < 2.0:
            return 800
        return 2000

    # --- init population (time-aware) ---
    pop: List[Solution] = [init_sol.copy()]
    while len(pop) < pop_size and time.time() < end_time:
        counts = []
        for j in range(inst.n_agents):
            counts.append(_random_counts_for_agent(len(inst.requirements[j]), inst.n_slots, rng))
        sol = Solution(counts)
        sol = repair_capacity_forward(inst, sol, rng=rng, max_passes=repair_budget())
        pop.append(sol)

    # if we couldn't fill pop (tiny time limit), duplicate best we have
    while len(pop) < pop_size:
        pop.append(pop[-1].copy())

    # fitness with caching per generation (avoid recomputing evaluate too much)
    def eval_key(sol: Solution) -> Tuple:
        return tuple(tuple(int(x) for x in c.tolist()) for c in sol.counts)

    def fitness_cached(sol: Solution, cache: Dict[Tuple, float]) -> float:
        k = eval_key(sol)
        if k in cache:
            return cache[k]
        f = objective_with_penalty(evaluate(inst, sol))
        cache[k] = f
        return f

    # initialize best tracking
    cache0: Dict[Tuple, float] = {}
    best = min(pop, key=lambda s: fitness_cached(s, cache0))
    (cur_ev, cur_obj,
     best_any_sol, best_any_ev, best_any_obj,
     best_feas_sol, best_feas_ev, best_feas_obj) = _init_best_tracking(inst, best)

    iters = 0
    if recorder is not None:
        ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
        ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
        recorder(0.0, ref_ev.max_avg_completion, ref_ev.sum_avg_completion, ref_obj)

    def tournament(pop: List[Solution], cache: Dict[Tuple, float], k: int = 3) -> Solution:
        cand = [pop[rng.randrange(len(pop))] for _ in range(k)]
        cand.sort(key=lambda s: fitness_cached(s, cache))
        return cand[0]

    # --- generations (time-aware) ---
    while time.time() < end_time:
        iters += 1
        gen_cache: Dict[Tuple, float] = {}

        scored = [(fitness_cached(s, gen_cache), s) for s in pop]
        scored.sort(key=lambda x: x[0])
        pop = [s for _, s in scored]

        # update best tracking from best of population
        best_pop = pop[0]
        ev = evaluate(inst, best_pop)
        obj = objective_with_penalty(ev)

        (best_any_sol, best_any_ev, best_any_obj,
         best_feas_sol, best_feas_ev, best_feas_obj) = _update_best(
            inst, best_pop, ev, obj,
            best_any_sol, best_any_ev, best_any_obj,
            best_feas_sol, best_feas_ev, best_feas_obj
        )

        # create next generation
        next_pop: List[Solution] = [pop[i].copy() for i in range(elite)]

        while len(next_pop) < pop_size and time.time() < end_time:
            p1 = tournament(pop, gen_cache)
            p2 = tournament(pop, gen_cache)

            child_counts = []
            for j in range(inst.n_agents):
                T = inst.n_slots
                cut = rng.randrange(1, T)
                a = p1.counts[j]
                b = p2.counts[j]
                c = np.concatenate([a[:cut], b[cut:]]).astype(int)

                qj = len(inst.requirements[j])
                diff = int(c.sum()) - qj
                while diff != 0:
                    t = rng.randrange(T)
                    if diff > 0 and c[t] > 0:
                        c[t] -= 1
                        diff -= 1
                    elif diff < 0:
                        c[t] += 1
                        diff += 1
                child_counts.append(c)

            child = Solution(child_counts)

            # mutation: random shifts
            if rng.random() < mutation_rate:
                for _ in range(rng.randint(1, 4)):
                    j = rng.randrange(inst.n_agents)
                    t = rng.randrange(inst.n_slots)
                    direction = rng.choice([-1, +1])
                    nt = t + direction
                    if 0 <= nt < inst.n_slots and child.counts[j][t] > 0:
                        child.counts[j][t] -= 1
                        child.counts[j][nt] += 1

            child = repair_capacity_forward(inst, child, rng=rng, max_passes=repair_budget())
            next_pop.append(child)

        pop = next_pop

        if recorder is not None:
            ref_ev = best_feas_ev if best_feas_ev is not None else best_any_ev
            ref_obj = best_feas_obj if best_feas_ev is not None else best_any_obj
            last_log = _maybe_record(recorder, start, last_log, log_every_s, ref_ev, ref_obj)

    if best_feas_sol is not None:
        final_sol = best_feas_sol
        final_ev = best_feas_ev
        final_obj = best_feas_obj
    else:
        final_sol = best_any_sol
        final_ev = best_any_ev
        final_obj = best_any_obj

    res = AlgoResult(
        algo="ga",
        init="custom",
        seed=seed,
        time_limit_s=time_limit_s,
        best_obj=float(final_obj),
        best_max_avg=float(final_ev.max_avg_completion),
        best_sum_avg=float(final_ev.sum_avg_completion),
        feasible=bool(final_ev.feasible),
        violation=float(final_ev.violation),
        n_iters=iters,
        elapsed_s=float(time.time() - start),
    )
    return final_sol, res


# -------------------- Dispatcher --------------------

def solve(inst: Instance, algo: str, init_method: str, seed: int, time_limit_s: int,
          verbose: bool = False, recorder=None, log_every_s: float = 1.0) -> Tuple[Solution, AlgoResult]:
    init_sol = build_initial(inst, init_method, seed)

    if algo == "ls":
        sol, res = local_search(inst, init_sol, time_limit_s, seed, verbose, recorder=recorder, log_every_s=log_every_s)
    elif algo == "sa":
        sol, res = simulated_annealing(inst, init_sol, time_limit_s, seed, verbose, recorder=recorder, log_every_s=log_every_s)
    elif algo == "tabu":
        sol, res = tabu_search(inst, init_sol, time_limit_s, seed, verbose, recorder=recorder, log_every_s=log_every_s)
    elif algo == "ga":
        sol, res = genetic_algorithm(inst, init_sol, time_limit_s, seed, verbose, recorder=recorder, log_every_s=log_every_s)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    res.init = init_method
    return sol, res