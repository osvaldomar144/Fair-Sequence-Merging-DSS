from __future__ import annotations
from typing import List
import numpy as np
from src.fairseq.types import Instance, Solution, Eval

def _agent_slot_sums(req: np.ndarray, counts: np.ndarray) -> np.ndarray:
    # returns sums per slot for one agent, using contiguous slicing by counts
    T = len(counts)
    out = np.zeros(T, dtype=float)
    idx = 0
    for t in range(T):
        k = int(counts[t])
        if k > 0:
            out[t] = float(req[idx:idx+k].sum())
        idx += k
    return out

def evaluate(inst: Instance, sol: Solution) -> Eval:
    n, T = inst.n_agents, inst.n_slots

    # sanity
    if len(sol.counts) != n:
        raise ValueError("counts length mismatch")
    for j in range(n):
        if len(sol.counts[j]) != T:
            raise ValueError("counts slot length mismatch")
        if int(sol.counts[j].sum()) != len(inst.requirements[j]):
            raise ValueError(f"agent {j} tasks mismatch: sum(counts)={sol.counts[j].sum()} q={len(inst.requirements[j])}")

    slot_loads = np.zeros(T, dtype=float)
    avg_by_agent = np.zeros(n, dtype=float)

    for j in range(n):
        counts_j = sol.counts[j]
        req_j = inst.requirements[j]
        sums_j = _agent_slot_sums(req_j, counts_j)
        slot_loads += sums_j

        # average completion slot index (1..T) replicated for tasks in that slot
        qj = len(req_j)
        if qj == 0:
            avg_by_agent[j] = 0.0
        else:
            total = 0.0
            for t in range(T):
                total += (t+1) * float(counts_j[t])
            avg_by_agent[j] = total / qj

    violation_vec = np.maximum(0.0, slot_loads - inst.capacities)
    violation = float(violation_vec.sum())
    feasible = violation <= 1e-9

    max_avg = float(avg_by_agent.max()) if n > 0 else 0.0
    sum_avg = float(avg_by_agent.sum())

    return Eval(
        feasible=feasible,
        max_avg_completion=max_avg,
        sum_avg_completion=sum_avg,
        avg_by_agent=avg_by_agent,
        slot_loads=slot_loads,
        violation=violation,
    )

def objective_with_penalty(ev: Eval, penalty_weight: float = 1e6) -> float:
    # for metaheuristics/GA: infeasible solutions get a huge penalty
    return ev.max_avg_completion + penalty_weight * ev.violation
