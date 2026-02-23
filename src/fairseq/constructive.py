from __future__ import annotations
from typing import List
import numpy as np
import random

from src.fairseq.types import Instance, Solution
from src.fairseq.evaluate import evaluate
from src.fairseq.repair import repair_capacity_forward


def _empty_counts(inst: Instance) -> List[np.ndarray]:
    return [np.zeros(inst.n_slots, dtype=int) for _ in range(inst.n_agents)]


def _finalize_initial(inst: Instance, sol: Solution, rng: random.Random, max_passes: int = 2000) -> Solution:
    """
    Try to repair capacity violations; if still infeasible, return anyway.
    Metaheuristics will drive feasibility via penalties.
    """
    sol = repair_capacity_forward(inst, sol, rng=rng, max_passes=max_passes)
    return sol

def round_robin(inst: Instance, seed: int = 1) -> Solution:
    rng = random.Random(seed)
    n, T = inst.n_agents, inst.n_slots
    counts = _empty_counts(inst)
    remaining = [len(inst.requirements[j]) for j in range(n)]
    ptr = [0] * n
    slot_remaining = inst.capacities.copy()

    # iterate slots from early to late; within each slot, cycle agents
    for t in range(T):
        made_progress = True
        while made_progress:
            made_progress = False
            for j in range(n):
                if remaining[j] <= 0:
                    continue
                task_size = float(inst.requirements[j][ptr[j]])
                if task_size <= slot_remaining[t] + 1e-9:
                    counts[j][t] += 1
                    ptr[j] += 1
                    remaining[j] -= 1
                    slot_remaining[t] -= task_size
                    made_progress = True

    # if any tasks remain, try to place them (may temporarily violate -> repaired later)
    for j in range(n):
        while remaining[j] > 0:
            size = float(inst.requirements[j][ptr[j]])
            placed = False
            for t in range(T):
                if size <= slot_remaining[t] + 1e-9:
                    counts[j][t] += 1
                    ptr[j] += 1
                    remaining[j] -= 1
                    slot_remaining[t] -= size
                    placed = True
                    break
            if not placed:
                # do NOT "accept" violation permanently: place temporarily and repair later
                counts[j][T - 1] += 1
                ptr[j] += 1
                remaining[j] -= 1

    sol = Solution([c.astype(int) for c in counts])
    return _finalize_initial(inst, sol, rng)


def random_construct(inst: Instance, seed: int = 1) -> Solution:
    rng = random.Random(seed)
    n, T = inst.n_agents, inst.n_slots
    counts = _empty_counts(inst)
    ptr = [0] * n
    remaining = [len(inst.requirements[j]) for j in range(n)]
    slot_remaining = inst.capacities.copy()

    # build a randomized sequence of "next task" choices among agents
    pool = []
    for j in range(n):
        pool.extend([j] * remaining[j])
    rng.shuffle(pool)

    # assign each next task to an early feasible slot when possible
    for j in pool:
        size = float(inst.requirements[j][ptr[j]])
        feasible_slots = [t for t in range(T) if size <= slot_remaining[t] + 1e-9]
        if feasible_slots:
            feasible_slots.sort()
            k = min(3, len(feasible_slots))
            t = rng.choice(feasible_slots[:k])
            slot_remaining[t] -= size
        else:
            # temporary placement; will be repaired
            t = T - 1

        counts[j][t] += 1
        ptr[j] += 1
        remaining[j] -= 1

    sol = Solution([c.astype(int) for c in counts])
    return _finalize_initial(inst, sol, rng)


def greedy_fair(inst: Instance, seed: int = 1) -> Solution:
    """
    Heuristic: assign next task to the agent currently "worst" (highest avg completion so far),
    placing as early as possible. If no feasible slot exists, place temporarily and repair.
    """
    rng = random.Random(seed)
    n, T = inst.n_agents, inst.n_slots
    counts = _empty_counts(inst)
    ptr = [0] * n
    remaining = [len(inst.requirements[j]) for j in range(n)]
    slot_remaining = inst.capacities.copy()

    assigned_counts = [0] * n
    assigned_sum_slots = [0.0] * n

    def current_avg(j: int) -> float:
        if assigned_counts[j] == 0:
            return 0.0
        return assigned_sum_slots[j] / assigned_counts[j]

    total_tasks = sum(remaining)
    for _ in range(total_tasks):
        cand = [j for j in range(n) if remaining[j] > 0]
        cand.sort(key=lambda j: (current_avg(j), rng.random()), reverse=True)
        j = cand[0]

        size = float(inst.requirements[j][ptr[j]])

        chosen = None
        for t in range(T):
            if size <= slot_remaining[t] + 1e-9:
                chosen = t
                break

        if chosen is None:
            chosen = T - 1  # temporary, repaired later
        else:
            slot_remaining[chosen] -= size

        counts[j][chosen] += 1
        ptr[j] += 1
        remaining[j] -= 1

        assigned_counts[j] += 1
        assigned_sum_slots[j] += (chosen + 1)

    sol = Solution([c.astype(int) for c in counts])
    return _finalize_initial(inst, sol, rng)


def build_initial(inst: Instance, method: str, seed: int) -> Solution:
    if method == "round_robin":
        return round_robin(inst, seed)
    if method == "random":
        return random_construct(inst, seed)
    if method == "greedy_fair":
        return greedy_fair(inst, seed)
    raise ValueError(f"Unknown init method: {method}")