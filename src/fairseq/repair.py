from __future__ import annotations
import random
from typing import Optional, Tuple, List
import numpy as np

from src.fairseq.types import Instance, Solution
from src.fairseq.evaluate import evaluate


def _boundary_task_size(inst: Instance, sol: Solution, j: int, t: int, which: str) -> Optional[Tuple[int, float]]:
    """
    Return (task_index, size) of a boundary task of agent j in slot t.
    which='last' -> last task in slot t
    which='first' -> first task in slot t
    Task index is within the agent chain.
    """
    counts = sol.counts[j]
    if counts[t] <= 0:
        return None
    start = int(counts[:t].sum())
    k = int(counts[t])
    if which == "last":
        idx = start + k - 1
    else:
        idx = start
    return idx, float(inst.requirements[j][idx])


def _max_req_leq_max_cap(inst: Instance) -> bool:
    max_cap = float(np.max(inst.capacities)) if inst.n_slots > 0 else 0.0
    for j in range(inst.n_agents):
        if len(inst.requirements[j]) == 0:
            continue
        if float(np.max(inst.requirements[j])) > max_cap + 1e-9:
            return False
    return True


def repair_capacity(
    inst: Instance,
    sol: Solution,
    rng: random.Random,
    max_passes: int = 10000,
    eps: float = 1e-9,
) -> Solution:
    """
    Repair capacity violations by moving ONE boundary task from an overloaded slot
    to a neighboring slot (t->t+1 using 'last', or t->t-1 using 'first').

    This preserves precedence/contiguity under the counts representation.

    Strategy:
    - pick most overloaded slot t
    - try move to neighbor with slack (prefer the one with more slack)
    - if no neighbor has slack, still move towards side with larger (less negative) slack,
      picking the smallest boundary task, to help future repairs.
    """
    if not _max_req_leq_max_cap(inst):
        # Infeasible instance: some task bigger than any slot capacity
        return sol.copy()

    new = sol.copy()
    T = inst.n_slots

    for _ in range(max_passes):
        ev = evaluate(inst, new)
        if ev.violation <= eps:
            return new

        over = ev.slot_loads - inst.capacities
        t = int(np.argmax(over))
        if over[t] <= eps:
            return new

        # neighbor slack values
        slack_left = None
        slack_right = None
        if t > 0:
            slack_left = float(inst.capacities[t - 1] - ev.slot_loads[t - 1])
        if t < T - 1:
            slack_right = float(inst.capacities[t + 1] - ev.slot_loads[t + 1])

        # choose direction
        # prefer a neighbor with positive slack; if both, take the larger slack
        direction = None
        if slack_left is not None and slack_left > eps and slack_right is not None and slack_right > eps:
            direction = -1 if slack_left >= slack_right else +1
        elif slack_right is not None and slack_right > eps:
            direction = +1
        elif slack_left is not None and slack_left > eps:
            direction = -1
        else:
            # no slack: pick direction that is "less bad"
            if t == 0:
                direction = +1
            elif t == T - 1:
                direction = -1
            else:
                # choose the side with larger slack (even if negative)
                # i.e., less overloaded neighbor
                if slack_left is None:
                    direction = +1
                elif slack_right is None:
                    direction = -1
                else:
                    direction = -1 if slack_left >= slack_right else +1

        nt = t + direction
        if nt < 0 or nt >= T:
            return new

        # Determine which boundary task is moved
        which = "last" if direction == +1 else "first"

        # Try to find an agent boundary task that fits the target slack (if slack is positive)
        target_slack = float(inst.capacities[nt] - ev.slot_loads[nt])

        candidates: List[Tuple[float, int]] = []  # (size, agent)
        for j in range(inst.n_agents):
            if new.counts[j][t] <= 0:
                continue
            bt = _boundary_task_size(inst, new, j, t, which=which)
            if bt is None:
                continue
            _, size = bt
            candidates.append((size, j))

        if not candidates:
            return new

        # Prefer a task that fits if we have slack; otherwise take smallest
        chosen_agent = None
        if target_slack > eps:
            fit = [(size, j) for (size, j) in candidates if size <= target_slack + eps]
            if fit:
                # choose the largest that fits (reduces overload faster)
                fit.sort(key=lambda x: x[0], reverse=True)
                chosen_agent = fit[0][1]

        if chosen_agent is None:
            # fallback: pick the smallest boundary task (gentlest move)
            candidates.sort(key=lambda x: x[0])
            chosen_agent = candidates[0][1]

        # Apply the move (counts update)
        new.counts[chosen_agent][t] -= 1
        new.counts[chosen_agent][nt] += 1

    return new


def repair_capacity_forward(inst: Instance, sol: Solution, rng: random.Random, max_passes: int = 10000) -> Solution:
    return repair_capacity(inst, sol, rng=rng, max_passes=max_passes)