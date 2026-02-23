from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import random

from src.fairseq.types import Instance, Solution
from src.fairseq.evaluate import evaluate, Eval


@dataclass(frozen=True)
class Move:
    kind: str  # 'shift' or 'swap'
    a: int
    t: int
    direction: int = 0
    b: int = -1


def _boundary_task_size(
    inst: Instance,
    sol: Solution,
    j: int,
    t: int,
    which: str = "last",
) -> Optional[Tuple[int, float]]:
    """
    Return (task_index, size) of a boundary task of agent j in slot t.
    which='last' -> last task of slot t (useful for shifting forward)
    which='first' -> first task of slot t (useful for shifting backward)
    """
    counts = sol.counts[j]
    if counts[t] <= 0:
        return None
    start = int(counts[:t].sum())
    k = int(counts[t])
    idx = (start + k - 1) if which == "last" else start
    return idx, float(inst.requirements[j][idx])


def apply_shift(inst: Instance, sol: Solution, j: int, t: int, direction: int) -> Optional[Solution]:
    T = inst.n_slots
    if direction not in (-1, +1):
        return None
    if t < 0 or t >= T:
        return None
    nt = t + direction
    if nt < 0 or nt >= T:
        return None
    if sol.counts[j][t] <= 0:
        return None

    new = sol.copy()
    new.counts[j][t] -= 1
    new.counts[j][nt] += 1
    return new


def apply_swap(inst: Instance, sol: Solution, a: int, b: int, t: int) -> Optional[Solution]:
    """
    "Swap-like" move that preserves contiguity/precedence by exchanging one boundary
    task across agents between adjacent slots t and t+1:
      - agent a: move one task from slot t -> t+1
      - agent b: move one task from slot t+1 -> t
    """
    if a == b:
        return None
    T = inst.n_slots
    if t < 0 or t >= T - 1:
        return None
    if sol.counts[a][t] <= 0 or sol.counts[b][t + 1] <= 0:
        return None

    new = sol.copy()
    new.counts[a][t] -= 1
    new.counts[a][t + 1] += 1
    new.counts[b][t + 1] -= 1
    new.counts[b][t] += 1
    return new


def random_move(inst: Instance, sol: Solution, rng: random.Random) -> Move:
    n, T = inst.n_agents, inst.n_slots
    if rng.random() < 0.7:
        j = rng.randrange(n)
        t = rng.randrange(T)
        direction = rng.choice([-1, +1])
        return Move(kind="shift", a=j, t=t, direction=direction)
    else:
        a = rng.randrange(n)
        b = rng.randrange(n)
        while b == a:
            b = rng.randrange(n)
        t = rng.randrange(max(1, T - 1))
        return Move(kind="swap", a=a, b=b, t=t)


def guided_move(inst: Instance, sol: Solution, rng: random.Random, ev: Optional[Eval] = None) -> Move:
    """
    Guided feasibility move (still only SHIFT):
    - If infeasible, pick most overloaded slot t and shift a boundary task
      toward a neighbor slot (t-1 or t+1) with more slack.
    - Prefer tasks that FIT into the target slack, so the move can actually
      reduce the global violation rather than just moving overload around.

    Optimization: you can pass ev=evaluate(inst, sol) to avoid recomputing it.
    """
    if ev is None:
        ev = evaluate(inst, sol)

    n, T = inst.n_agents, inst.n_slots
    if ev.violation <= 1e-9 or T <= 1:
        return random_move(inst, sol, rng)

    overload = ev.slot_loads - inst.capacities
    t = int(np.argmax(overload))

    # choose direction toward more slack (neighbor with larger slack)
    if t == 0:
        direction = +1
    elif t == T - 1:
        direction = -1
    else:
        left_slack = float(inst.capacities[t - 1] - ev.slot_loads[t - 1])
        right_slack = float(inst.capacities[t + 1] - ev.slot_loads[t + 1])
        direction = -1 if left_slack >= right_slack else +1

    nt = t + direction
    if nt < 0 or nt >= T:
        return random_move(inst, sol, rng)

    target_slack = float(inst.capacities[nt] - ev.slot_loads[nt])
    which = "last" if direction == +1 else "first"

    agents = [j for j in range(n) if sol.counts[j][t] > 0]
    if not agents:
        return random_move(inst, sol, rng)

    candidates_fit: list[tuple[float, int]] = []
    candidates_all: list[tuple[float, int]] = []

    for j in agents:
        bt = _boundary_task_size(inst, sol, j, t, which=which)
        if bt is None:
            continue
        _, size = bt
        candidates_all.append((size, j))
        if size <= target_slack + 1e-9:
            candidates_fit.append((size, j))

    if candidates_fit:
        # choose the largest that fits (reduces overload faster)
        candidates_fit.sort(key=lambda x: x[0], reverse=True)
        j = candidates_fit[0][1]
        return Move(kind="shift", a=j, t=t, direction=direction)

    if candidates_all:
        # fallback: smallest boundary task (least harm)
        candidates_all.sort(key=lambda x: x[0])
        j = candidates_all[0][1]
        return Move(kind="shift", a=j, t=t, direction=direction)

    return random_move(inst, sol, rng)


def apply_move(inst: Instance, sol: Solution, mv: Move) -> Optional[Solution]:
    if mv.kind == "shift":
        return apply_shift(inst, sol, mv.a, mv.t, mv.direction)
    if mv.kind == "swap":
        return apply_swap(inst, sol, mv.a, mv.b, mv.t)
    return None