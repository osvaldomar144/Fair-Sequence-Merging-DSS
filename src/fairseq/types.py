from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

@dataclass(frozen=True)
class Instance:
    name: str
    n_agents: int
    n_slots: int
    capacities: np.ndarray            # shape (T,)
    requirements: List[np.ndarray]    # len n_agents; each array shape (q_j,)

    @property
    def n_tasks(self) -> int:
        return int(sum(len(r) for r in self.requirements))

@dataclass
class Solution:
    # For each agent j: counts[j][t] = number of tasks of agent j assigned to slot t (contiguous per slot).
    counts: List[np.ndarray]  # list length n_agents; each shape (T,), sum counts[j] = q_j

    def copy(self) -> "Solution":
        return Solution([c.copy() for c in self.counts])

@dataclass(frozen=True)
class Eval:
    feasible: bool
    max_avg_completion: float
    sum_avg_completion: float
    avg_by_agent: np.ndarray   # shape (n_agents,)
    slot_loads: np.ndarray     # shape (T,)
    violation: float           # total capacity violation (0 if feasible)
