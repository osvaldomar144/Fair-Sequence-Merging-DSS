from __future__ import annotations
from pathlib import Path
from typing import List
import glob
import numpy as np

from src.fairseq.types import Instance


def _split_semicolon_line(line: str) -> List[str]:
    return [p for p in line.strip().split(";") if p != ""]


def load_instance(instance_dir: str | Path) -> Instance:
    d = Path(instance_dir)

    size_files = list(d.glob("*_size.csv"))
    cap_files = list(d.glob("*_capacities.csv"))
    req_files = list(d.glob("*_requirements.csv"))
    if not size_files or not cap_files or not req_files:
        raise FileNotFoundError(f"Missing csv files in {d}")

    size_path = size_files[0]
    cap_path = cap_files[0]
    req_path = req_files[0]

    # --- size ---
    size_parts = _split_semicolon_line(size_path.read_text())
    if len(size_parts) < 2:
        raise ValueError(f"Invalid size file: {size_path}")
    n_agents = int(float(size_parts[0]))
    n_slots = int(float(size_parts[1]))

    # --- capacities ---
    cap_parts = _split_semicolon_line(cap_path.read_text())
    capacities = np.array([float(x) for x in cap_parts], dtype=float)
    if len(capacities) != n_slots:
        raise ValueError(
            f"Capacities length mismatch in {cap_path}: got {len(capacities)}, expected {n_slots}"
        )

    # --- requirements ---
    lines = req_path.read_text().splitlines()
    if len(lines) < n_agents:
        raise ValueError(f"Requirements file has {len(lines)} rows, expected {n_agents}: {req_path}")

    requirements: List[np.ndarray] = []
    for i in range(n_agents):
        parts = _split_semicolon_line(lines[i])
        requirements.append(np.array([float(x) for x in parts], dtype=float))

    name = d.name
    return Instance(
        name=name,
        n_agents=n_agents,
        n_slots=n_slots,
        capacities=capacities,
        requirements=requirements,
    )