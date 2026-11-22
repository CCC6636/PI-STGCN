"""Generate synthetic demo inputs for PI-STGCN (Sample/ directory).

The released model is trained on MIKE 21 simulation output, which cannot be redistributed.
This script builds *synthetic* files that reproduce the exact input structure expected by
`main.py`, so the pipeline can be smoke-tested end to end:

    Sample/data/condition_*.csv   one file per operating condition, rows = output time steps
    Sample/section_distance.csv   N x N signed influence matrix (source node -> target node)

Columns 0-3 are metadata, columns 4-6 are the three boundary flows (Q1, Q2, Q3) and columns
7-62 are the 56 node water levels, which matches config['q_indices'] = '4,5,6' and
config['node_indices'] = 'range:7-62' in main.py.

The hydrodynamics here are a smooth relaxation toy model, not a flow solution: metrics
obtained from these files have no physical meaning and must never be reported as results.

Usage:  python generate_sample_data.py [--conditions 12] [--steps 180]
Requires: numpy only.
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np

NODES = 56
INJECTION_IDX = (0, 9, 55)          # Q1 upstream node 1, Q2 units node 10, Q3 downstream node 56
DT_SECONDS = 60.0                   # output interval of each demo condition
MATRIX_SCALE = 15.0                 # influence values stay within +/-15, as assumed by main.py

# Water levels are normalised with WATER_GLOBAL_MIN = 2581 / WATER_GLOBAL_RANGE = 32 in
# data_utils.py, so the synthetic profiles are kept close to the reported operating band.
LEVEL_BASE = 2600.0
LEVEL_SILL = {20: 0.30, 21: 0.34, 22: 0.36, 23: 0.33, 24: 0.28, 25: 0.20,
              26: 0.12, 27: 0.08, 28: 0.12}


def longitudinal_profile(rng: np.random.Generator) -> np.ndarray:
    """Mean water surface along the reach: mild slope plus bed sills and the constriction."""
    i = np.arange(NODES)
    slope = LEVEL_BASE + 0.90 * (1.0 - i / (NODES - 1))
    profile = slope + rng.normal(0.0, 0.02, NODES)
    for idx, bump in LEVEL_SILL.items():
        profile[idx:] += bump
    return profile


def steady_response(profile: np.ndarray, q1: float, q2: float, q3: float) -> np.ndarray:
    """Steady-state water-level change caused by the three boundary flows."""
    i = np.arange(NODES)

    # Pumping (q2 < 0) carves a drawdown funnel around the inlet/outlet structures,
    # generation (q2 > 0) piles water up and sends a forward wave downstream.
    funnel = np.exp(-np.abs(i - INJECTION_IDX[1]) / 6.0)
    units = -q2 * 0.00080 * funnel
    units += np.where(i > INJECTION_IDX[1], -q2 * 0.00018 * np.exp(-(i - INJECTION_IDX[1]) / 14.0), 0.0)

    inflow = q1 * 0.00120 * np.exp(-i / 18.0)
    release = -q3 * 0.00100 * np.exp(-(NODES - 1 - i) / 16.0)

    return profile + inflow + units + release


def relaxation_time_scale(distance: np.ndarray) -> np.ndarray:
    """Far nodes react later, which gives smooth increments like the reference simulations."""
    return 900.0 + 260.0 * distance


def write_condition(path: pathlib.Path, name: str, case: str, steps: int,
                    profile: np.ndarray, q1: float, q2: float, q3: float,
                    rng: np.random.Generator) -> None:
    t = np.arange(steps) * DT_SECONDS
    dist_sources = np.min(np.abs(np.arange(NODES)[:, None] - np.array(INJECTION_IDX)[None, :]), axis=1).astype(float)
    tau = relaxation_time_scale(dist_sources)

    # Boundary flows ramp in and wander slowly, as they do in the scheduled-operation output.
    phase = rng.uniform(0.0, 2.0 * math.pi, 3)
    ramp = 1.0 - np.exp(-t / 1200.0)
    q1_t = q1 * (1.0 + 0.08 * np.sin(2.0 * math.pi * t / 7200.0 + phase[0]))
    q2_t = q2 * ramp
    q3_t = q3 * (1.0 + 0.08 * np.sin(2.0 * math.pi * t / 7200.0 + phase[2]))

    header = ["Time_s", "Condition_ID", "H0_m", "Case_Type", "Q1_m3s", "Q2_m3s", "Q3_m3s"]
    header += [f"Node_{j + 1:02d}" for j in range(NODES)]
    lines = [",".join(header)]
    for k in range(steps):
        # Each node relaxes towards the steady state of the instantaneous forcing, with a lag.
        target = steady_response(profile, q1_t[k], q2_t[k], q3_t[k])
        levels = profile + (target - profile) * (1.0 - np.exp(-t[k] / tau))
        levels += 0.020 * np.sin(2.0 * math.pi * t[k] / 2400.0 - dist_sources / 6.0) * math.exp(-t[k] / 5400.0)
        levels += rng.normal(0.0, 0.002, NODES)
        meta = [f"{t[k]:.0f}", name, f"{profile[0]:.2f}", case,
                f"{round(q1_t[k], 2) + 0.0:.2f}", f"{round(q2_t[k], 2) + 0.0:.2f}", f"{round(q3_t[k], 2) + 0.0:.2f}"]
        lines.append(",".join(meta + [f"{v:.4f}" for v in levels]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_influence_matrix(rng: np.random.Generator) -> np.ndarray:
    """Signed source-to-target influence: positive downstream, negative upstream, decaying."""
    raw = np.zeros((NODES, NODES))
    for src in range(NODES):
        for tgt in range(NODES):
            if src == tgt:
                continue
            gap = tgt - src
            decay = math.exp(-abs(gap) / 12.0)
            terrain = 1.0 + 0.12 * math.sin(abs(gap) / 3.0 + (src + tgt) / 9.0)
            # The constriction around nodes 26-28 restricts reverse-flow propagation.
            blocked = 0.35 if (gap < 0 and 25 <= tgt <= 28 and src > 28) else 1.0
            raw[src, tgt] = math.copysign(1.0, gap) * MATRIX_SCALE * decay * terrain * blocked
    raw += rng.normal(0.0, 0.05, (NODES, NODES)) * (1.0 - np.eye(NODES))
    # main.py assumes |raw| <= 15 because it rescales the matrix into [0, 1].
    raw = np.clip(raw, -MATRIX_SCALE, MATRIX_SCALE)
    np.fill_diagonal(raw, 0.0)
    return raw


def write_matrix(path: pathlib.Path, raw: np.ndarray) -> None:
    labels = [f"CS_{j + 1:02d} (Node_{j + 1:02d})" for j in range(NODES)]
    lines = [",".join(["Section"] + labels)]
    for i, label in enumerate(labels):
        lines.append(",".join([label] + [f"{v:.4f}" for v in raw[i]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--conditions', type=int, default=12, help='number of demo conditions (default 12)')
    parser.add_argument('--steps', type=int, default=180, help='time steps per condition (default 180)')
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--out', default=str(pathlib.Path(__file__).resolve().parent))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = pathlib.Path(args.out)
    data_dir = out / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    write_matrix(out / 'section_distance.csv', build_influence_matrix(rng))

    # Idle / pumping / generation conditions, roughly one third each.
    q2_schedule = []
    modes = ['idle', 'pump', 'gen']
    for n in range(args.conditions):
        mode = modes[n % 3]
        if mode == 'idle':
            q2 = 0.0
        elif mode == 'pump':
            q2 = -75.0 - 550.0 * (0.25 + 0.75 * rng.random())
        else:
            q2 = 75.0 + 550.0 * (0.25 + 0.75 * rng.random())
        q2_schedule.append((mode, q2))

    for n, (mode, q2) in enumerate(q2_schedule, start=1):
        profile = longitudinal_profile(rng) + rng.uniform(-2.0, 2.0)   # different initial pool level
        h0 = profile[0]
        q1 = 120.0 + 260.0 * rng.random()
        q3 = 100.0 + 240.0 * rng.random()
        name = f"condition_{mode}_{n:02d}"
        write_condition(data_dir / f"{name}.csv", name, mode, args.steps, profile, q1, q2, q3, rng)
        print(f"wrote {name}.csv  H0={h0:.2f} m  Q1={q1:.1f}  Q2={q2:.1f}  Q3={q3:.1f} m3/s")

    print(f"\n{args.conditions} conditions x {args.steps} steps in {data_dir}")
    print(f"influence matrix: {out / 'section_distance.csv'}")


if __name__ == '__main__':
    main()
