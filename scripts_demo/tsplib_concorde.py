"""Convert a TSPLIB .tsp instance into DIFUSCO's Concorde txt line, with a
PROVABLY OPTIMAL label tour solved by Concorde (pyconcorde).

Steps:
  1. Parse coords with tsplib95.
  2. Min-max normalize into the unit square [0,1]^2 (DIFUSCO's training scale),
     using a single uniform scale so aspect ratio is preserved.
  3. Solve the normalized instance to optimality with Concorde (EUC_2D).
  4. Emit one line:  x1 y1 ... xN yN output n1 n2 ... nN n1   (1-indexed, first repeated)
"""
import argparse
import os

import numpy as np
import tsplib95
from concorde.tsp import TSPSolver


def normalize(coords):
    lo = coords.min(axis=0)
    span = (coords.max(axis=0) - lo).max()  # uniform scale -> keeps aspect ratio
    return (coords - lo) / span


def solve_one(coords):
    scale = 1_000_000
    xs = coords[:, 0] * scale
    ys = coords[:, 1] * scale
    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    sol = solver.solve(verbose=False)
    return np.array(sol.tour, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsp", required=True, help="path to .tsp file")
    ap.add_argument("--out", required=True, help="output concorde .txt path")
    args = ap.parse_args()

    prob = tsplib95.load(args.tsp)
    nodes = sorted(prob.node_coords.keys())
    coords = np.array([prob.node_coords[i] for i in nodes], dtype=np.float64)
    coords = normalize(coords)
    n = len(coords)

    tour = solve_one(coords)

    coord_str = " ".join(f"{x} {y}" for x, y in coords)
    tour_str = " ".join(str(t + 1) for t in tour) + f" {tour[0] + 1}"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"{coord_str} output {tour_str}\n")

    pts = coords[tour]
    tlen = np.sqrt(((pts - np.roll(pts, -1, axis=0)) ** 2).sum(1)).sum()
    print(f"Wrote {n}-city instance to {args.out}")
    print(f"Label = Concorde optimal; normalized optimal length = {tlen:.4f}")


if __name__ == "__main__":
    main()
