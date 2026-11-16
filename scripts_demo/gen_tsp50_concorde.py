"""Generate synthetic TSP-50 data with PROVABLY OPTIMAL tours from Concorde (pyconcorde).

Same output as gen_tsp50_synthetic.py (DIFUSCO/Concorde txt line format), but the label
tour is solved to optimality by Concorde instead of nearest-neighbor + 2-opt.

Line format:
    x1 y1 ... xN yN output t1 t2 ... tN t1   (1-indexed node ids, first repeated)
"""
import argparse
import os
import sys

import numpy as np
from concorde.tsp import TSPSolver


def solve_one(coords):
    # Concorde (EUC_2D) uses integer coords internally, so scale up the [0,1] points.
    scale = 1_000_000
    xs = (coords[:, 0] * scale)
    ys = (coords[:, 1] * scale)
    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    sol = solver.solve(verbose=False)
    return np.array(sol.tour, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_nodes", type=int, default=50)
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--filename", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    all_coords = rng.random((args.num_samples, args.n_nodes, 2))

    os.makedirs(os.path.dirname(os.path.abspath(args.filename)), exist_ok=True)
    with open(args.filename, "w") as f:
        for i in range(args.num_samples):
            coords = all_coords[i]
            tour = solve_one(coords)
            coord_str = " ".join(f"{x} {y}" for x, y in coords)
            tour_str = " ".join(str(t + 1) for t in tour) + f" {tour[0] + 1}"
            f.write(f"{coord_str} output {tour_str}\n")
            if (i + 1) % 100 == 0:
                print(f"  solved {i + 1}/{args.num_samples}", flush=True)
    print(f"Wrote {args.num_samples} OPTIMAL TSP-{args.n_nodes} instances to {args.filename}")


if __name__ == "__main__":
    main()
