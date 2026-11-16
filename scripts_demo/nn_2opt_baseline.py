"""Classical TSP heuristic baseline: Nearest-Neighbor construction + 2-opt refinement.

Runs on the SAME normalized instances DIFUSCO is tested on (reads the coords from a
DIFUSCO/Concorde txt line), so the tour length is directly comparable to DIFUSCO's
`solved_cost` and Concorde's optimal `gt_cost`.

Usage:
    python -I scripts_demo/nn_2opt_baseline.py --inp data/tsplib_large/st70_concorde.txt
"""
import argparse
import time

import numpy as np


def load_coords(path):
    """Read the coords (everything before the `output` token) from a concorde txt line."""
    with open(path) as f:
        line = f.readline().strip()
    head = line.split(" output ")[0].split()
    vals = np.array(head, dtype=np.float64)
    return vals.reshape(-1, 2)


def tour_length(tour, dist):
    return float(dist[tour, np.roll(tour, -1)].sum())


def nearest_neighbor(dist):
    n = len(dist)
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].copy()
        d[visited] = np.inf
        nxt = int(d.argmin())
        tour.append(nxt)
        visited[nxt] = True
    return np.array(tour)


def two_opt(tour, dist, max_passes=1000):
    """Full 2-opt to a local optimum (best-improvement per edge pair, first-improve sweep)."""
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                if d == a:
                    continue
                if (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d]) < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    b = tour[i + 1]
    return tour, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="path to *_concorde.txt instance")
    args = ap.parse_args()

    coords = load_coords(args.inp)
    n = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    t0 = time.time()
    nn_tour = nearest_neighbor(dist)
    nn_cost = tour_length(nn_tour, dist)
    opt_tour, passes = two_opt(nn_tour.copy(), dist)
    final_cost = tour_length(opt_tour, dist)
    elapsed = time.time() - t0

    print(f"instance={args.inp}")
    print(f"N={n}")
    print(f"NN_cost={nn_cost:.4f}")
    print(f"NN+2opt_cost={final_cost:.4f}")
    print(f"2opt_passes={passes}")
    print(f"runtime_s={elapsed:.3f}")


if __name__ == "__main__":
    main()
