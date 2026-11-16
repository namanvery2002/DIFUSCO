"""Convert a TSPLIB instance (+ optional .opt.tour) into DIFUSCO's Concorde txt line.

Steps:
  1. Parse coords with tsplib95.
  2. Min-max normalize coords into the unit square [0,1]^2 (DIFUSCO's training scale).
  3. Use the TSPLIB optimal tour as the label if given, else nearest-neighbor + 2-opt.
  4. Emit one line:  x1 y1 ... xN yN output n1 n2 ... nN n1   (1-indexed, first node repeated)
"""
import argparse

import numpy as np
import tsplib95


def normalize(coords):
    lo = coords.min(axis=0)
    span = (coords.max(axis=0) - lo).max()  # uniform scale -> keeps aspect ratio
    return (coords - lo) / span


def nearest_neighbor(coords):
    n = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    visited = np.zeros(n, dtype=bool)
    tour = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].copy(); d[visited] = np.inf
        nxt = int(d.argmin()); tour.append(nxt); visited[nxt] = True
    return np.array(tour), dist


def two_opt(tour, dist, max_passes=100):
    n = len(tour); improved = True; p = 0
    while improved and p < max_passes:
        improved = False; p += 1
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                c = tour[j]; d = tour[(j + 1) % n]
                if d == a:
                    continue
                if (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d]) < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]; improved = True
    return tour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsp", required=True, help="path to .tsp file")
    ap.add_argument("--opt_tour", default=None, help="path to .opt.tour file (optional)")
    ap.add_argument("--out", required=True, help="output concorde .txt path")
    args = ap.parse_args()

    prob = tsplib95.load(args.tsp)
    nodes = sorted(prob.node_coords.keys())           # usually 1..N
    coords = np.array([prob.node_coords[i] for i in nodes], dtype=np.float64)
    coords = normalize(coords)
    n = len(coords)

    if args.opt_tour:
        opt = tsplib95.load(args.opt_tour)
        tour0 = list(opt.tours[0])                     # 1-indexed node ids
        node_pos = {nid: k for k, nid in enumerate(nodes)}
        tour = np.array([node_pos[t] for t in tour0])  # -> 0-indexed positions
        label = "TSPLIB optimal"
    else:
        tour, dist = nearest_neighbor(coords)
        tour = two_opt(tour, dist)
        label = "NN + 2-opt"

    coord_str = " ".join(f"{x} {y}" for x, y in coords)
    tour_str = " ".join(str(t + 1) for t in tour) + f" {tour[0] + 1}"
    with open(args.out, "w") as f:
        f.write(f"{coord_str} output {tour_str}\n")

    # report normalized tour length of the label
    pts = coords[tour]
    tlen = np.sqrt(((pts - np.roll(pts, -1, axis=0)) ** 2).sum(1)).sum()
    print(f"Wrote {n}-city instance to {args.out}")
    print(f"Label = {label}; normalized tour length = {tlen:.4f}")


if __name__ == "__main__":
    main()
