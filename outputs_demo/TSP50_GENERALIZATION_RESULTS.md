# DIFUSCO TSP-50 → Larger Instances — Generalization Test (TSPLIB)

Testing the **TSP-50-trained** DIFUSCO checkpoint on **other TSPLIB instances** it has
**never seen** (22 up to 493 cities — both smaller and nearly 10× larger than the training size). The
model weights are frozen — this is a pure **size-generalization** test, no retraining.

- **Checkpoint:** `models/tsp50_concorde_1epoch/fm6khhos/checkpoints/last.ckpt`
  (the 1-epoch, Concorde-labelled TSP-50 model).
- **Why this works:** DIFUSCO's denoiser is an anisotropic **GNN** with weights shared across
  nodes/edges, so it runs on **any** number of cities `N`. A model trained at N=50 can be
  evaluated at N=70…150 directly.
- **Instances:** real TSPLIB problems, each solved to **provable optimality by Concorde** for the
  `gt_cost` baseline (TSPLIB coords min-max normalized to `[0,1]²` first, keeping aspect ratio).

> ⚠️ The checkpoint was trained for only **1 epoch on 2000 synthetic TSP-50 instances (CPU)**.
> This is a demonstration of *generalization behaviour*, not a reproduction of the paper's numbers.

---

## 1. Results

| Instance | Cities `N` | DIFUSCO(10) cost | DIFUSCO(10) gap | DIFUSCO(50) cost | DIFUSCO(50) gap | Baseline (heuristic) cost | Baseline (heuristic) gap | Optimal `gt_cost` (Concorde) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ulysses22` | 22  | 2.4268  | +1.05 %     | 2.4015  | **+0.00 %** | 2.4078  | +0.26 %     | 2.4015 |
| `att48`     | 48  | 4.4752  | +3.48 %     | 4.4752  | +3.48 %     | 4.4604  | **+3.14 %** | 4.3245 |
| `st70`      | 70  | 7.0807  | **+3.53 %** | 7.0807  | **+3.53 %** | 7.3454  | +7.40 %     | 6.8395 |
| `eil76`     | 76  | 7.7054  | +1.91 %     | 7.6629  | **+1.35 %** | 8.3202  | +10.05 %    | 7.5607 |
| `pr76`      | 76  | 5.6933  | +3.17 %     | 5.6916  | **+3.14 %** | 5.9275  | +7.42 %     | 5.5183 |
| `rat99`     | 99  | 5.8292  | **+1.83 %** | 5.9027  | +3.12 %     | 6.0684  | +6.01 %     | 5.7241 |
| `kroA100`   | 100 | 5.4837  | +1.40 %     | 5.4347  | **+0.49 %** | 5.8323  | +7.85 %     | 5.4079 |
| `pr124`     | 124 | 6.3244  | **+0.98 %** | 6.3244  | **+0.98 %** | 6.6586  | +6.31 %     | 6.2632 |
| `pr136`     | 136 | 9.4555  | +10.66 %    | 9.2485  | **+8.23 %** | 9.3604  | +9.54 %     | 8.5449 |
| `ch150`     | 150 | 9.7476  | +4.35 %     | 9.6020  | +2.79 %     | 9.5339  | **+2.06 %** | 9.3416 |
| `kroA200`   | 200 | 7.6919  | +3.22 %     | 7.6642  | +2.84 %     | 7.6267  | **+2.34 %** | 7.4523 |
| `pr299`     | 299 | 7.5957  | **+5.23 %** | 7.6283  | +5.68 %     | 7.6977  | +6.65 %     | 7.2181 |
| `rd400`     | 400 | 15.8145 | +3.06 %     | 15.7347 | **+2.54 %** | 16.4707 | +7.34 %     | 15.3446 |
| `d493`      | 493 | 9.6512  | **+3.22 %** | 9.7357  | +4.12 %     | 9.9309  | +6.21 %     | 9.3502 |

*`DIFUSCO(10)` and `DIFUSCO(50)` are the **same frozen TSP-50 checkpoint**, decoded with **10** vs
**50** reverse-diffusion steps (`--inference_diffusion_steps`). The bolded gap in each row is that
row's winner (smallest gap to optimal). **With 50 steps DIFUSCO beats the classical
Nearest-Neighbor + 2-opt heuristic on 11 of the 14 instances** (up from 9/14 at 10 steps).*

### 1.1 Condensed summary

| Instance | Cities `N` | Optimal cost | DIFUSCO(10) gap | DIFUSCO(50) gap | Baseline (heuristic) gap | Runtime(10) | Runtime(50) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ulysses22` | 22  | 2.4015  | +1.05 %     | **+0.00 %** | +0.26 %     | 4.6 s   | 6.9 s   |
| `att48`     | 48  | 4.3245  | +3.48 %     | +3.48 %     | **+3.14 %** | 5.6 s   | 11.8 s  |
| `st70`      | 70  | 6.8395  | **+3.53 %** | **+3.53 %** | +7.40 %     | 7.6 s   | 20.8 s  |
| `eil76`     | 76  | 7.5607  | +1.91 %     | **+1.35 %** | +10.05 %    | 7.6 s   | 25.6 s  |
| `pr76`      | 76  | 5.5183  | +3.17 %     | **+3.14 %** | +7.42 %     | 7.6 s   | 26.7 s  |
| `rat99`     | 99  | 5.7241  | **+1.83 %** | +3.12 %     | +6.01 %     | 9.7 s   | 35.6 s  |
| `kroA100`   | 100 | 5.4079  | +1.40 %     | **+0.49 %** | +7.85 %     | 9.7 s   | 36.7 s  |
| `pr124`     | 124 | 6.2632  | **+0.98 %** | **+0.98 %** | +6.31 %     | 12.6 s  | 50.7 s  |
| `pr136`     | 136 | 8.5449  | +10.66 %    | **+8.23 %** | +9.54 %     | 13.5 s  | 55.6 s  |
| `ch150`     | 150 | 9.3416  | +4.35 %     | +2.79 %     | **+2.06 %** | 16.7 s  | 71.6 s  |
| `kroA200`   | 200 | 7.4523  | +3.22 %     | +2.84 %     | **+2.34 %** | 30.7 s  | 155.6 s |
| `pr299`     | 299 | 7.2181  | **+5.23 %** | +5.68 %     | +6.65 %     | 63.7 s  | 357.8 s |
| `rd400`     | 400 | 15.3446 | +3.06 %     | **+2.54 %** | +7.34 %     | 123.7 s | 627.0 s |
| `d493`      | 493 | 9.3502  | **+3.22 %** | +4.12 %     | +6.21 %     | 193.8 s | 982.0 s |

*Runtime = end-to-end wall-clock per instance on CPU (includes ~2–3 s Python/torch/wandb startup and
both a validate and a test pass over the single instance). `Runtime(10)` / `Runtime(50)` are the
10-step and 50-step decodes. It grows with `N` because the graph is dense (`N×N` edges): ch150 is a
150×150 = 22 500-edge graph vs st70's 4 900, so each diffusion step and the greedy merge do much more
work — and running 50 steps instead of 10 multiplies the GNN passes ~5×.*

For reference, the model's **in-distribution** TSP-50 test gap was **+3.0 %**.

- `solved_cost` — length of the tour **DIFUSCO produced** (lower is better).
- `Baseline (heuristic) cost` — length of a tour from a **classical heuristic** (Nearest-Neighbor
  construction + 2-opt refinement to a local optimum), run on the **same normalized points**. This is
  the non-learned baseline the GNN is compared against.
- `gt_cost` — length of the **provably optimal** tour (Concorde) on the same normalized points.
- `gap = (cost − gt_cost) / gt_cost` — how far above optimal each method's tour is.
- `eil76` fills the requested "60-city" slot: **TSPLIB has no 60-city instance**, and `eil76` is the
  nearest real problem with clean Euclidean coordinates.
- `ulysses22` (originally GEO) and `att48` (originally ATT) have non-Euclidean metrics in TSPLIB. Here
  their **2D coordinates are reinterpreted as planar Euclidean points** after normalization, and
  Concorde computes the optimal under that same Euclidean metric — so DIFUSCO, the baseline, and the
  optimal are all self-consistent (these costs are **not** comparable to TSPLIB's official GEO/ATT
  optima).

### What the numbers say

- **The TSP-50 model generalizes remarkably well across the whole range — 22 up to ~10× its training
  size.** On 13 of 14 instances the gap sits in a **~1 – 5 %** band, right around its own
  in-distribution 50-city gap (+3.0 %), with the sub-1 % `pr124` (+0.98 %) its single best result.
  There is **no collapse** as `N` grows: d493 (+3.22 %) is no worse than st70 (+3.53 %). The one
  outlier is `pr136` (+10.66 %) — the sole instance where the frozen heatmap badly misreads the
  layout. This is the headline result — a tiny 1-epoch model holds its relative quality across an
  order of magnitude in problem size.
- Most of that robustness comes from the **greedy-merge + 2-opt decoder**, not the raw heatmap. As
  `N` grows the heatmap gets noisier, but the decoder repairs it — and the cost of that repair is
  visible: `merge_iters` explodes from **1.6k (st70)** to **230k (d493)**. At d493 the merge scans
  almost the entire `N²` ≈ 243k edge list. So quality is held roughly constant, but **decode effort
  (and runtime) grows super-linearly** with `N`.
- **`pr136` is the hardest** (+10.66 %), with `pr299` next (+5.23 %): the `pr*` family is highly
  structured/grid-like, which the heatmap handles worst. **`pr124` is easiest** (+0.98 %), followed by
  `kroA100` (+1.40 %) — clustered points where even an imperfect heatmap yields a near-optimal tour.
  Note `pr124` and `pr136` are both `pr*` yet land at opposite extremes, underlining how sensitive the
  result is to the exact layout.
- The gap is driven more by **instance structure** than by raw size: there's no clean upward trend
  with `N`. With a single instance per size these are **point estimates, not averages** — the
  ranking reflects each problem's layout as much as its city count.
- **vs the classical NN + 2-opt baseline (at 10 steps): DIFUSCO wins 9 of 14.** The learned model is clearly better
  on the small/medium clustered instances (eil76 +1.91 % vs +10.05 %, kroA100 +1.40 % vs +7.85 %,
  pr124 +0.98 % vs +6.31 %) and holds its edge at the largest sizes (rd400, d493). The heuristic wins
  on `ulysses22`, `att48`, `pr136`, `ch150` and `kroA200` — the tiny instances (where NN + 2-opt is
  already near-optimal) and the few where DIFUSCO's heatmap was weakest, so 2-opt from a greedy NN
  tour happened to land lower. **Caveat on cost:** the heuristic is ~1000× faster (NN + 2-opt finishes
  in **0.001–0.27 s**, vs DIFUSCO's 4.6–193.8 s on CPU), because it skips the diffusion passes
  entirely. So the takeaway is *quality*, not speed: a TSP-50 model generalizes across a wide size
  range and still **out-qualities a standard hand-written heuristic on most instances** — which is the
  point of the learned approach.
- **10 vs 50 diffusion steps: more steps help, but modestly and not for free.** Going from 10 → 50
  reverse-diffusion steps improves the gap on **8 of the 14** instances, leaves 3 unchanged
  (att48, st70, pr124), and slightly *worsens* 3 (`rat99` +1.83 % → +3.12 %, `pr299` +5.23 % → +5.68 %,
  `d493` +3.22 % → +4.12 %) — diffusion decoding is **stochastic**, so extra steps aren't monotonically
  better on every instance. The biggest wins are exactly the hard/noisy cases: `pr136`
  +10.66 % → **+8.23 %**, `ch150` +4.35 % → +2.79 %, `kroA100` +1.40 % → +0.49 %, and `ulysses22`
  reaches the **optimal** (+0.00 %). The cost: runtime roughly **2.5–5× longer** (e.g. kroA200
  30.7 s → 155.6 s, pr299 63.7 s → 357.8 s, d493 193.8 s → 982.0 s), since each instance now runs 50
  GNN denoising passes instead of 10. With 50 steps DIFUSCO's head-to-head record vs the heuristic
  improves to **11 of 14**.

---

## 2. Method (per instance)

1. **Import** the TSPLIB `.tsp` file; parse coordinates with `tsplib95`.
2. **Normalize** coordinates into `[0,1]²` with one uniform scale (preserves aspect ratio) — the
   same scale DIFUSCO trained on.
3. **Label** the normalized instance with **Concorde** → provably optimal tour = `gt_cost`.
4. **Test** the frozen TSP-50 checkpoint on that instance: reverse-diffusion (10 steps, cosine
   schedule) → edge heatmap → greedy cycle-merge → 2-opt → `solved_cost`.

```
TSPLIB .tsp ─► normalize [0,1]² ─► Concorde (optimal label = gt_cost)
                     │
                     └─► TSP-50 checkpoint ─► 10× reverse diffusion ─► heatmap
                                                   │
                                        greedy merge ─► 2-opt ─► solved_cost
```

---

## 3. Files

| What | Path |
|---|---|
| TSPLIB raw instances | [`data/tsplib_raw/`](../data/tsplib_raw/) (`ulysses22`, `att48`, `st70`, `eil76`, `pr76`, `rat99`, `kroA100`, `pr124`, `pr136`, `ch150`, `kroA200`, `pr299`, `rd400`, `d493`) |
| Normalized + Concorde-labelled | [`data/tsplib_large/`](../data/tsplib_large/) (`*_concorde.txt`) |
| TSPLIB→Concorde converter | [`scripts_demo/tsplib_concorde.py`](../scripts_demo/tsplib_concorde.py) |
| NN + 2-opt heuristic baseline | [`scripts_demo/nn_2opt_baseline.py`](../scripts_demo/nn_2opt_baseline.py) |
| Per-instance test logs | [`outputs_demo/large/`](large/) (`test_<name>.log`) |
| Checkpoint under test | `models/tsp50_concorde_1epoch/fm6khhos/checkpoints/last.ckpt` |

---

## 4. How to reproduce

```bash
conda activate difusco
cd DIFUSCO
export PYTHONPATH="$PWD:$PYTHONPATH"; export WANDB_MODE=offline
CKPT=models/tsp50_concorde_1epoch/fm6khhos/checkpoints/last.ckpt

for name in ulysses22 att48 st70 eil76 pr76 rat99 kroA100 pr124 pr136 ch150 kroA200 pr299 rd400 d493; do
  # 1) import + normalize + Concorde-optimal label
  python -I scripts_demo/tsplib_concorde.py \
    --tsp data/tsplib_raw/$name.tsp \
    --out data/tsplib_large/${name}_concorde.txt

  # 2) test the frozen TSP-50 checkpoint on it
  f=data/tsplib_large/${name}_concorde.txt
  python -u difusco/train.py \
    --task tsp --diffusion_type categorical --do_test \
    --ckpt_path "$CKPT" --storage_path "$PWD" \
    --training_split "$f" --validation_split "$f" --test_split "$f" \
    --wandb_logger_name tsp50_test_${name} \
    --batch_size 1 --num_epochs 1 --num_workers 1 --validation_examples 1 \
    --diffusion_steps 1000 --inference_schedule cosine \
    --inference_diffusion_steps 10 --two_opt_iterations 1000

  # 3) classical NN + 2-opt heuristic baseline on the SAME normalized points
  python -I scripts_demo/nn_2opt_baseline.py --inp "$f"
done
```

(The `.tsp` files come from the mastqe/tsplib mirror; Concorde provides the optimal labels, so no
`.opt.tour` files are needed.)
