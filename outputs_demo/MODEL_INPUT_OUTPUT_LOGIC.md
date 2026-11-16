# DIFUSCO TSP — Inputs, Outputs, and the Logic (plain-language explainer)

A compact guide to **what goes into the model, what comes out, and how the diffusion logic
connects the two**. It explains the TSP-50 setup used in this repo (dense graph,
categorical diffusion), but the same logic holds at any `N`.

> **One-sentence summary:** DIFUSCO learns to turn a **noisy guess of which edges form the tour**
> back into the **clean set of tour edges**, and at test time it starts from pure noise and repeats
> that "clean-up" step until a real tour appears.

---

## 1. The big picture

TSP asks: *given `N` cities, what is the shortest cyclic route visiting each once?*

DIFUSCO does **not** output a route directly. It outputs, for **every pair of cities**, a score for
the question *"does the optimal tour use the edge between these two cities?"* — a **heatmap** over
edges. A simple decoder then stitches the high-scoring edges into one valid tour.

It learns this the way image diffusion models learn: by practising **removing noise**. Here the
"image" being denoised is not pixels — it is the **0/1 edge matrix of the tour**.

```
      cities (coordinates)                    the tour, as edges
            [N, 2]                               [N, N]  (0/1)
               │                                     ▲
               │            THE MODEL                │
               └──────────►  (GNN denoiser)  ────────┘
                          learns the mapping
                   coords + noisy edges  ──►  clean edges
```

---

## 2. INPUT — what the model actually receives

For each instance the dataset
([tsp_graph_dataset.py](../difusco/co_datasets/tsp_graph_dataset.py)) and training step
([pl_tsp_model.py](../difusco/pl_tsp_model.py)) hand the network these tensors:

| Tensor | Shape | What it is | Role |
|---|---|---|---|
| `points` | `[N, 2]` | the `(x, y)` coordinates of the `N` cities | **node features** — the *fixed* problem |
| `x_t` | `[N, N]` | a **noisy** 0/1 edge matrix (the tour corrupted by noise) | **the thing being denoised** |
| `t` | scalar | the **diffusion timestep** = how much noise is in `x_t` | tells the model *how noisy* its input is |

And during **training only**, one more tensor is the *target* (the answer, not an input):

| Tensor | Shape | What it is | Role |
|---|---|---|---|
| `adj_matrix` (`x_0`) | `[N, N]` | the **clean** ground-truth tour as a 0/1 edge matrix | **training target** |
| `tour` | `[N]` | the ground-truth visiting order | only used to build `adj_matrix` / to score |

### How the three inputs fit together

- **`points` is the question.** It never changes and is never noised — the cities are a given.
- **`x_t` is a noisy guess at the answer.** Start from the clean tour matrix `x_0` (1 where the tour
  uses edge `i–j`, else 0), then randomly flip edges according to the noise level `t`.
- **`t` is the noise dial**, a scalar in `[1, 1000]`:
  - `t ≈ 1` → `x_t` is almost the real tour (a few edges flipped).
  - `t ≈ 1000` → `x_t` is almost pure random edges.
  - Each instance draws its **own random `t`** during training, so the model practises every noise
    level — easy clean-ups *and* near-impossible ones — equally across the epoch.
  - It is fed in because a denoiser's job at `t=10` (polish) is completely different from `t=950`
    (reconstruct from almost nothing); the network must know which it faces.

> Dense variant (`sparse_factor = -1`): the graph is **fully connected**, so there is a label for
> every one of the `N×N` city pairs. That is why both `x_t` and the output are `[N, N]`.

---

## 3. OUTPUT — what the model produces

The network outputs, **per edge, a 2-class logit**: the scores for the two classes
*"edge NOT in tour" (0)* and *"edge IN tour" (1)*.

| Output | Shape | Meaning |
|---|---|---|
| edge logits | `[N, N, 2]` | for each city pair, how strongly the model believes that edge is / isn't in the tour |

- During **training**, these logits are compared to the true 0/1 labels `x_0` with **cross-entropy
  loss** — "did you predict the right class for each edge?"
- During **inference**, the logits become **edge probabilities** in `[0, 1]`; collected over all
  edges they form the **heatmap** that the decoder turns into a tour.

So the raw model output is **not a tour** — it is a per-edge belief. A tour is produced afterward by
decoding (§5).

---

## 4. THE LOGIC — training (learning to denoise)

One training step, per batch
([pl_tsp_model.py `categorical_training_step`](../difusco/pl_tsp_model.py)):

1. **Take the clean answer `x_0`** — the true 0/1 tour-edge matrix.
2. **Pick a random `t`** in `[1, 1000]` for each instance.
3. **Add noise (forward diffusion).** Corrupt `x_0` into `x_t` by sampling from the categorical
   noise kernel at level `t`. Bigger `t` ⇒ more edges flipped ⇒ noisier.
4. **Denoise (the network).** The GNN sees `points`, `x_t`, and `t`, and predicts the clean edge
   labels `x_0` (2 logits per edge). See [gnn_encoder.py](../difusco/models/gnn_encoder.py):
   coordinate embedding + sinusoidal `t` embedding feed a 12-layer anisotropic GNN with edge gating.
5. **Loss + update.** Cross-entropy between predicted edge logits and the true labels; backprop with
   AdamW.

```
 x_0  ──(add noise at level t)──►  x_t  ──► [ GNN(points, x_t, t) ] ──► predicted x_0
 clean tour                       noisy tour                            (compared to true x_0
                                                                         via cross-entropy loss)
```

The model never memorises specific tours — it learns the **local rule**: *given these coordinates
and this noisy edge pattern, which edges really belong?* Because the GNN's weights are **shared
across all nodes and edges**, that rule works at any `N` — which is why a TSP-50 model can be run on
a 400-city instance.

---

## 5. THE LOGIC — inference (generating a tour from noise)

At test time ([pl_tsp_model.py `test_step`](../difusco/pl_tsp_model.py)) there is no ground truth —
the model **generates**:

1. **Start from pure noise.** Initialise the edge matrix `x_T` as random 0/1 edges.
2. **Iterative denoising (reverse diffusion).** For each of the `inference_diffusion_steps`
   (**10** here; the paper uses up to 50), the GNN predicts clean edge probabilities, and we sample a
   **less-noisy** edge matrix for the next step. Step by step, random edges resolve into a confident
   pattern. After the last step we have the **heatmap** (a score in `[0,1]` per edge).
3. **Decode a tour from the heatmap.** A **greedy cycle-merge**
   ([cython_merge](../difusco/utils/cython_merge/)) adds edges from highest score down, skipping any
   that would break the single-cycle rule → one concrete valid tour.
4. **Refine with 2-opt.** Local 2-opt swaps shorten that tour (`two_opt_iterations = 1000`).
5. **Score.** `TSPEvaluator` measures the tour length = `solved_cost`, compared to Concorde's
   optimal `gt_cost`.

```
 x_T (random edges)
      │
      │   10× reverse-diffusion steps   (GNN denoiser, using points + current x_t + t)
      ▼
 edge heatmap  [N, N] in [0,1]
      │
      │   greedy cycle-merge   (add best edges, keep one valid cycle)
      ▼
 a valid tour
      │
      │   2-opt refinement   (swap to shorten)
      ▼
 final tour  ──►  length = solved_cost
```

> **Key insight:** training and inference run the **same denoiser**. Training teaches it to undo
> *one* noise step at *any* level; inference chains many such steps, from full noise down to a clean
> tour.

---

## 6. Why the output is edges, not a route (and why that generalises)

- Predicting a **route** (an ordered list) would tie the network to a specific `N` and a specific
  ordering. Predicting **per-edge beliefs** keeps every prediction *local* — it only ever asks "is
  this one edge good?" — so the same weights apply to 50 or 500 cities.
- The messy global constraint ("must be exactly one Hamiltonian cycle") is **not** forced on the
  neural net. The net just emits a heatmap; the **decoder** (greedy merge + 2-opt) enforces validity.
  This split is what lets a tiny, imperfect heatmap still yield a good tour — the decoder repairs it.

---

## 7. Cheat-sheet

| | Training | Inference |
|---|---|---|
| `points` `[N,2]` | input (fixed) | input (fixed) |
| `x_t` `[N,N]` | clean tour + random noise at level `t` | starts as pure noise, denoised step by step |
| `t` | one random value per instance | walks from high → low over the 10 steps |
| model output `[N,N,2]` | compared to true `x_0` (cross-entropy) | becomes the heatmap |
| ground-truth `x_0` | the target | not available |
| tour | already known (the label) | decoded from heatmap (greedy merge + 2-opt) |

**Related docs:** [`TSP50_DEMO_RESULTS.md`](TSP50_DEMO_RESULTS.md) (the 1-epoch train/test demo) and
[`TSP50_GENERALIZATION_RESULTS.md`](TSP50_GENERALIZATION_RESULTS.md) (running the frozen model on
larger TSPLIB instances).
