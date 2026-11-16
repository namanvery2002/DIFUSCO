# DIFUSCO TSP-50 — 1-Epoch Train + Test Demo (Concorde-optimal labels)

A minimal, self-contained run of DIFUSCO on the **Traveling Salesman Problem with 50 cities (TSP-50)**,
trained for **1 epoch** and then **tested**, on **CPU only** (no GPU available on this machine).
Tour labels are **provably optimal**, solved by the **Concorde** exact solver.

> ⚠️ This is a *demonstration* run, not a reproduction of the paper. It uses a small dataset and a
> single training epoch, so tour quality is far from the paper's numbers.

---

## 1. Results

| Metric | Value | Meaning |
|---|---:|---|
| `test/solved_cost` | **5.882** | Avg. length of the tour **DIFUSCO produced** (lower is better) |
| `test/gt_cost` | 5.710 | Avg. length of the **optimal label tour** (Concorde) |
| optimality gap | **+3.0%** | `(5.882 − 5.710) / 5.710` — model is 3% above true optimal |
| final `train/loss` | ~0.095 | Cross-entropy of the denoiser on the training batches |
| `test/2opt_iterations` | 7.9 | Avg. 2-opt sweeps applied to refine each decoded tour |
| `test/merge_iterations` | 1285.2 | Avg. greedy edge-merge steps used to decode a tour from the heatmap |

Because the labels are **provably optimal** (Concorde), the +3.0% gap is honest: after just 1 epoch
on CPU with 10 inference steps, DIFUSCO's tours are ~3% longer than optimal — a believable demo
result. Concorde solves each 50-city instance to optimality in ~0.05 s.

- Model size: **5,333,762** trainable parameters (12-layer anisotropic GNN, hidden dim 256).
- Training steps in the epoch: **31** (2000 instances / batch size 64, `drop_last`).
- Wall-clock: **~9 min total** (~7.5 min train + ~1.5 min test) on a 16-core CPU.
  Concorde only affects label *quality*, not training time; it added ~2 min of one-time data generation.

---

## 2. What the task is

Given 50 points (cities) in the unit square, find the shortest cyclic route that visits every city
exactly once and returns to the start. DIFUSCO treats this as **learning to generate the set of edges**
that form the optimal tour, using a **graph-based discrete diffusion model**.

---

## 3. The model process, end to end

### 3.1 Input to the model

For each instance the dataset ([tsp_graph_dataset.py](../difusco/co_datasets/tsp_graph_dataset.py)) produces:

| Tensor | Shape (per instance) | What it is |
|---|---|---|
| `points` | `[50, 2]` | the (x, y) coordinates of the 50 cities — the **node features** |
| `adj_matrix` | `[50, 50]` | the **ground-truth tour** as a 0/1 adjacency matrix (edge = 1 if the tour goes i→j). This is the training *target*. |
| `t` (train) | scalar per instance | a random **diffusion timestep** in `[1, 1000]` |
| `tour` | `[50]` | the ground-truth visiting order (used only to build `adj_matrix` / for evaluation) |

Because we run the **dense** variant (`sparse_factor = -1`), the graph is fully connected: every
pair of cities is an edge, and the model predicts a label for each of the 50×50 edges.

### 3.2 Training (categorical diffusion) — one epoch

Per batch ([pl_tsp_model.py `categorical_training_step`](../difusco/pl_tsp_model.py)):

1. **Add noise (forward diffusion).** Take the clean 0/1 edge matrix, one-hot it to 2 classes,
   and corrupt it to a noisy version `x_t` by sampling from the categorical diffusion kernel at a
   random timestep `t`. Larger `t` ⇒ closer to pure noise.
2. **Denoise (the network).** The GNN (`GNNEncoder`) sees the **city coordinates**, the **noisy edge
   matrix `x_t`**, and the **timestep `t`**, and predicts the **clean edge labels `x0`** (2 logits per edge).
3. **Loss.** Cross-entropy between the predicted edge logits and the true 0/1 edge labels.
   Backprop with AdamW (lr 2e-4, cosine-decay, weight decay 1e-4).

The network: coordinate embedding + sinusoidal timestep embedding feed a 12-layer anisotropic
graph neural network with edge gating; it outputs a 2-class logit for every edge. See
[gnn_encoder.py](../difusco/models/gnn_encoder.py).

### 3.3 Output of the model + decoding (test time)

At test time ([pl_tsp_model.py `test_step`](../difusco/pl_tsp_model.py)) we **generate** a tour by
running the reverse diffusion:

1. **Start from noise.** Initialize the edge matrix `x_T` as random 0/1 noise.
2. **Iterative denoising.** For each of the `inference_diffusion_steps` (= **10** here; the paper uses
   up to 50) run the categorical reverse step: the GNN predicts clean edge probabilities, and we
   sample the next, less-noisy edge matrix from the categorical posterior. After the last step we
   have a **heatmap**: a score in `[0, 1]` for every edge (how likely it belongs to the tour).
3. **Decode a tour from the heatmap.** A greedy merge
   ([cython_merge](../difusco/utils/cython_merge/)) adds edges in order of score while keeping the
   result a valid single cycle → one concrete tour.
4. **Refine with 2-opt.** Local 2-opt swaps shorten the tour (`two_opt_iterations = 1000`).
5. **Score.** `TSPEvaluator` computes the tour length (`solved_cost`) and compares it to the length
   of the optimal label tour (`gt_cost`).

```
coords (50x2) ──► [ + random noisy edges x_T, timestep T ]
                              │
                   10× reverse-diffusion steps  (GNN denoiser)
                              │
                         edge heatmap (50x50 in [0,1])
                              │
                   greedy cycle-merge  ──►  valid tour
                              │
                         2-opt refinement
                              │
                      final tour  ──►  length = solved_cost
```

---

## 4. Exact configuration used

| Argument | Value |
|---|---|
| `--task` | `tsp` |
| `--diffusion_type` | `categorical` |
| `--num_epochs` | **1** |
| `--batch_size` | 64 |
| `--diffusion_steps` (train T) | 1000 |
| `--inference_diffusion_steps` | 10 *(reduced from 50 for CPU speed)* |
| `--inference_schedule` | `cosine` |
| `--learning_rate` | 2e-4 |
| `--lr_scheduler` | `cosine-decay` |
| `--weight_decay` | 1e-4 |
| `--n_layers` / `--hidden_dim` | 12 / 256 (defaults) |
| `--sparse_factor` | -1 (dense graph) |
| `--two_opt_iterations` | 1000 |
| `--validation_examples` | 8 |
| accelerator | CPU (no CUDA on this machine) |

---

## 5. Data

Synthetic TSP-50 with **Concorde-optimal** tour labels:

- **Train:** 2000 instances — [`data/tsp50_concorde/tsp50_train_concorde.txt`](../data/tsp50_concorde/tsp50_train_concorde.txt)
- **Test / Val:** 128 instances — [`data/tsp50_concorde/tsp50_test_concorde.txt`](../data/tsp50_concorde/tsp50_test_concorde.txt)

Each instance = 50 uniform-random points in the unit square; the label tour is solved to
**provable optimality by Concorde** ([pyconcorde](https://github.com/jvkersch/pyconcorde), which
builds Concorde + the QSopt LP backend from source).
Generator: [`scripts_demo/gen_tsp50_concorde.py`](../scripts_demo/gen_tsp50_concorde.py).

File format (one instance per line, matching DIFUSCO's Concorde format):
```
x1 y1 x2 y2 ... x50 y50 output n1 n2 ... n50 n1
```
i.e. 50 coordinate pairs, the literal token `output`, then the tour as 1-indexed node ids with the
first node repeated at the end.

---

## 6. How to reproduce

```bash
conda activate difusco      # CPU env: torch 1.11, pytorch-lightning 1.7.7, torch-geometric 2.2
cd DIFUSCO
export PYTHONPATH="$PWD:$PYTHONPATH"
export WANDB_MODE=offline

# (data already generated under data/tsp50_concorde/; to regenerate with Concorde:)
# python -I scripts_demo/gen_tsp50_concorde.py --num_samples 2000 --seed 1234 \
#   --filename data/tsp50_concorde/tsp50_train_concorde.txt
# python -I scripts_demo/gen_tsp50_concorde.py --num_samples 128  --seed 4321 \
#   --filename data/tsp50_concorde/tsp50_test_concorde.txt

python -u difusco/train.py \
  --task tsp --diffusion_type categorical --do_train --do_test \
  --storage_path "$PWD" \
  --training_split  data/tsp50_concorde/tsp50_train_concorde.txt \
  --validation_split data/tsp50_concorde/tsp50_test_concorde.txt \
  --test_split       data/tsp50_concorde/tsp50_test_concorde.txt \
  --wandb_logger_name tsp50_concorde_1epoch \
  --batch_size 64 --num_epochs 1 --num_workers 4 --validation_examples 8 \
  --diffusion_steps 1000 --inference_schedule cosine --inference_diffusion_steps 10 \
  --learning_rate 0.0002 --weight_decay 0.0001 --lr_scheduler cosine-decay \
  --two_opt_iterations 1000
```

Full console log: [`outputs_demo/run_concorde.log`](run_concorde.log).

---

## 7. Notes / deviations from the paper

1. **CPU only** — the pinned environment is CUDA-only; this run uses CPU wheels (torch 1.11+cpu, matching PyG wheels).
2. **Tiny data + 1 epoch** — paper trains 50 epochs on ~1.5M instances; results here are not comparable.
3. **10 inference steps** instead of 50, to keep test time short on CPU.
4. Fixed a Python-3.9 incompatibility: `pickle5` import in
   [`mis_dataset.py`](../difusco/co_datasets/mis_dataset.py) now falls back to the stdlib `pickle`.
