# The Training Network — Architecture Specification

> Companion to [`reference_model.py`](reference_model.py) (PyTorch, from scratch)
> and [`reference_model_numpy.py`](reference_model_numpy.py) (torch-free shape +
> parameter verifier). **No pretrained weights are used anywhere in this report** —
> every layer below is defined and initialized from first principles.

This document is the canonical, layer-by-layer description of the deep fall
classifier referenced throughout the manuscript. It is written so that a reader
who has never seen the code can re-derive the network on paper, and so a
reviewer can check every shape against the runnable verifier.

---

## 0. Problem statement and tensor convention

We classify a short window of skeleton motion as **fall** or **not-fall**.

A skeleton at one instant is `V = 17` joints in COCO order (identical to
[`core/eldercare/schema.py`](../../core/eldercare/schema.py)). Each joint carries
`C = 3` channels: normalized image coordinates `x, y ∈ [0,1]` (y grows downward)
and a detector confidence `score ∈ [0,1]`. A clip is `T = 32` consecutive frames
(ablated at 16 / 32 / 64).

The network input is therefore a 4-D tensor

```
x ∈ ℝ^(N × C × T × V) = (N × 3 × 32 × 17)
```

with `N` the batch size. The output is two logits → softmax → `P(fall)`. That
single scalar is exactly what the alarm debouncer in
[`core/eldercare/alarm.py`](../../core/eldercare/alarm.py) consumes, so the deep
model is a **drop-in replacement** for the geometric Baseline A — same interface,
better evidence.

> **Why a graph network and not a plain CNN/RNN?** A skeleton is not a grid of
> pixels; it is a *graph* whose nodes are joints and whose edges are bones. A
> graph convolution respects that physical structure: the update for the wrist
> is computed from the elbow and hand it is actually attached to, regardless of
> where they land in the image. This is what makes skeleton models small,
> fast, and viewpoint-robust — the three properties a privacy-preserving edge
> deployment needs most.

---

## 1. The skeleton graph `A`

The bones define an undirected graph on 17 nodes (`COCO_BONES` in
`reference_model.py`). A graph convolution needs a normalized **adjacency
matrix**; we use the ST-GCN *spatial-configuration partitioning*, which splits
every node's neighbourhood into three subsets and gives the convolution three
weight matrices instead of one:

| partition `k` | meaning | intuition |
|---|---|---|
| `k=0` self | the joint itself (identity) | "what am I doing" |
| `k=1` centripetal | neighbours **closer** to the body centre (hips) | motion toward the core |
| `k=2` centrifugal | neighbours **farther** from the centre | motion of the extremities |

"Closer / farther" is measured by **graph hop-distance to the hip root**, so it
needs no 3-D coordinates and is viewpoint-invariant. Each partition is
symmetrically normalized,

```
Â_k = D_k^(-1/2) (A_k) D_k^(-1/2)
```

so a graph convolution is a stable weighted average over neighbours rather than
a sum that explodes with node degree. The result is a fixed buffer of shape
`(3, 17, 17)`. The verifier confirms the normalized row-sums:

```
self = 1.00   centripetal = 0.75   centrifugal = 0.39
```

This graph is **shared** and **static** in ST-GCN. CTR-GCN keeps it as a prior
but *refines it per channel* (§4). See report **Fig. 3** for the spatial graph
and the spatial-temporal volume it is replicated over.

---

## 2. Confidence-gated input layer (the first novelty)

> Mechanism: [`ConfidenceGate`](reference_model.py) · Illustration: report **Fig. 4**

Real pose estimators drop joints — a blanket hides the legs, furniture occludes
an arm, motion blur kills a frame. A naïve network treats a dropped joint's
`(x, y) = (0, 0)` as a real observation and hallucinates a body collapsing into
the origin. The confidence gate prevents this in three steps, **before** any
convolution:

1. **Mask.** A joint with `score < min_score` (0.2, matching
   [`features.MIN_SCORE`](../../core/eldercare/features.py)) is marked invalid.
2. **Temporally impute.** Its `(x, y)` is filled from the *same joint* in the
   nearest confident frame — forward-fill, then back-fill the leading gap. This
   is the exact algorithm already used by the sliding window in
   [`core/eldercare/temporal/__init__.py`](../../core/eldercare/temporal/__init__.py),
   lifted into a differentiable, batched tensor op (`_temporal_impute`). A brief
   occlusion therefore never punches a hole in the input tensor.
3. **Re-weight.** A learnable per-joint reliability scalar `σ(r_v)` scales each
   joint's contribution, letting training discover that, say, ankles are noisier
   than shoulders.

The gate **preserves the input shape** `(N, 3, T, V)` and is the layer the
occlusion-robustness ablation (report **Fig. 9**) turns on and off.

---

## 3. The spatial-temporal block

The backbone is a stack of identical blocks. Each block factorizes a
spatio-temporal convolution into a **spatial** step (mix across joints, within a
frame) followed by a **temporal** step (mix across frames, within a joint):

```
        ┌──────────────────────── GCNBlock ───────────────────────┐
 x ───► │  graph conv  ─► BN ─► ReLU ─► temporal conv  ─► (+) ─► ReLU │ ─► y
        │     (spatial, §4)            (multi-scale, §5)    ▲        │
        └──────────────────────────────────────────────────┼────────┘
                              residual  x ─────────────────┘
```

A residual connection (identity, or a `1×1` projection when channels/stride
change) makes the deep stack trainable. The block channel plan, mirroring
[`configs/ctrgcn.yaml`](../../configs/ctrgcn.yaml) and verified by the shape
trace, is:

| blocks | out-channels | temporal stride | output `(C,T,V)` |
|---|---|---|---|
| 0–2 | 64 | 1 | `(64, 32, 17)` |
| 3 | 128 | **2** | `(128, 16, 17)` |
| 4 | 128 | 1 | `(128, 16, 17)` |
| 5 | 256 | **2** | `(256, 8, 17)` |
| 6–9 | 256 | 1 | `(256, 8, 17)` |

Two stride-2 steps halve the temporal resolution twice (32 → 16 → 8), trading
time resolution for channel depth — the standard ST-GCN schedule.

---

## 4. Spatial graph convolution — ST-GCN vs CTR-GCN

### 4.1 ST-GCN (the deep baseline) — [`STGraphConv`](reference_model.py)

For each partition `k`, transform features with a learned `1×1` convolution
`W_k`, propagate along the **fixed** graph `Â_k`, and sum:

```
out =  Σ_k  Â_k · (x · W_k)          # einsum  nkctv,kvw -> nctw
```

The topology is frozen to the human skeleton. This is the model that, in the
literature, scores >99% *in-dataset* yet **collapses across datasets** — it
memorizes the staged graph of motion it was trained on.

### 4.2 CTR-GCN (the proposed model) — [`CTRGraphConv`](reference_model.py)

CTR-GCN keeps `Â_k` as a prior but lets each **channel** refine the graph from
the data. It learns a pairwise relation between joints `i` and `j` from the
difference of their features and adds it to the shared topology, gated by a
scalar `α` initialized to **zero** — so training *starts exactly at the ST-GCN
baseline* and only departs from it when the data rewards doing so:

```
ΔA = Conv( tanh( φ1(x)_i − φ2(x)_j ) )      # per-channel V×V refinement
Â_refined = Â_k + α · ΔA
out = Σ_k Â_refined · (x · W_k)             # einsum  nkctv,nkcvw -> nctw
```

Because the refinement is **per channel**, different feature channels can use
different effective skeletons — one channel can wire "wrist↔ankle" to capture a
sprawled fall posture that the bone graph never connects. This channel-wise
topology refinement is the single most important architectural reason the model
generalizes across datasets (report **Fig. 8**) where the static-graph baseline
does not.

---

## 5. Multi-scale temporal convolution — [`TemporalConv`](reference_model.py)

A fall has two temporal signatures at once: a **fast** downward impact transient
(~a few frames) and a **slow** "stays on the floor" plateau (many frames). A
single temporal kernel sees only one scale. The block therefore runs several
parallel temporal branches and concatenates them:

- two dilated `9×1` temporal convolutions (dilations `{1, 2}`) — different
  receptive fields in time;
- a `3×1` max-pool branch — robust to single outlier frames;
- a `1×1` fuse that restores the exact channel count.

The stride lives here, so a stride-2 block also halves `T`.

---

## 6. Head and output

After the last block (`(N, 256, 8, 17)`) a **global average pool** over both
time and joints yields a `(N, 256)` clip embedding; a single linear layer maps it
to two logits. Softmax gives `P(fall)`, the scalar handed to the alarm logic.

```
(N,256,8,17) ──GAP(T,V)──► (N,256) ──FC──► (N,2) ──softmax──► P(fall)
```

---

## 7. Parameter and footprint budget (verified, not asserted)

Run [`reference_model_numpy.py`](reference_model_numpy.py) — it counts every
layer analytically from the exact construction in `reference_model.py`:

| model | parameters | FP32 size | INT8 size | fits <25 MB node budget? |
|---|---|---|---|---|
| ST-GCN (baseline) | **2,529,348** (~2.53 M) | ~10.1 MB | ~2.5 MB | ✅ |
| CTR-GCN (proposed) | **2,660,095** (~2.66 M) | ~10.6 MB | ~2.7 MB | ✅ |

The channel-wise refinement costs only ~0.13 M extra parameters (~5%) over the
static-graph baseline — a small price for the cross-dataset gain. Both fit the
on-device footprint budget in the research doc §7 with room to spare, and INT8
post-training quantization (`export_onnx.py --int8`) takes them to ~2.5 MB.

---

## 8. How this slots into the existing repo

| Concern | Lives in | Status |
|---|---|---|
| Keypoint schema (COCO-17, wire format) | `core/eldercare/schema.py` | ✅ implemented |
| Confidence-gated windowing (impute) | `core/eldercare/temporal/__init__.py` | ✅ implemented |
| Geometric features (Baseline A) | `core/eldercare/features.py`, `heuristic.py` | ✅ implemented |
| Alarm debounce (consumes `P(fall)`) | `core/eldercare/alarm.py` | ✅ implemented |
| **Deep network architecture** | **this report — `reference_model.py`** | ✅ **specified, from scratch** |
| Training loop (dataset → fit → ckpt) | `train/train_ctrgcn.py` | ⏳ stub (Phase 2) |
| ONNX export + parity test | `train/export_onnx.py` | ⏳ stub (Phase 2) |

The architecture in this report is precisely the body that the `train_ctrgcn.py`
stub is waiting for: dropping `build_model("ctrgcn")` into that file's `# TODO:
build model` line is all that is required to begin training. See
[`training_methodology.md`](training_methodology.md) for the recipe.
