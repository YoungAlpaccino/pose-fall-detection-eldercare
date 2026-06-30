# Training Methodology

> The recipe that turns the architecture in [`architecture_spec.md`](architecture_spec.md)
> into trained weights. This is the design the [`train/train_ctrgcn.py`](../../train/train_ctrgcn.py)
> stub will implement (research-doc Phase 2). It is written to be **reproducible**:
> fixed seeds, frozen splits, a single config file, and a parity gate on export.
> **No pretrained backbone is used** — training is from random initialization on
> skeleton sequences only.

---

## 1. Data pipeline

```
raw videos ──pose──► COCO-17 skeletons ──window──► (C,T,V) clips ──► model
 (datasets)  (data/prepare.py)        (sliding, stride)        (reference_model.py)
```

1. **Pose extraction** ([`data/prepare.py`](../../data/prepare.py)). Each clip is
   run through the chosen pose backend (MoveNet Thunder / BlazePose) and
   normalized to the shared 17-joint COCO schema via `normalize_to_coco`. Per-joint
   confidence is **retained** — it is an input channel and the gate depends on it.
   Output is cached `.npz` skeleton sequences so training never re-runs pose.
2. **Windowing.** Sliding windows of `T = 32` frames (stride from the config) form
   the training clips. Label = 1 if the window overlaps an annotated fall, else 0.
3. **Normalization.** Coordinates are already in `[0,1]`; we additionally
   centre each clip on its hip midpoint and scale by torso length
   ([`features.torso_length`](../../core/eldercare/features.py)) so absolute
   position and subject size do not leak into the classifier.

### Datasets and the critical split discipline

| pool | datasets | role |
|---|---|---|
| **train / val** | UP-Fall + an NTU fall/ADL subset | model fitting (subject-wise split, **no subject leakage**) |
| **test (held out)** | UR Fall (URFD), Le2i | **zero-shot** — never seen in training |
| comparison | SisFall, FallAllD (wearable) | status-quo baseline contrast |

The headline metric is **cross-dataset zero-shot** F1 (train on UP-Fall/NTU, test
on URFD/Le2i with no fine-tuning). In-dataset accuracy is reported only to expose
the generalization gap (report **Fig. 8**), never as the primary claim.

---

## 2. Augmentation (skeleton-space, label-preserving)

Because we never touch pixels, augmentation is cheap and physically meaningful:

- **Random rotation / shear** in the image plane (± small angle) — viewpoint robustness.
- **Joint jitter** — Gaussian noise on `(x,y)` scaled by `(1 − score)` so low-confidence joints jitter more.
- **Confidence dropout** — randomly zero a joint's score for a span of frames to *simulate occlusion*; this is what teaches the confidence gate (§ architecture 2) to lean on temporal context. The 0 / 30 / 50 % dropout ablation (report **Fig. 9**) reuses this exact transform at test time.
- **Temporal crop / speed jitter** — resample the window slightly to decouple from frame rate.
- **Horizontal flip** with left/right joint relabeling (left_shoulder ↔ right_shoulder, …).

---

## 3. Optimization

Values mirror [`configs/ctrgcn.yaml`](../../configs/ctrgcn.yaml):

| hyperparameter | value | note |
|---|---|---|
| optimizer | AdamW | `weight_decay = 5e-4` |
| base learning rate | `1e-3` | cosine decay to 0 |
| warmup | 5 epochs linear | stabilizes the topology-refinement `α` |
| epochs | 80 | early-stop on val F1 |
| batch size | 64 | clips |
| loss | class-balanced focal / weighted cross-entropy | falls are rare → weight positives |
| seed | 42 | fixed; logged with every run |
| label smoothing | 0.1 | calibration for the downstream threshold |

**Why focal / class-balanced loss?** In continuous footage falls are a tiny
minority of windows. Plain cross-entropy would optimize accuracy by predicting
"not-fall" always. The loss up-weights the rare positive and the hard examples so
sensitivity is not sacrificed — directly serving the false-alarm-rate objective.

**Curriculum on `α`.** The CTR-GCN refinement scalar starts at 0 (pure ST-GCN
graph). Warmup lets the value layers settle before the topology is allowed to
move, which empirically avoids early instability.

---

## 4. Validation, calibration, threshold freezing

1. Train; select the checkpoint with the best **validation event-level F1**.
2. On the validation stream, sweep the alarm parameters
   (`τ`, `ema_alpha`, `k`, `m` — see [`alarm.py`](../../core/eldercare/alarm.py))
   to the chosen operating point: **≤ 1 false alert / hour** at the highest
   sensitivity (report **Fig. 6**).
3. **Freeze** all thresholds. They are never tuned on the test sets — that is
   what makes the cross-dataset number honest.

Both frame-level and event-level metrics ([`metrics.py`](../../core/eldercare/metrics.py))
are reported, plus ROC/AUC for the sweep and the time-to-alert p50/p95
distribution (report **Fig. 7**).

---

## 5. Export and the parity gate

`train/export_onnx.py` (Phase 2) exports the trained model to a single ONNX file
served identically on the edge node and the backend, then runs a **parity test**:

```
max | logit_torch − logit_onnxruntime |  <  1e-3
```

This guarantees the weights behave the same in PyTorch (training/eval) and in
ONNX Runtime (the Pi, the FastAPI backend, and onnxruntime-web in the dashboard).
INT8 post-training quantization (`--int8`) then takes the model to ~2.5 MB for the
device, with its own accuracy-vs-latency ablation.

---

## 6. Reproducibility checklist (Phase 6 "done when")

- [x] Architecture defined from scratch, no pretrained weights — `reference_model.py`
- [x] Parameter budget verified — `reference_model_numpy.py` (2.53 M / 2.66 M)
- [x] Config single-source-of-truth — `configs/ctrgcn.yaml`
- [x] Fixed seed (42), frozen subject-wise + cross-dataset splits
- [ ] Training loop wired (`train_ctrgcn.py`) — Phase 2
- [ ] ONNX export + parity test (`export_onnx.py`) — Phase 2
- [ ] Reproducibility pack: configs, seeds, ONNX weights, eval scripts — Phase 6

> A third party should be able to clone the repo, run `prepare.py` → `train_ctrgcn.py`
> → `export_onnx.py` → `eval/cross_dataset.py`, and reproduce the headline
> cross-dataset F1. Everything above the unchecked boxes already exists in the repo
> or in this report.

---

## 7. From this report to a running trainer (concrete next step)

The architecture is the missing body the trainer is waiting for. The stub's three
TODOs map onto existing, specified pieces:

```python
# train/train_ctrgcn.py  — Phase 2 wiring
from report.network.reference_model import build_model, ModelConfig

cfg   = ModelConfig()                       # from configs/ctrgcn.yaml
model = build_model(args.model, cfg)        # "stgcn" | "ctrgcn"  (no checkpoint)
# dataset: cached .npz from data/prepare.py, subject-wise split (no leakage)
# loss:    class-balanced focal;  opt: AdamW + cosine;  seed: 42
# loop:    train -> val F1 -> checkpoint best -> freeze thresholds -> export_onnx
```

That is the entire gap between the documented network in this report and a
trained, exportable, on-device fall classifier.
