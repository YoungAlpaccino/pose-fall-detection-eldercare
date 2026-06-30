# Results & Ablation Tables

> The empty/target cells are the Phase-5 evaluation deliverables. Each table has a
> fixed measurement procedure (manuscript §VI) and a paired figure. Real,
> reproduced-today values are marked **(real)**; pre-registered objectives are
> marked *(target)*. Fill the target cells from `eval/` once weights exist.

---

## Table I — Verified model capacity (real, from `reference_model_numpy.py`)

| model | parameters | FP32 size | INT8 size | < 25 MB budget |
|---|---:|---:|---:|:---:|
| ST-GCN (deep baseline) | 2,529,348 | ~10.1 MB | ~2.5 MB | ✓ **(real)** |
| CTR-GCN (proposed) | 2,660,095 | ~10.6 MB | ~2.7 MB | ✓ **(real)** |

Refinement overhead: **+130,747 params (+5.2 %)** over the static-graph baseline.

---

## Table II — Headline: cross-dataset zero-shot generalization (Fig. 8)

Train: UP-Fall + NTU (subject-wise). Test: URFD, Le2i (zero-shot, no fine-tune).

| method | in-dataset F1 | URFD F1 | Le2i F1 | mean cross F1 | gap |
|---|---:|---:|---:|---:|---:|
| Geometric heuristic (Baseline A) | _(target)_ | _(target)_ | _(target)_ | _(target)_ | _(target)_ |
| ST-GCN | _(target)_ | _(target)_ | _(target)_ | _(target)_ | _(target)_ |
| **CTR-GCN + gating (ours)** | _(target)_ | _(target)_ | _(target)_ | _(target)_ | **≤ −10** *(target)* |
| RGB 3D-CNN (non-deployable UB) | _(target)_ | _(target)_ | _(target)_ | _(target)_ | _(target)_ |

---

## Table III — Detection quality at the frozen operating point (Figs. 6, 7)

Operating point: ≤ 1 false alarm / hour, thresholds frozen on validation.

| metric | Baseline A | ST-GCN | CTR-GCN (ours) |
|---|---:|---:|---:|
| Sensitivity (recall on falls) | _(target)_ | _(target)_ | _(target)_ |
| Specificity | _(target)_ | _(target)_ | _(target)_ |
| Precision | _(target)_ | _(target)_ | _(target)_ |
| F1 (event-level) | _(target)_ | _(target)_ | _(target)_ |
| False alarms / hour | _(target)_ | _(target)_ | < 1 *(target)* |
| Time-to-alert p50 (s) | **0.57 (synthetic, real)** | _(target)_ | ≈ 0.8 *(target)* |
| Time-to-alert p95 (s) | _(target)_ | _(target)_ | ≈ 1.8 *(target)* |

> The **0.57 s** p50 is reproduced today by running Baseline A on the synthetic
> episode (`generate_figures.py`, Fig. 5). On-dataset latencies are Phase-5.

---

## Table IV — Ablation: confidence gating × occlusion (Fig. 9)

Sensitivity at simulated joint dropout. Gating is the proposed input layer.

| joint dropout | unmasked baseline | + confidence gating | Δ |
|---|---:|---:|---:|
| 0 % | _(target)_ | _(target)_ | _(target)_ |
| 30 % | _(target)_ | _(target)_ | **+6–9** *(target)* |
| 50 % | _(target)_ | _(target)_ | **+6–9** *(target)* |

---

## Table V — Ablation: design choices

| factor | settings | metric reported |
|---|---|---|
| window length `T` | 16 / 32 / 64 | F1, latency |
| classifier stride | 1 / 2 / 4 | F1, FPS |
| confirmation `k`-of-`m` | (3,5) / (5,8) / (7,10) | sensitivity, FAR/hr |
| pose backend | Lightning / Thunder / BlazePose | F1, FPS |
| precision | FP32 / INT8 | F1, latency, size |
| topology | static (ST-GCN) / refined (CTR-GCN) | cross-dataset F1 |

All *(target)* — to be filled by `eval/` (`--ablation`) in Phase 5.

---

## Table VI — Privacy evaluation (manuscript §VI-E, Fig. 1, §III)

| check | method | target | status |
|---|---|---|---|
| Static frame-egress audit | assert no frame buffer reaches WS/serialization | pass | `test_wire_carries_no_pixels` passes **(real)** |
| Runtime byte audit | assert zero pixel bytes on the wire | 0 bytes | _(target)_ |
| Reconstruction SSIM | keypoints→image decoder vs held-out frames | < 0.15 | _(target)_ |
| Reconstruction LPIPS | same | > 0.60 | _(target)_ |
| Re-identification accuracy | re-ID from reconstruction | ≈ chance | _(target)_ |

---

## Table VII — On-device latency budget (research doc §7)

| stage | p50 | p95 |
|---|---:|---:|
| capture + preprocess | 3 ms | 6 ms |
| pose (MoveNet Lightning) | 18 ms | 28 ms |
| temporal classifier (CTR-GCN, T=32) | 9 ms | 16 ms |
| alarm + WS publish | <1 ms | 2 ms |
| **end-to-end (Lightning path)** | **~30 ms** | **~50 ms** |

Throughput target: 25–30 FPS (Lightning). On-hardware measurement is Phase-3.

---

### How to fill these in (Phase 5)

```bash
python data/prepare.py    --dataset up-fall --src data/raw/up-fall --out data/up-fall
python train/train_ctrgcn.py --config configs/ctrgcn.yaml --model ctrgcn
python train/export_onnx.py  --ckpt runs/ctrgcn/best.pt --out models/ctrgcn.onnx --int8
python eval/cross_dataset.py --model models/ctrgcn.onnx --test urfd le2i   # Tables II, III
python eval/reconstruction_attack.py --keypoints data/urfd --frames data/raw/urfd  # Table VI
```
