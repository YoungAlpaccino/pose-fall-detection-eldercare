# Report — Privacy-Preserving Skeleton-Based Fall Detection for Elder Care

> A self-contained academic dossier for the IEEE write-up and the professor
> showcase. It documents the system, **specifies the deep training network from
> first principles** (no pretrained weights anywhere), and provides
> publication-quality figures, schematics, and result/ablation scaffolds.

This folder is the **Phase-6 "write-up" deliverable** of the project roadmap. It
turns the working MVP (a real edge → backend → dashboard pipeline) and the
designed-but-unimplemented deep model into a reviewable research package.

---

## What's inside

```
report/
├── README.md                       ← you are here (start + reading order)
├── paper/
│   └── IEEE_manuscript.md          ← the full IEEE-style article (the centerpiece)
├── network/                        ← THE TRAINING NETWORK (from scratch)
│   ├── reference_model.py          ← PyTorch ST-GCN + CTR-GCN, confidence gate, NO pretrained weights
│   ├── reference_model_numpy.py    ← torch-free verifier: shape flow + parameter budget (runs anywhere)
│   ├── architecture_spec.md        ← layer-by-layer spec + the graph-conv math
│   └── training_methodology.md     ← data, augmentation, optimizer, splits, export, parity gate
├── diagrams/
│   └── diagrams.md                 ← 10 Mermaid + ASCII schematics
├── figures/
│   ├── generate_figures.py         ← regenerates all 10 figures (one uses the REAL core pipeline)
│   └── fig1..fig10 *.png           ← 300-dpi publication rasters
└── tables/
    └── results_tables.md           ← result + ablation tables (real values + Phase-5 targets)
```

---

## Suggested reading order (for a reviewer / professor)

1. **[paper/IEEE_manuscript.md](paper/IEEE_manuscript.md)** — the full story:
   abstract, contributions, architecture, method, protocol, results.
2. **[network/architecture_spec.md](network/architecture_spec.md)** — how the deep
   classifier is built, joint-by-joint and layer-by-layer.
3. **[network/reference_model.py](network/reference_model.py)** — the actual,
   from-scratch network code (read alongside the spec).
4. **[network/training_methodology.md](network/training_methodology.md)** — exactly
   how it would be trained and exported, reproducibly.
5. **[diagrams/diagrams.md](diagrams/diagrams.md)** — all schematics in one place.
6. **[tables/results_tables.md](tables/results_tables.md)** — what's measured today
   vs the pre-registered Phase-5 targets.

---

## Reproduce everything from a clean checkout

```bash
# 1) the report's figures (incl. a REAL run of the project's own pipeline)
python report/figures/generate_figures.py
#    -> writes report/figures/fig1..fig10 and prints the real baseline result:
#       "time-to-alert = 0.57 s (impact 3.0s -> alert 3.57s)"

# 2) verify the network's shape flow + parameter budget WITHOUT PyTorch
python report/network/reference_model_numpy.py
#    -> STGCN 2,529,348 params | CTRGCN 2,660,095 params, all shapes consistent

# 3) run the real network forward pass (only if torch is installed)
python report/network/reference_model.py
#    -> STGCN/CTRGCN  in (4,3,32,17) -> logits (4,2)   params 2.53M / 2.66M

# 4) the implemented core logic the report builds on
pytest -q
```

> The MVP venv ships without PyTorch, so step (3) is optional — step (2) verifies
> the same architecture (shapes + parameter count) with NumPy only.

---

## The integrity rules this report follows

- **No pretrained models.** Every network parameter is freshly initialized
  ([`reference_model.py`](network/reference_model.py)); the figures use synthetic
  or illustrative data, or a **real** run of the repo's own heuristic pipeline.
  The deep network is drawn from its *definition*, never a checkpoint.
- **Real vs target is always labelled.** Numbers reproduced today (the 0.57 s
  time-to-alert, the 2.53 M / 2.66 M parameter counts, the passing privacy
  unit-test) are marked **(real)**; pre-registered Phase-5 objectives are marked
  *(target)* with their measurement procedure fixed in the manuscript §VI.
- **Everything traces to the code.** Each claim links to the implementing file in
  the repo (`core/`, `edge/`, `backend/`, `train/`, `eval/`).

---

## How this maps to the project roadmap

| roadmap phase | status | where in this report |
|---|---|---|
| Phase 0 — scaffold + hello-skeleton | ✅ implemented | manuscript §III, diagrams §1–2 |
| Phase 1 — geometric Baseline A | ✅ implemented | **Fig. 5 (real result)**, §VII-A |
| Phase 2 — deep model (ST-GCN/CTR-GCN) | 🎯 **designed here** | `network/`, Figs. 2–4, §IV |
| Phase 3 — real-time on Pi | ⏳ scoped | §V, §VII-E, Table VII |
| Phase 4 — three surfaces + TS parity | ✅ core implemented | diagrams §8 |
| Phase 5 — rigorous eval | ⏳ scaffolded | Figs. 6–9, Tables II–VI |
| Phase 6 — write-up | ✅ **this report** | everything |

The contribution of this report is to make Phases 2 and 6 concrete: a complete,
verifiable network design and a submission-shaped manuscript, so the remaining
training and evaluation work is unambiguous and ready to execute.
