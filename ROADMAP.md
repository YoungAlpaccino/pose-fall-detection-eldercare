# Roadmap

Build process converted from the research doc (§8). Each phase lists its tasks,
deliverable, and the "done when" criterion.

## Phase 0 — Scaffold

- [ ] Monorepo with `core/`, `backend/`, `frontend/`, `edge/`
- [ ] Define keypoint schema + WS message contract
- [ ] CI (pytest + Vitest)
- [ ] Pre-commit (ruff, mypy, eslint)
- [ ] End-to-end "hello skeleton": fake keypoints flow node → backend → dashboard

**Deliverable:** end-to-end "hello skeleton" — fake keypoints flow node→backend→dashboard.
**Done when:** a synthetic keypoint stream renders on the React canvas with < 100 ms glass-to-glass on LAN.

## Phase 1 — Baseline

- [ ] Geometric heuristic fall detector (Baseline A) in `core/`
- [ ] Evaluate on URFD + Le2i clips
- [ ] Produce metrics report (sensitivity / specificity / F1)

**Deliverable:** metrics report for the rules.
**Done when:** heuristic reaches a documented, reproducible F1 we can beat.

## Phase 2 — Core model

- [ ] Train ST-GCN on UP-Fall + NTU subset (PyTorch)
- [ ] Train CTR-GCN / PoseConv3D
- [ ] Export ONNX
- [ ] PyTorch vs ONNX parity test (logits within 1e-3)

**Deliverable:** ONNX classifier + parity test.
**Done when:** proposed model beats Baseline A on in-dataset F1 by a clear margin.

## Phase 3 — Real-time path

- [ ] Full edge pipeline on the Pi: capture → pose → window → classify → alarm → WS
- [ ] Latency / FPS report on real hardware

**Deliverable:** latency/FPS report on real hardware.
**Done when:** sustained ≥ 25 FPS and time-to-alert p95 < 2 s on-device.

## Phase 4 — Three surfaces

- [ ] Python `core/` reused by edge + backend
- [ ] TS port of smoothing/geometry/alarm
- [ ] Golden-vector parity fixtures (Python ↔ TS)
- [ ] onnxruntime-web replay in dashboard

**Deliverable:** cross-language parity test suite green.
**Done when:** TS and Python produce identical alerts on shared fixtures.

## Phase 5 — Rigorous eval

- [ ] Cross-dataset zero-shot (train UP-Fall/NTU → test URFD/Le2i)
- [ ] Occlusion ablation (0/30/50% joint dropout)
- [ ] False-alarm-rate over continuous footage
- [ ] Wearable comparison (SisFall / FallAllD)
- [ ] Reconstruction-attack privacy study (SSIM/LPIPS/reID)

**Deliverable:** full results + ablation tables.
**Done when:** every claim in §2 has a number behind it.

## Phase 6 — Write-up

- [ ] Paper + figures
- [ ] Reproducibility pack (configs, seeds, ONNX weights, eval scripts)
- [ ] Demo video

**Deliverable:** submission-ready manuscript.
**Done when:** a third party reproduces the headline metric from the repo.
