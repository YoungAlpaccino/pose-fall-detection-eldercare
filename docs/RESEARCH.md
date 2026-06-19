# 09 — Privacy-Preserving Real-Time Human Pose & Fall Detection for Elder Care

> On-device skeleton-based fall detection where raw video never leaves the camera node — only pose telemetry and alerts cross the wire, robust to occlusion and tuned to keep nuisance alarms below one per hour.

## 1. Why this is cool

Falls are the leading cause of injury death for adults over 65, and the single biggest fear that keeps older people from living independently. Wearables (pendants, smart watches) get taken off, forgotten on the nightstand, or not pressed during the very confusion that follows a fall. A camera that *watches* is far more reliable — but a camera in a bedroom or bathroom is also the most invasive object you can put in someone's home.

This project resolves that tension head-on. The camera node never emits a single frame of raw video. The capture → pose-estimation → fall-classification pipeline runs entirely on a Raspberry Pi class device. What leaves the node is a stream of 2D skeleton keypoints (17–33 joints, x/y/confidence) and discrete alert events. Even if the network, the cloud account, or the caregiver's phone is compromised, there is no pixel data to leak. The skeleton is the privacy boundary, enforced by architecture rather than by policy.

It is also a genuinely hard real-time CV problem: pose estimation under partial occlusion (furniture, blankets, bathroom fixtures), distinguishing a fall from sitting-down-fast or lying-down-to-sleep, and doing it with a latency budget tight enough that an alert fires within a couple of seconds while a person is still on the floor. It exercises the full house stack — train in PyTorch, export ONNX, serve the same weights on the Pi edge node, the FastAPI backend, and a React caregiver dashboard.

## 2. Research novelty & contributions

- **Architectural privacy guarantee, measured, not asserted.** We define and enforce a *frame-egress = 0* property: a verifiable invariant that no raw or reconstructable pixel data leaves the node, with a reconstruction-attack study showing keypoint streams yield SSIM < 0.15 / LPIPS > 0.6 against held-out frames (i.e., the body identity and scene cannot be recovered from telemetry).
- **Cross-dataset generalization as the primary metric, not in-dataset accuracy.** Prior skeleton fall work reports >99% on UR Fall or Le2i by train/test on the same staged dataset. We train on UP-Fall + NTU subsets and evaluate *zero-shot* on UR Fall and Le2i, targeting a ≤10-point F1 drop where typical methods lose 25–40 points.
- **False-alarm-rate-per-hour as a first-class objective.** We optimize and report nuisance alarms over continuous unstaged footage (target < 1 false alert / hour / camera), reflecting deployment cost rather than balanced-accuracy on segmented clips.
- **Occlusion-robust temporal fusion.** A confidence-gated ST-GCN/CTR-GCN variant that masks low-confidence joints and imputes them from temporal context, improving sensitivity by a target +6–9 points under simulated 30–50% joint dropout versus an unmasked baseline.
- **Quantified time-to-alert distribution.** We report detection latency (impact → alert) as a p50/p95 distribution with a tunable confirmation window, making the sensitivity/latency/false-alarm trade-off explicit and reproducible.

## 3. System architecture

```
                          EDGE NODE (Raspberry Pi 5 + Camera)
        +-------------------------------------------------------------+
        |  capture (OpenCV/picamera2)                                 |
        |     |  frames stay HERE — never transmitted                |
        |     v                                                       |
        |  pose estimator  (MoveNet / BlazePose, ONNX Runtime)        |
        |     |  -> 17-33 keypoints (x,y,score) per frame            |
        |     v                                                       |
        |  temporal buffer (sliding window, T=32-64 frames)           |
        |     v                                                       |
        |  fall classifier (ST-GCN/CTR-GCN/PoseConv3D, ONNX)          |
        |     v                                                       |
        |  alarm logic (threshold + temporal smoothing + confirm)    |
        +----------------------|--------------------------------------+
                               | WebSocket: {keypoints[], event, ts}
                               |  (skeleton telemetry + alerts ONLY)
                               v
   +-------------------------------- core/ (Python library) ----------------------------+
   |  shared, ported Python<->TS: keypoint schema, smoothing/EMA, fall heuristics,      |
   |  geometry (aspect ratio, centroid velocity, joint angles), metrics                 |
   +-----------------------------------------------------------------------------------+
           ^                                                          ^
           | reused by                                               | ported to TS
           v                                                          v
   +------------------ FastAPI BACKEND ------------------+   +------- REACT FRONTEND -------+
   |  WS hub (fan-in nodes, fan-out dashboards)          |   |  caregiver dashboard          |
   |  SQLModel + SQLite: nodes, residents, events, ack   |   |  live skeleton canvas overlay |
   |  JWT auth (caregiver/admin roles)                   |   |  onnxruntime-web (replay/QA)  |
   |  alert escalation, audit log                        |   |  event timeline + ack/dismiss |
   +-----------------------------------------------------+   +-------------------------------+
```

The Python `core/` library is the single source of truth for the keypoint schema, temporal smoothing, geometric fall features, and the metric definitions. The edge node imports `core/` directly. The FastAPI backend imports it for server-side validation and event aggregation. The browser re-implements the *critical* hot path (smoothing, geometry, alarm confirmation) in TypeScript so the dashboard can replay and re-score recorded keypoint streams for QA without a round-trip, and so a clinician can audit *why* an alert fired. The keypoint schema and feature formulas are kept byte-for-byte equivalent across Python and TS via a shared golden-vector test fixture.

Raw frames live only inside the edge node process and are never written to disk by default. The WebSocket payload is a compact JSON (or MessagePack) record of keypoints, the classifier score, the discrete event, and a timestamp.

## 4. Tech stack

| Layer | Technology | Role |
|---|---|---|
| Training | PyTorch, PyTorch Lightning (optional) | Train pose-temporal models, export to ONNX |
| Pose estimation | MoveNet Thunder/Lightning, MediaPipe BlazePose | 2D keypoints on-device |
| Temporal model | ST-GCN, CTR-GCN, PoseConv3D (MMAction2 ref) | Skeleton action / fall classification |
| Inference runtime | ONNX Runtime (CPU/XNNPACK; CUDA on dev) | Same weights, edge + backend |
| CV / numerics | OpenCV, NumPy | Capture, preprocessing, geometry |
| Edge capture | picamera2 / OpenCV VideoCapture | Pi camera + USB cam |
| Backend | FastAPI, Uvicorn, WebSockets | WS hub, REST, escalation |
| Persistence | SQLModel + SQLite | Nodes, residents, events, acks, audit |
| Auth | JWT (python-jose / fastapi-users) | Caregiver/admin roles |
| Frontend | React 19 + TypeScript + Vite | Caregiver dashboard |
| Browser inference | onnxruntime-web (WASM/WebGPU) | Replay + QA re-scoring |
| Overlay | Canvas 2D / WebGL | Live skeleton rendering |
| Testing | pytest, Vitest, golden-vector fixtures | Cross-language parity, metrics |
| Edge HW | Raspberry Pi 5 (8GB), Camera Module 3 | Reference deployment |

## 5. Datasets

| Dataset | Modality | One-line note |
|---|---|---|
| UR Fall Detection (URFD) | RGB-D + accel | 70 sequences (30 falls), Kinect depth + IMU; classic skeleton-friendly benchmark. |
| Le2i Fall Detection | RGB | ~190 staged videos across home/office/coffee-room/lecture scenes; varied lighting. |
| UP-Fall Detection | RGB (2 views) + wearable + EEG | 17 subjects, 11 activities incl. 5 fall types; multimodal, large, well-labeled. |
| NTU RGB+D 60/120 | RGB-D + 3D skeleton | Large-scale action recognition; provides falling-down + ADL classes for pretraining. |
| FallAllD | Wearable (acc/gyr/mag/baro) | 26 subjects, 35 fall + ADL types; wearable comparison / cross-modal sanity. |
| SisFall | Wearable (accelerometer) | 38 subjects incl. elderly; standard wearable baseline to contrast camera method. |

Camera datasets (URFD, Le2i, UP-Fall, NTU) drive the vision pipeline; FallAllD and SisFall anchor a fair comparison against the wearable status quo the project aims to replace. UP-Fall + an NTU fall/ADL subset form the training pool; URFD and Le2i are held out entirely for cross-dataset zero-shot evaluation.

## 6. Models & algorithms

**Stage 1 — Pose estimation (per frame).**
- *Baseline:* MoveNet Lightning (fast, 17 keypoints) on the Pi.
- *Proposed:* MoveNet Thunder or BlazePose Full (higher accuracy under occlusion), with per-joint confidence retained downstream. Output normalized to a shared 17-joint schema (COCO order) regardless of backend.

**Stage 2 — Temporal fall classification (over a window of T frames).**
- *Baseline A (heuristic):* geometric rules — bounding-box aspect ratio flip, vertical centroid velocity spike, head-to-hip angle, sustained low posture — with temporal smoothing. Cheap, interpretable, surprisingly strong, and a fair "do we even need deep learning" control.
- *Baseline B (RGB):* a lightweight 3D-CNN (e.g., MobileNet3D / X3D-S) on the *node-local* RGB clip — used only as an internal upper-bound reference; never deployed because it would require frames, violating the privacy invariant.
- *Proposed:* skeleton-based **CTR-GCN** (channel-wise topology refinement) or **PoseConv3D** (heatmap volumes), with a *confidence-gated* input layer: low-score joints are masked and temporally imputed from the window. ST-GCN serves as the deep baseline to quantify the topology-refinement gain.

**Stage 3 — Alarm logic.**
- Per-frame fall probability → exponential moving average → threshold τ → **confirmation window** (probability must stay above τ for k of the last m frames). This converts a noisy frame-level score into a single debounced alert event and is the principal knob for the sensitivity vs false-alarm-rate trade-off. Thresholds calibrated on a validation stream, frozen before test.

## 7. Real-time / edge budget

Reference hardware: **Raspberry Pi 5 (8GB)**, Camera Module 3, ONNX Runtime CPU + XNNPACK, no accelerator. Dev/training on a single RTX-class GPU.

| Stage | p50 latency | p95 latency | Notes |
|---|---|---|---|
| Capture + preprocess | 3 ms | 6 ms | 640×480, BGR→RGB, resize/letterbox |
| Pose (MoveNet Lightning) | 18 ms | 28 ms | INT8/FP16, 17 kpts |
| Pose (MoveNet Thunder) | 38 ms | 55 ms | accuracy mode |
| Temporal classifier (CTR-GCN, T=32) | 9 ms | 16 ms | runs every N frames (stride) |
| Alarm logic + WS publish | <1 ms | 2 ms | JSON/MessagePack |
| **End-to-end (Lightning path)** | **~30 ms** | **~50 ms** | per-frame |

- **Throughput:** 25–30 FPS (Lightning), ~15–18 FPS (Thunder). Classifier runs on a stride (e.g., every 4th frame) over the rolling window to save compute.
- **Time-to-alert (impact → alert event):** target p50 ≈ 0.8 s, p95 ≈ 1.8 s with a ~0.7 s confirmation window.
- **Model sizes:** MoveNet Lightning ≈ 3–5 MB (INT8); Thunder ≈ 12 MB; CTR-GCN ≈ 1.5 M params / ~6 MB FP32, ~2 MB INT8. Total node footprint < 25 MB.
- **Memory:** < 400 MB RSS for the node process at 30 FPS.

## 8. Build process

**Phase 0 — Scaffold.** Monorepo with `core/`, `backend/`, `frontend/`, `edge/`. Keypoint schema, WS message contract, CI (pytest + Vitest), pre-commit (ruff, mypy, eslint). *Deliverable:* end-to-end "hello skeleton" — fake keypoints flow node→backend→dashboard. *Done when:* a synthetic keypoint stream renders on the React canvas with < 100 ms glass-to-glass on LAN.

**Phase 1 — Baseline.** Geometric heuristic fall detector (Baseline A) in `core/`, evaluated on URFD + Le2i clips. *Deliverable:* metrics report (sensitivity/specificity/F1) for the rules. *Done when:* heuristic reaches a documented, reproducible F1 we can beat.

**Phase 2 — Core model.** Train ST-GCN then CTR-GCN/PoseConv3D on UP-Fall + NTU subset in PyTorch; export ONNX. *Deliverable:* ONNX classifier + parity test (PyTorch vs ONNX logits within 1e-3). *Done when:* proposed model beats Baseline A on in-dataset F1 by a clear margin.

**Phase 3 — Real-time path.** Full edge pipeline on the Pi: capture → pose → window → classify → alarm → WS. *Deliverable:* latency/FPS report on real hardware. *Done when:* sustained ≥ 25 FPS and time-to-alert p95 < 2 s on-device.

**Phase 4 — Three surfaces.** Python `core/` reused by edge + backend; TS port of smoothing/geometry/alarm with golden-vector parity; onnxruntime-web replay in the dashboard. *Deliverable:* cross-language parity test suite green. *Done when:* TS and Python produce identical alerts on shared fixtures.

**Phase 5 — Rigorous eval.** Cross-dataset zero-shot (train UP-Fall/NTU → test URFD/Le2i), occlusion ablation, false-alarm-rate over continuous footage, wearable comparison (SisFall/FallAllD), reconstruction-attack privacy study. *Deliverable:* full results + ablation tables. *Done when:* every claim in §2 has a number behind it.

**Phase 6 — Write-up.** Paper, figures, reproducibility pack (configs, seeds, ONNX weights, eval scripts), demo video. *Deliverable:* submission-ready manuscript. *Done when:* a third party reproduces the headline metric from the repo.

## 9. Evaluation protocol & metrics

**Metrics.** Sensitivity (recall on falls), specificity, precision, F1, false-alarm rate per hour (over continuous unstaged streams), detection latency / time-to-alert (p50/p95), frame-level vs event-level scoring both reported. ROC/AUC for the threshold sweep.

**Splits.**
- *In-dataset:* UP-Fall official subject-wise split (no subject leakage between train/val/test).
- *Cross-dataset (headline):* train on UP-Fall + NTU fall/ADL subset; test zero-shot on URFD and Le2i with their full sets — no fine-tuning.
- *Continuous false-alarm test:* long unsegmented ADL footage scored end-to-end for alarms/hour.

**Baselines.** (1) Geometric heuristic; (2) ST-GCN; (3) RGB 3D-CNN internal upper bound (non-deployable); (4) wearable detectors on SisFall/FallAllD as the status-quo comparison.

**Ablations.** Confidence-gating on/off; window length T (16/32/64); classifier stride; confirmation-window k-of-m; pose backend (Lightning vs Thunder vs BlazePose); INT8 vs FP32 (accuracy ↔ latency); occlusion robustness at 0/30/50% simulated joint dropout.

**Privacy evaluation.** Frame-egress audit (static + runtime: assert zero pixel bytes on the wire). Reconstruction attack: train a decoder from keypoints→image and report SSIM/LPIPS/identity-reID accuracy to show non-recoverability.

## 10. Suggested repo structure

```
eldercare-fall/
├── core/                       # shared Python library (single source of truth)
│   ├── schema.py               # keypoint + WS message contracts
│   ├── pose/                   # ONNX pose runners, normalization
│   ├── temporal/               # sliding window, smoothing/EMA
│   ├── features.py             # geometric fall features (ported to TS)
│   ├── alarm.py                # threshold + confirmation logic
│   └── metrics.py              # sensitivity, FAR/hr, time-to-alert
├── edge/                       # Raspberry Pi node runner
│   ├── capture.py              # picamera2 / OpenCV
│   ├── pipeline.py             # capture->pose->classify->alarm->WS
│   └── config.yaml
├── backend/                    # FastAPI
│   ├── main.py                 # WS hub + REST
│   ├── models.py               # SQLModel: nodes, residents, events
│   ├── auth.py                 # JWT roles
│   └── escalation.py
├── frontend/                   # React 19 + Vite + TS
│   ├── src/lib/                # TS port of features/alarm (parity-tested)
│   ├── src/components/SkeletonCanvas.tsx
│   ├── src/components/EventTimeline.tsx
│   └── src/ort/                # onnxruntime-web replay
├── training/                   # PyTorch
│   ├── train_stgcn.py
│   ├── train_ctrgcn.py
│   ├── export_onnx.py
│   └── configs/
├── eval/                       # cross-dataset, ablations, privacy attack
│   ├── cross_dataset.py
│   ├── reconstruction_attack.py
│   └── far_continuous.py
├── tests/                      # pytest + golden vectors
│   └── golden/                 # shared Python<->TS fixtures
└── README.md
```

## 11. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Staged-dataset overfit → fails in real homes | High | High | Cross-dataset zero-shot as headline metric; continuous-footage FAR test. |
| High nuisance alarm rate erodes caregiver trust | High | High | Confirmation window + EMA; FAR/hr optimized and reported; per-resident calibration. |
| Occlusion (blankets, furniture) drops keypoints | High | Med | Confidence-gating + temporal imputation; multi-view stretch goal. |
| Pi can't sustain real-time | Med | Med | INT8/FP16 quantization, classifier stride, Lightning fallback; optional Hailo/Coral. |
| Privacy claim challenged by reviewers | Med | High | Static + runtime frame-egress audit; reconstruction-attack study with SSIM/LPIPS. |
| Bathroom/bedroom deployment ethics | Med | High | On-device only; opt-in per room; skeleton-only; resident & guardian consent flow. |
| Edge cases: pets, multiple people, children | Med | Med | Multi-person tracking + per-track classification; age/size-agnostic features. |
| Power/network outage misses an event | Low | High | Local event buffer with replay on reconnect; heartbeat + node-offline alerting. |

**Ethics.** Deployment requires informed consent from the resident (and guardian where appropriate), per-room opt-in, the ability to pause monitoring, and a clear data-handling notice. Because raw video never leaves the node, the dashboard cannot "spy" — it can only show skeletons and alerts, which is both a privacy feature and a dignity feature for the monitored person.

## 12. Stretch goals

- Multi-camera fusion within a home for occlusion-robust 3D pose and room-level tracking.
- On-device personalization: few-shot calibration to a resident's gait and ADL patterns to cut false alarms.
- Pre-fall / instability detection (sway, near-falls) for proactive intervention.
- Hardware acceleration (Hailo-8L, Coral Edge TPU) to run Thunder + CTR-GCN at 30 FPS headroom.
- Federated learning across nodes — improve the model without ever centralizing data, consistent with the privacy invariant.
- Sensor fusion with an optional wearable (FallAllD-style) for ground-truth-grade confirmation in high-risk patients.

## 13. Publication plan

- **Top venue:** *IEEE Journal of Biomedical and Health Informatics (JBHI)* — the natural home for a clinically framed, privacy-preserving health-monitoring system with rigorous cross-dataset and false-alarm evaluation.
- **Mid venue:** *IEEE Sensors Journal* / *IEEE Sensors Conference* — strong fit for the edge-node + on-device-inference angle and the wearable-vs-camera comparison.
- **Alternative journal:** *Elsevier Expert Systems with Applications* — for the algorithmic (confidence-gated temporal GCN) and applied-deployment contributions.
- **Conference → journal path:** publish the real-time edge system + cross-dataset result at a sensors/biomedical conference, then extend to a JBHI journal article with the full ablation suite, the reconstruction-attack privacy study, and a longitudinal in-home pilot.
- **Key figures:** (1) architecture with the frame-egress=0 boundary highlighted; (2) cross-dataset F1 bar chart vs in-dataset (the generalization gap); (3) sensitivity vs false-alarm-rate operating curve with the chosen operating point; (4) time-to-alert distribution; (5) occlusion-robustness ablation; (6) reconstruction-attack qualitative panel (input frame vs best decoder reconstruction from keypoints) with SSIM/LPIPS.
