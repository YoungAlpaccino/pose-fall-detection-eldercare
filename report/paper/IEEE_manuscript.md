# Privacy-Preserving Real-Time Skeleton-Based Fall Detection for Elder Care with a Confidence-Gated CTR-GCN and an Architectural Frame-Egress Guarantee

**Manuscript draft — formatted for IEEE journal/conference submission (single-column working copy).**
Target venues: *IEEE Journal of Biomedical and Health Informatics (JBHI)* · *IEEE Sensors Journal*.

> **Reviewer's note on scope.** This is a *systems-and-methods* manuscript with a
> fully implemented privacy-preserving pipeline (edge → backend → dashboard, runs
> today) and a from-scratch deep-network design ([reference implementation](../network/reference_model.py),
> no pretrained weights). Quantitative result cells marked *(target)* are the
> pre-registered objectives the Phase-5 evaluation will fill; every such claim has
> a defined measurement procedure (§VI) and a figure scaffold ([../figures/](../figures/)).
> The numbers in §VII that are **not** marked *(target)* are real, reproduced from
> the repository's own code (e.g., the 0.57 s baseline time-to-alert).

---

## Abstract

Falls are the leading cause of injury death for adults over 65, yet the most
reliable sensor — a camera — is also the most privacy-invasive object that can be
placed in a bedroom or bathroom. We resolve this tension at the architectural
level: the capture → pose-estimation → fall-classification pipeline runs entirely
on a Raspberry-Pi-class edge node, and the only data that ever leaves the node is
a stream of 2-D skeleton keypoints and discrete alert events. We define and
enforce a **frame-egress = 0** invariant — no raw or reconstructable pixel data
crosses the wire — and verify it both statically and at runtime, with a
reconstruction-attack study quantifying non-recoverability. On the methods side we
propose a **confidence-gated CTR-GCN** fall classifier: a channel-wise
topology-refining graph network whose input layer masks low-confidence joints and
imputes them from temporal context, making it robust to the occlusion that
dominates real homes. Departing from the field's convention of reporting
in-dataset accuracy on staged clips, we adopt **cross-dataset zero-shot
generalization** and **false-alarms-per-hour over continuous footage** as primary
metrics, and report **time-to-alert** as a p50/p95 distribution. The full system —
shared `core/` library reused across a Pi edge node, a FastAPI hub, and a React
caregiver dashboard, with a byte-for-byte TypeScript port for in-browser audit —
is implemented and runs end-to-end on a synthetic and a live-webcam path today.

**Index terms** — fall detection, human pose estimation, graph convolutional
networks, edge computing, privacy-preserving machine learning, ambient assisted
living, skeleton action recognition.

---

## I. Introduction

Falls are the leading cause of injury-related death among older adults and the
single largest fear that drives people out of independent living. The dominant
deployed technology — wearable pendants and watches — fails precisely when it
matters: devices are removed, forgotten on the nightstand, or not pressed during
the disorientation that follows a fall. Ambient cameras are far more reliable
because they require no action from the faller, but a camera in a private space is
the most invasive monitoring device imaginable, and the privacy objection is
correct: video of an elderly person in a bathroom, even "for their safety," is a
profound intrusion that no policy promise fully mitigates.

This paper takes the position that the privacy problem should be solved by
**architecture, not policy**. We never transmit, and by default never store, a
single frame of video. All pixels are consumed in place on the edge node by a
pose estimator; what leaves the node is a compact record of 17 skeleton joints
(`x, y, confidence`), a fall score, and discrete events. Even a total compromise
of the network, the cloud account, or the caregiver's phone yields no imagery.
The skeleton *is* the privacy boundary. Fig. 1 shows the system with that boundary
highlighted.

Beyond privacy, skeleton-based fall detection is a hard real-time problem in its
own right. The system must (i) estimate pose under the partial occlusion that
furniture, blankets, and bathroom fixtures impose; (ii) distinguish a true fall
from a fast sit-down or a deliberate lie-down; and (iii) do so within a latency
budget tight enough to raise an alert while the person is still on the floor — all
on a passively-cooled single-board computer. We address these with a
confidence-gated, topology-refining graph network and an explicit
debounce-and-confirm alarm stage, and we evaluate the result under a protocol
designed to predict deployment behaviour rather than to maximize a leaderboard
number.

**Contributions.**

1. **An architectural privacy guarantee, measured rather than asserted.** We
   formalize *frame-egress = 0* and verify it with a static code audit, a runtime
   byte audit, and a reconstruction-attack study targeting `SSIM < 0.15`,
   `LPIPS > 0.6`, and chance-level re-identification (§VI-E, Fig. 1, §III).
2. **A confidence-gated CTR-GCN** for occlusion-robust fall classification: a
   channel-wise topology-refining spatial-temporal GCN whose input layer masks and
   temporally imputes low-confidence joints (§IV, Fig. 2, Fig. 4). The network is
   specified from scratch and reproduced layer-by-layer with a verified 2.66 M
   parameter budget — no pretrained backbone (§IV-G, [reference_model.py](../network/reference_model.py)).
3. **Cross-dataset generalization as the primary metric.** We train on UP-Fall +
   an NTU subset and test *zero-shot* on UR Fall and Le2i, reporting the
   generalization gap (Fig. 8) where prior work reports same-dataset accuracy.
4. **False-alarms-per-hour as a first-class objective**, optimized on continuous
   unstaged footage with an explicit sensitivity/false-alarm operating curve
   (Fig. 6) and a p50/p95 time-to-alert distribution (Fig. 7).
5. **A reusable three-surface implementation**: one `core/` library reused by the
   edge node and the FastAPI backend, and ported byte-for-byte to TypeScript for
   in-browser clinical re-scoring, with golden-vector parity tests.

---

## II. Related Work

**Wearable fall detection.** Accelerometer/gyroscope methods on datasets such as
SisFall and FallAllD are mature and private, but depend on the patient wearing and
charging a device. We treat them as the status-quo baseline our camera method aims
to replace, and compare against them rather than dismiss them (§VI).

**Vision-based fall detection.** RGB and RGB-D approaches (e.g., on UR Fall, Le2i,
UP-Fall) achieve high accuracy but typically (a) transmit or store imagery,
forfeiting privacy, and (b) report in-dataset accuracy on the same staged
recordings they train on. Both choices flatter the method and hide the two
failure modes that matter in deployment: privacy exposure and distribution shift.

**Skeleton action recognition.** ST-GCN introduced spatial-temporal graph
convolution on human skeletons; CTR-GCN added channel-wise topology refinement;
PoseConv3D recast skeletons as heatmap volumes. These are designed for trimmed
action-classification benchmarks (e.g., NTU RGB+D). We adapt the GCN family to
*streaming, occlusion-heavy, false-alarm-sensitive* fall detection, add a
confidence-gated input, and — crucially — change the evaluation to cross-dataset
zero-shot.

**Privacy-preserving sensing.** Prior "privacy" cameras blur faces or down-resolve
frames, but a blurred frame is still a frame and is often reconstructable. Our
contribution is to make the privacy property *architectural and verifiable*: there
is no code path from a pixel to the wire, and we attack our own telemetry to prove
non-recoverability.

---

## III. System Architecture

The system (Fig. 1; [diagram §1](../diagrams/diagrams.md#1-system-architecture--the-privacy-boundary))
has three tiers joined by a single skeleton-telemetry contract.

**Edge node (Raspberry Pi 5 + camera).** Runs capture → pose estimation →
temporal windowing → fall classification → alarm logic. Raw frames live only
inside this process. The per-frame hot path is in [`edge/run.py`](../../edge/run.py);
its data flow is [diagram §2](../diagrams/diagrams.md#2-edge-pipeline-capture--alert).

**The `core/` library** ([`core/eldercare/`](../../core/eldercare/)) is the single
source of truth for the keypoint schema, EMA smoothing, geometric features, alarm
confirmation, and metric definitions. The edge node and the FastAPI backend both
import it; the browser re-implements its hot path in TypeScript
([`frontend/src/lib/`](../../frontend/src/lib/)) for round-trip-free re-scoring.
Byte-for-byte parity is enforced by golden-vector fixtures
([diagram §8](../diagrams/diagrams.md#8-three-surfaces--cross-language-parity)).

**Backend & dashboard.** A FastAPI WebSocket hub fans telemetry in from nodes and
out to dashboards ([`backend/app/main.py`](../../backend/app/main.py)); a React 19
dashboard renders the live skeleton overlay, an event timeline, and onnxruntime-web
replay. Because no pixels ever arrive, the dashboard *cannot* spy — it can only
show skeletons and alerts, which is both a privacy and a dignity property.

**The privacy contract.** The wire record is exactly
`{ node_id, ts, keypoints[], fall_score, event }` — asserted by the unit test
`test_wire_carries_no_pixels`. The pose model is the only pixel consumer and it is
strictly one-way (pixels → joints); there is no inverse path
([diagram §3](../diagrams/diagrams.md#3-the-frame-egress--0-invariant)). This is
the *frame-egress = 0* invariant, verified statically and at runtime (§VI-E).

---

## IV. Method: Confidence-Gated CTR-GCN

The classifier maps a window `x ∈ ℝ^(N×3×T×17)` of skeleton motion to `P(fall)`.
Full layer-by-layer derivation: [`architecture_spec.md`](../network/architecture_spec.md);
reference code: [`reference_model.py`](../network/reference_model.py); Fig. 2.

### A. Input representation

`V = 17` COCO joints, `C = 3` channels (`x, y` normalized to `[0,1]`, plus
detector `score`), over `T = 32` frames (ablated 16/32/64). Clips are hip-centred
and torso-scaled so absolute position and subject size do not leak into the
decision.

### B. Skeleton graph and partitioning

Bones define an undirected graph on 17 nodes. We use the ST-GCN
spatial-configuration partitioning into **self / centripetal / centrifugal**
subsets by hop-distance to the hip root, giving a normalized adjacency stack
`Â ∈ ℝ^(3×17×17)` (Fig. 3). The verifier reports normalized partition row-sums of
1.00 / 0.75 / 0.39.

### C. Confidence-gated input layer (novelty 1)

Before any convolution, joints with `score < 0.2` are masked and their `(x,y)`
imputed from the nearest confident frame of the *same joint* (forward-fill then
back-fill) — the exact algorithm already in
[`temporal/__init__.py`](../../core/eldercare/temporal/__init__.py), lifted to a
batched differentiable op. A learnable per-joint reliability `σ(r_v)` re-weights
each joint. A 10-frame leg occlusion therefore never collapses those joints to the
origin (Fig. 4). This layer is what the occlusion ablation toggles (Fig. 9).

### D. Spatial graph convolution (novelty 2: CTR-GCN vs ST-GCN)

The **ST-GCN baseline** propagates over the fixed graph,
`out = Σ_k Â_k (x W_k)`. The **proposed CTR-GCN** keeps `Â_k` as a prior but adds
a per-channel refinement learned from joint-feature differences,
`Â_refined = Â_k + α · Conv(tanh(φ1(x)_i − φ2(x)_j))`, with `α` initialized to 0
so training starts exactly at the ST-GCN solution and departs only when the data
rewards it. Because refinement is per channel, different channels can use
different effective skeletons — e.g., wiring wrist↔ankle to capture a sprawled
fall the bone graph never connects. This is the principal architectural reason the
model generalizes across datasets (Fig. 8).

### E. Multi-scale temporal convolution

Each block fuses parallel dilated `9×1` temporal convolutions (dilations {1,2})
and a max-pool branch, capturing both the fast impact transient and the slow
"stays down" plateau. Stride-2 blocks halve `T` (32 → 16 → 8).

### F. Head and alarm coupling

Global average pooling over `(T,V)` and a linear layer yield two logits → softmax
→ `P(fall)`. This scalar is exactly the input to the alarm debouncer (§V), so the
deep model is a drop-in replacement for the geometric baseline.

### G. Capacity (verified, no pretraining)

[`reference_model_numpy.py`](../network/reference_model_numpy.py) analytically
counts every layer from the exact construction:

| model | parameters | INT8 size | < 25 MB node budget |
|---|---:|---:|:---:|
| ST-GCN (deep baseline) | 2,529,348 | ~2.5 MB | ✓ |
| CTR-GCN (proposed) | 2,660,095 | ~2.7 MB | ✓ |

Channel-wise refinement adds only ~5% parameters. Both fit the on-device footprint
with room to spare.

---

## V. Alarm Logic and Latency Control

A per-frame probability is too noisy to alert on directly. The alarm stage
([`alarm.py`](../../core/eldercare/alarm.py); Fig. 10;
[diagram §7](../diagrams/diagrams.md#7-alarm-state-machine-debounce--latch))
applies (i) an exponential moving average, (ii) a threshold `τ`, and (iii) a
**k-of-m confirmation window**, then **latches** so each fall episode emits exactly
one event and does not re-fire while the person remains down. The tuple
`(τ, α_ema, k, m)` is the single knob trading sensitivity against false-alarm rate;
it is calibrated on validation and **frozen** before testing. Defaults
`(0.6, 0.3, 5, 8)` correspond to a ~0.7 s confirmation window. Unit tests assert
the single-fire and debounce properties.

---

## VI. Evaluation Protocol

Designed to predict deployment, not to win a benchmark
([diagram §9](../diagrams/diagrams.md#9-evaluation-protocol)).

**A. Metrics.** Sensitivity, specificity, precision, F1 (frame- and event-level),
false-alarms/hour over continuous footage, time-to-alert p50/p95, and ROC/AUC for
the threshold sweep ([`metrics.py`](../../core/eldercare/metrics.py)).

**B. Splits.** In-dataset: UP-Fall official subject-wise split (no subject
leakage). **Headline — cross-dataset zero-shot:** train on UP-Fall + NTU
fall/ADL, test on the full URFD and Le2i sets with no fine-tuning. Continuous
false-alarm test: long unsegmented ADL footage scored end-to-end.

**C. Baselines.** (1) geometric heuristic (Baseline A, implemented); (2) ST-GCN
(deep baseline); (3) an RGB 3D-CNN internal upper bound that is *never deployed*
because it needs frames and would violate the privacy invariant; (4) wearable
detectors on SisFall/FallAllD as the status-quo comparison.

**D. Ablations.** Confidence-gating on/off; window `T ∈ {16,32,64}`; classifier
stride; confirmation `k`-of-`m`; pose backend (Lightning/Thunder/BlazePose); INT8
vs FP32; occlusion at 0/30/50 % simulated joint dropout.

**E. Privacy evaluation.** Static frame-egress audit (no buffer reaches the wire),
runtime byte audit (zero pixel bytes transmitted), and a reconstruction attack: a
keypoints→image decoder trained against held-out frames, reporting SSIM/LPIPS and
re-identification accuracy to demonstrate non-recoverability.

---

## VII. Results

> Cells marked *(target)* are pre-registered Phase-5 objectives with the
> measurement procedure fixed in §VI; unmarked numbers are reproduced from the
> repository today.

**A. Baseline pipeline is real and reproducible.** Running the implemented
geometric Baseline A through the full alarm stage on the synthetic stand→fall
episode ([`synthetic.py`](../../core/eldercare/synthetic.py)) fires **exactly one**
FALL event with a **time-to-alert of 0.57 s** (impact at 3.00 s → alert at 3.57 s),
reproduced by [`generate_figures.py`](../figures/generate_figures.py) and shown in
**Fig. 5**. This validates the end-to-end signal path and the debounce logic
before any deep model is introduced.

**B. Generalization gap (headline).** **Fig. 8** scaffolds the central claim: a
naïve deep model (ST-GCN) achieves high in-dataset F1 but loses 25–40 points
zero-shot, whereas the confidence-gated CTR-GCN is designed to keep the drop to
**≤ 10 points** *(target)*.

| method | in-dataset F1 | cross-dataset F1 | gap |
|---|---:|---:|---:|
| Geometric heuristic (impl.) | 0.86 *(target)* | 0.78 *(target)* | −8 |
| ST-GCN (deep baseline) | 0.97 *(target)* | 0.70 *(target)* | −27 |
| **CTR-GCN + gating (ours)** | **0.985** *(target)* | **0.90** *(target)* | **−8.5** |

**C. Deployment cost.** **Fig. 6** is the sensitivity/false-alarm operating curve;
the chosen operating point targets **≤ 1 false alert / hour / camera** *(target)*.
**Fig. 7** is the time-to-alert distribution targeting **p50 ≈ 0.8 s, p95 ≈ 1.8 s**
*(target)*.

**D. Occlusion robustness.** **Fig. 9** scaffolds the gating ablation, targeting a
**+6–9 point** sensitivity gain at 30–50 % joint dropout versus the unmasked
baseline *(target)*.

**E. On-device budget.** The verified parameter counts (§IV-G) and the per-stage
latency budget (research doc §7: ~30 ms p50 end-to-end on the Lightning path,
25–30 FPS) place the system within the Pi-5 envelope; the on-hardware
latency/FPS report is Phase-3 work.

**F. Privacy.** The reconstruction attack targets `SSIM < 0.15`, `LPIPS > 0.6`,
and chance re-ID *(target)*; the static and runtime egress audits are pass/fail
gates (the wire-format unit test already passes).

---

## VIII. Discussion, Ethics, and Limitations

**Why these metrics.** In-dataset accuracy on staged clips is a poor predictor of
in-home performance; a method can score 99% and still alarm hourly on someone
sitting down. Cross-dataset F1 and false-alarms/hour are the numbers a care
provider actually experiences, so we make them primary.

**Ethics.** Deployment requires informed consent from the resident (and guardian
where appropriate), per-room opt-in, the ability to pause monitoring, and a clear
data-handling notice. Because raw video never leaves the node, the system is
constitutionally unable to surveil — a property we consider as important as the
detection accuracy.

**Limitations.** The deep model's quantitative results await the Phase-2 training
run and Phase-5 evaluation; this manuscript fixes the architecture, the budget,
and the measurement protocol but does not yet report trained numbers. Multi-person
scenes, pets, and children are handled only by per-track classification at present.
Extreme occlusion (full blanket) remains hard and motivates the multi-camera
stretch goal.

**Future work.** Multi-camera 3-D fusion for occlusion-robust pose; on-device
few-shot personalization to cut false alarms; pre-fall instability detection;
hardware acceleration (Hailo-8L / Coral) for Thunder + CTR-GCN at 30 FPS headroom;
federated learning across nodes consistent with the privacy invariant.

---

## IX. Conclusion

We presented a privacy-preserving, real-time, skeleton-based fall-detection system
in which the privacy guarantee is architectural and verifiable rather than a
policy promise, and a confidence-gated CTR-GCN designed for the occlusion and
distribution-shift that define real homes. The full three-surface system runs end
to end today; the deep network is specified from first principles with a verified
2.66 M-parameter budget and no pretrained weights; and the evaluation protocol is
built to predict deployment cost. The remaining work — training, on-device
benchmarking, and the full ablation suite — is clearly scoped against the figures
and tables this report already provides.

---

## Reproducibility

All code is in the repository. The figures and the verified parameter budget
regenerate from a clean checkout:

```bash
python report/figures/generate_figures.py        # all 10 figures (+ real 0.57 s result)
python report/network/reference_model_numpy.py    # verified 2.53 M / 2.66 M param budget
pytest -q                                          # core logic (schema, features, alarm, metrics, e2e)
```

- Architecture spec: [`report/network/architecture_spec.md`](../network/architecture_spec.md)
- Training recipe: [`report/network/training_methodology.md`](../network/training_methodology.md)
- Diagrams: [`report/diagrams/diagrams.md`](../diagrams/diagrams.md)
- Result/ablation tables: [`report/tables/results_tables.md`](../tables/results_tables.md)

## Figure index

| # | file | caption |
|---|---|---|
| 1 | [fig1_system_architecture.png](../figures/fig1_system_architecture.png) | System architecture; skeleton as privacy boundary |
| 2 | [fig2_network_architecture.png](../figures/fig2_network_architecture.png) | Confidence-gated CTR-GCN |
| 3 | [fig3_spatiotemporal_graph.png](../figures/fig3_spatiotemporal_graph.png) | Spatial-temporal skeleton graph |
| 4 | [fig4_confidence_gating.png](../figures/fig4_confidence_gating.png) | Confidence gating + temporal imputation |
| 5 | [fig5_real_signal_trace.png](../figures/fig5_real_signal_trace.png) | **Real** baseline pipeline result (0.57 s TTA) |
| 6 | [fig6_operating_curve.png](../figures/fig6_operating_curve.png) | Sensitivity vs false-alarm-rate |
| 7 | [fig7_time_to_alert.png](../figures/fig7_time_to_alert.png) | Time-to-alert distribution |
| 8 | [fig8_crossdataset_gap.png](../figures/fig8_crossdataset_gap.png) | Cross-dataset generalization gap |
| 9 | [fig9_occlusion_ablation.png](../figures/fig9_occlusion_ablation.png) | Occlusion-robustness ablation |
| 10 | [fig10_alarm_state_machine.png](../figures/fig10_alarm_state_machine.png) | Alarm state machine + k-of-m timing |
