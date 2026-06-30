# Diagrams & Schematics

> Mermaid + ASCII schematics for the manuscript and slides. Mermaid blocks render
> natively in GitHub, VS Code (with the Markdown Preview Mermaid extension), and
> most IEEE-friendly Markdown tooling. Each diagram pairs with a rendered PNG in
> [`../figures/`](../figures/) where a publication-grade raster is needed.

**Index**

1. [System architecture & the privacy boundary](#1-system-architecture--the-privacy-boundary)
2. [Edge pipeline (capture → alert)](#2-edge-pipeline-capture--alert)
3. [The frame-egress = 0 invariant](#3-the-frame-egress--0-invariant)
4. [Deep network (confidence-gated CTR-GCN)](#4-deep-network-confidence-gated-ctr-gcn)
5. [Spatial-temporal skeleton graph](#5-spatial-temporal-skeleton-graph)
6. [Confidence gating & temporal imputation](#6-confidence-gating--temporal-imputation)
7. [Alarm state machine (debounce + latch)](#7-alarm-state-machine-debounce--latch)
8. [Three surfaces & cross-language parity](#8-three-surfaces--cross-language-parity)
9. [Evaluation protocol](#9-evaluation-protocol)
10. [Build phases (research roadmap)](#10-build-phases-research-roadmap)

---

## 1. System architecture & the privacy boundary

> Rendered figure: [`figures/fig1_system_architecture.png`](../figures/fig1_system_architecture.png)

```mermaid
flowchart TB
    subgraph EDGE["EDGE NODE — Raspberry Pi 5 (privacy boundary: frame-egress = 0)"]
        direction LR
        CAP["Capture<br/>OpenCV / picamera2"] -->|frames| POSE["Pose estimator<br/>MoveNet / BlazePose"]
        POSE -->|"17 kpts (x,y,score)"| WIN["Temporal window<br/>T = 32–64"]
        WIN --> CLS["Fall classifier<br/>CTR-GCN (this report)"]
        CLS -->|"P(fall)"| ALM["Alarm logic<br/>EMA + k-of-m confirm"]
    end

    ALM -->|"WebSocket: { keypoints[], fall_score, event, ts }"| CORE

    subgraph CORE["core/ — single source of truth (Python, ported to TypeScript)"]
        direction LR
        C1["keypoint schema"] --- C2["EMA smoothing"] --- C3["geometric features"] --- C4["alarm confirm"] --- C5["metrics"]
    end

    CORE -->|reused by| BE["FastAPI backend<br/>WS hub · SQLModel · JWT · escalation"]
    CORE -->|ported to TS| FE["React dashboard<br/>skeleton overlay · onnxruntime-web · timeline"]
    BE <-->|live frames + events| FE

    classDef edge fill:#f3fbf6,stroke:#1f9d61,stroke-width:2px;
    classDef net fill:#fff5f3,stroke:#e2483b,stroke-width:2px;
    classDef shared fill:#eef3fb,stroke:#2f6fed,stroke-width:1.5px;
    class EDGE edge; class CLS net; class CORE,BE,FE shared;
```

**Reading it:** raw video exists only inside the green box. The only thing that
ever crosses the WebSocket is a compact skeleton-telemetry record — never a
pixel. The shared `core/` library is reused server-side and re-implemented in
TypeScript so the dashboard can re-score recorded streams for clinical audit.

---

## 2. Edge pipeline (capture → alert)

> The per-frame hot path that runs on the device. Mirrors
> [`edge/run.py`](../../edge/run.py).

```mermaid
flowchart LR
    A["frame t<br/>(stays on node)"] --> B["pose infer<br/>→ 17 keypoints"]
    B --> C["push to<br/>sliding window"]
    C --> D{"window full?"}
    D -- "no" --> A
    D -- "yes (every Nth frame)" --> E["confidence gate<br/>mask + impute"]
    E --> F["CTR-GCN<br/>→ P(fall)"]
    F --> G["EMA smooth"]
    G --> H{"P̄ ≥ τ for<br/>k of last m?"}
    H -- "no" --> A
    H -- "yes & not latched" --> I["emit FALL event"]
    I --> J["WebSocket publish<br/>telemetry only"]
    J --> A
```

> The classifier runs on a **stride** (every Nth frame) over the rolling window,
> not every frame — that is the compute saving that keeps the Pi at 25–30 FPS
> (research doc §7).

---

## 3. The frame-egress = 0 invariant

> Companion to [`docs/PRIVACY.md`](../../docs/PRIVACY.md). The privacy claim is
> *architectural*: there is no code path from a pixel buffer to the wire.

```mermaid
flowchart LR
    subgraph NODE["edge node process memory"]
        FR["raw frame buffer<br/>(numpy ndarray)"]:::danger
        KP["COCO keypoints<br/>(x, y, score)"]:::safe
        FR -->|"pose model<br/>(one-way: pixels → joints)"| KP
    end
    KP -->|"to_wire()"| WIRE["WebSocket payload<br/>{ node_id, ts, keypoints[], fall_score, event }"]:::safe
    FR -. "NO PATH<br/>(verified: static + runtime audit)" .-x WIRE

    classDef danger fill:#fff5f3,stroke:#e2483b,stroke-width:2px;
    classDef safe fill:#f3fbf6,stroke:#1f9d61,stroke-width:2px;
```

Enforcement (research doc §9, `docs/PRIVACY.md`):

- **Static audit** — assert no frame buffer reaches any serialization / WS path.
- **Runtime audit** — assert zero pixel bytes on the wire.
- **Reconstruction attack** — train a keypoints→image decoder; report
  `SSIM < 0.15`, `LPIPS > 0.6`, low re-ID accuracy → telemetry is non-recoverable.

The unit test `test_wire_carries_no_pixels` already asserts the wire record's
keys are exactly `{node_id, ts, keypoints, fall_score, event}`.

---

## 4. Deep network (confidence-gated CTR-GCN)

> Rendered figure: [`figures/fig2_network_architecture.png`](../figures/fig2_network_architecture.png) ·
> Full spec: [`../network/architecture_spec.md`](../network/architecture_spec.md)

```mermaid
flowchart LR
    X["input<br/>(N, 3, T, V=17)"] --> G["confidence gate<br/>mask + temporal impute"]:::net
    G --> BN["data BN"]
    BN --> B0["10 × spatial-temporal blocks<br/>64 → 128 → 256<br/>(stride-2 at blocks 3 & 5)"]
    B0 --> P["global avg pool (T,V)"]
    P --> FC["FC → 2"]:::net
    FC --> S["softmax → P(fall)"]:::net

    subgraph BLK["one block"]
        direction LR
        SG["spatial graph conv<br/>ST-GC or CTR-GC"] --> TB["BN+ReLU"] --> TC["multi-scale<br/>temporal conv"] --> R["+ residual"]
    end
    B0 -.-> BLK

    classDef net fill:#fff5f3,stroke:#e2483b,stroke-width:2px;
```

CTR-GC block internals (the topology-refinement novelty):

```
   x ──► shared topology  Â_k  (static skeleton graph, k = self/centripetal/centrifugal)
   x ──► channel refine   ΔA = Conv( tanh( φ1(x)_i − φ2(x)_j ) )
                          Â_refined = Â_k + α·ΔA      (α init 0 → starts at ST-GCN)
   out = Σ_k  Â_refined · (x · W_k)   ─►  multi-scale temporal conv  ─►  + residual
```

---

## 5. Spatial-temporal skeleton graph

> Rendered figure: [`figures/fig3_spatiotemporal_graph.png`](../figures/fig3_spatiotemporal_graph.png)

COCO-17 joints (indices match `core/eldercare/schema.py`):

```
                 0 nose
        1 eye ─  ●  ─ 2 eye
      3 ear ●         ● 4 ear
                 │
     5 ●━━━━━━━━━┻━━━━━━━━━● 6      shoulders
       ┃                   ┃
     7 ●                   ● 8      elbows
       ┃                   ┃
     9 ●                   ● 10     wrists
              11 ●━━━● 12          hips
                 ┃   ┃
             13 ●     ● 14          knees
                 ┃   ┃
             15 ●     ● 16          ankles
```

Each bone is a spatial edge. The graph is **replicated over the T frames** of the
window, and the *same joint in consecutive frames* is joined by a temporal edge —
this 2-D (joints × time) lattice is the domain the network convolves over.

```mermaid
flowchart LR
    subgraph t0["frame t"]
        A0((hip)) --- B0((knee)) --- C0((ankle))
    end
    subgraph t1["frame t+1"]
        A1((hip)) --- B1((knee)) --- C1((ankle))
    end
    A0 -. temporal .- A1
    B0 -. temporal .- B1
    C0 -. temporal .- C1
```

---

## 6. Confidence gating & temporal imputation

> Rendered figure: [`figures/fig4_confidence_gating.png`](../figures/fig4_confidence_gating.png) ·
> Code: [`ConfidenceGate`](../network/reference_model.py),
> [`SlidingWindow.as_tensor`](../../core/eldercare/temporal/__init__.py)

```mermaid
flowchart LR
    IN["joint v over T frames<br/>some frames score < 0.2"] --> M{"score ≥ 0.2?"}
    M -- "yes" --> KEEP["keep (x,y)"]
    M -- "no" --> IMP["impute (x,y):<br/>forward-fill from last<br/>confident frame, then<br/>back-fill leading gap"]
    KEEP --> REL["× learnable<br/>reliability σ(r_v)"]
    IMP --> REL
    REL --> OUT["gated tensor<br/>(shape preserved)"]
```

A blanket over the legs for 10 frames no longer collapses those joints to the
origin; they are carried from the nearest visible frame, and the network learns
(via the reliability scalar) how much to trust each joint.

---

## 7. Alarm state machine (debounce + latch)

> Rendered figure: [`figures/fig10_alarm_state_machine.png`](../figures/fig10_alarm_state_machine.png) ·
> Code: [`AlarmState`](../../core/eldercare/alarm.py)

```mermaid
stateDiagram-v2
    [*] --> ARMED
    ARMED --> COUNTING: EMA(P) ≥ τ
    COUNTING --> ARMED: EMA(P) < τ
    COUNTING --> LATCHED: k of last m frames ≥ τ  / emit FALL
    LATCHED --> ARMED: EMA(P) < τ  (re-arm; never re-fires while still down)
    LATCHED --> LATCHED: EMA(P) ≥ τ  (stay latched, silent)
```

This converts a noisy per-frame probability into **exactly one** debounced alert
per fall episode (unit tests `test_alarm_fires_once_on_sustained_fall`,
`test_alarm_debounces_single_spike`). The `(τ, ema_alpha, k, m)` tuple is the
single knob for the sensitivity ↔ false-alarm-rate trade-off.

---

## 8. Three surfaces & cross-language parity

```mermaid
flowchart TB
    CORE["core/eldercare (Python)<br/>schema · smoothing · features · alarm · metrics"]
    CORE --> EDGE["edge node<br/>imports core directly"]
    CORE --> BE["FastAPI backend<br/>imports core for validation"]
    CORE -->|"ported byte-for-byte"| TS["frontend/src/lib (TypeScript)<br/>same schema · smoothing · alarm"]
    GOLD["tests/golden/<br/>shared fixtures"] -. "parity asserted" .-> CORE
    GOLD -. "parity asserted" .-> TS

    classDef shared fill:#eef3fb,stroke:#2f6fed,stroke-width:1.5px;
    class CORE,EDGE,BE,TS shared;
```

The browser re-implements the critical hot path so a clinician can replay and
**re-score recorded skeleton streams without a round-trip** and audit *why* an
alert fired. Golden-vector fixtures keep Python and TypeScript byte-for-byte
identical.

---

## 9. Evaluation protocol

```mermaid
flowchart TB
    subgraph TRAIN["training pool (subject-wise split, no leakage)"]
        UP["UP-Fall"] --- NTU["NTU fall/ADL subset"]
    end
    TRAIN --> M["model<br/>(weights frozen, thresholds frozen)"]
    M --> XD["cross-dataset zero-shot<br/>URFD + Le2i (held out)"]:::head
    M --> FAR["continuous footage<br/>false-alarms / hour"]
    M --> OCC["occlusion ablation<br/>0 / 30 / 50% dropout"]
    M --> LAT["time-to-alert<br/>p50 / p95"]
    M --> PRIV["reconstruction attack<br/>SSIM / LPIPS / re-ID"]
    WEAR["SisFall / FallAllD<br/>(wearable status-quo)"] -.compare.- XD

    classDef head fill:#fff5f3,stroke:#e2483b,stroke-width:2px;
    class XD head;
```

- **Headline:** cross-dataset zero-shot F1 (report Fig. 8).
- **Deployment cost:** false-alarms/hour and the operating curve (Fig. 6).
- **Robustness:** occlusion ablation (Fig. 9), time-to-alert (Fig. 7).
- **Privacy:** reconstruction attack (non-recoverability).

---

## 10. Build phases (research roadmap)

> Source: [`ROADMAP.md`](../../ROADMAP.md). ✅ = implemented in the repo today.

```mermaid
flowchart LR
    P0["Phase 0<br/>scaffold +<br/>hello-skeleton ✅"] --> P1["Phase 1<br/>geometric<br/>baseline A ✅"]
    P1 --> P2["Phase 2<br/>deep model<br/>(this report)"]
    P2 --> P3["Phase 3<br/>real-time<br/>on Pi"]
    P3 --> P4["Phase 4<br/>three surfaces<br/>+ TS parity ✅(core)"]
    P4 --> P5["Phase 5<br/>rigorous<br/>eval"]
    P5 --> P6["Phase 6<br/>write-up<br/>(this report)"]

    classDef done fill:#f3fbf6,stroke:#1f9d61,stroke-width:2px;
    classDef now fill:#fff5f3,stroke:#e2483b,stroke-width:2px;
    class P0,P1 done; class P2,P6 now;
```

The MVP (Phases 0–1) runs end-to-end today; this report delivers the Phase-2
network design and the Phase-6 figures/write-up, leaving training and on-device
deployment as the clearly-scoped remaining work.
