# Eldercare Fall Detection

> Privacy-preserving real-time human pose & fall detection for elder care — raw video never leaves the camera node; only skeleton telemetry and alerts cross the wire.

## Overview

On-device, skeleton-based fall detection running on a Raspberry Pi class device.
The capture → pose-estimation → fall-classification pipeline runs entirely on the
edge node. What leaves the node is a stream of 2D keypoints (17–33 joints,
x/y/confidence) and discrete alert events — never a frame of raw video. The
skeleton is the privacy boundary, enforced by architecture rather than policy.

A shared Python `core/` library is the single source of truth for the keypoint
schema, temporal smoothing, geometric fall features, and metric definitions. It
is reused by the edge node and the FastAPI backend, and ported to TypeScript for
in-browser replay/QA in the React caregiver dashboard.

## Architecture

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

## Quickstart (MVP — "hello skeleton")

The MVP runs the full **edge → backend → dashboard** path with no camera, no
pose model, and no dataset: a synthetic stand→fall skeleton is scored by the
geometric heuristic baseline (Baseline A) and streamed live to the dashboard.

```bash
# 1. Python env + the shared core library (minimal MVP deps only)
python -m venv .venv && .venv/Scripts/activate      # Windows
# source .venv/bin/activate                          # macOS/Linux
pip install -e ".[dev]"

# 2. Tests (core: schema, features, heuristic, alarm, metrics, e2e)
pytest -q

# 3. Backend WS hub (port 8006)
uvicorn backend.app.main:app --port 8006

# 4. Dashboard (http://localhost:5173)
cd frontend && npm install && npm run dev

# 5a. Edge node — synthetic mode (no hardware needed); --loop to repeat
python edge/run.py --synthetic --loop --backend ws://localhost:8006/ws/node

# 5b. Edge node — REAL webcam via MediaPipe pose (CPU, no GPU, no ONNX)
#     One-time: download the pose model bundle (~6 MB)
python -c "import urllib.request as u; u.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task','models/pose_landmarker_lite.task')"
python edge/run.py --camera --source 0 --backend ws://localhost:8006/ws/node
#     add --preview for a local skeleton-only debug window (still no video on the wire)
```

Open http://localhost:5173 and watch the skeleton — in synthetic mode it falls
and the alert fires (~0.6 s time-to-alert); in `--camera` mode it tracks *you*
live. The Pi on-device path is the same command with
`--source 0 --pose <movenet.onnx> --classifier <ctrgcn.onnx>` once those models
exist (ROADMAP Phases 2-3).

> Pose runs on the CPU. The lite model does a few FPS on a laptop — fine for the
> demo; lower the capture resolution or use a faster machine for higher rates.

> Implemented: Phase 0 (end-to-end hello-skeleton) + Phase 1 (heuristic
> Baseline-A fall detector). Phases 2-6 (deep model, eval, write-up) remain.

## Links

- [Research doc](docs/RESEARCH.md)
- [Privacy invariant](docs/PRIVACY.md)
- [Roadmap](ROADMAP.md)
