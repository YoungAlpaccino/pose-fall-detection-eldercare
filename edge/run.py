"""Edge node runner (Raspberry Pi 5).

Capture -> pose -> sliding window -> classify -> alarm -> WS publish.
Frames stay inside this process and are never transmitted (see docs/PRIVACY.md).

For the MVP / Phase-0 "hello skeleton" there is a ``--synthetic`` mode that
replaces the camera + ONNX pose model + ONNX classifier with a generated
stand->fall skeleton stream scored by the geometric heuristic baseline. This
lets the full edge -> backend -> dashboard path run with no hardware or weights:

    python edge/run.py --synthetic --backend ws://localhost:8000/ws/node

The real on-device path (camera + MoveNet + CTR-GCN) is wired the same way once
the ONNX models exist:

    python edge/run.py --source 0 --backend ws://hub.local:8000/ws/node \
        --pose models/movenet_lightning.onnx --classifier models/ctrgcn.onnx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

# Make the shared core/ library importable when running from a source checkout
# without installing the package (python edge/run.py ...).
sys.path.insert(0, __file__.rsplit("edge", 1)[0] + "core")

from eldercare.alarm import AlarmConfig, AlarmState  # noqa: E402
from eldercare.heuristic import HeuristicFallDetector  # noqa: E402
from eldercare.schema import EventType, PoseFrame  # noqa: E402
from eldercare.synthetic import synthetic_stream  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eldercare edge node")
    p.add_argument("--source", default="0", help="camera index or device path")
    p.add_argument("--backend", required=True, help="WS hub URL")
    p.add_argument("--pose", help="pose ONNX model (omit in --synthetic mode)")
    p.add_argument("--classifier", help="fall classifier ONNX (omit in --synthetic mode)")
    p.add_argument("--node-id", default="node-1")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="emit a generated stand->fall skeleton scored by the heuristic baseline",
    )
    p.add_argument(
        "--camera",
        action="store_true",
        help="real webcam via MediaPipe pose (CPU) + heuristic baseline — no ONNX",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="repeat the synthetic episode forever (demo mode)",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="(--camera) show a local skeleton-only preview window for debugging",
    )
    return p.parse_args()


async def run_synthetic(args: argparse.Namespace) -> None:
    """Stream a heuristic-scored synthetic skeleton episode to the WS hub."""
    import websockets

    detector = HeuristicFallDetector()
    alarm = AlarmState(AlarmConfig())
    dt = 1.0 / args.fps

    print(f"[edge] connecting to {args.backend} as {args.node_id} ...", flush=True)
    async with websockets.connect(args.backend) as ws:
        print("[edge] connected — streaming synthetic skeleton (no video leaves this node)", flush=True)
        while True:
            detector.reset()
            alarm = AlarmState(AlarmConfig())
            for t, kps in synthetic_stream(fps=args.fps):
                fall_prob = detector.score(kps, dt)
                event = alarm.update(fall_prob)
                frame = PoseFrame(
                    node_id=args.node_id,
                    ts=time.time(),
                    keypoints=kps,
                    fall_score=round(alarm.smoothed, 4),
                    event=event,
                )
                await ws.send(json.dumps(frame.to_wire()))
                if event is EventType.FALL:
                    print(f"[edge] *** FALL detected at t={t:.2f}s -> alert sent ***", flush=True)
                await asyncio.sleep(dt)
            if not args.loop:
                break
            await asyncio.sleep(1.0)
    print("[edge] episode complete.", flush=True)


async def run_camera(args: argparse.Namespace) -> None:
    """Capture from a real webcam, run MediaPipe pose, score, and publish.

    The captured frames live only inside this coroutine — only COCO keypoints,
    the fall score, and events are sent over the WebSocket (docs/PRIVACY.md).
    """
    import cv2
    import websockets

    from eldercare.pose import MediaPipePoseRunner

    src: object = int(args.source) if str(args.source).isdigit() else args.source
    # On Windows the default MSMF backend can hang on first read(); DirectShow
    # initializes fast and reliably for USB/built-in webcams.
    if sys.platform == "win32" and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera source {args.source!r}")

    pose = MediaPipePoseRunner()
    detector = HeuristicFallDetector()
    alarm = AlarmState(AlarmConfig())
    dt = 1.0 / args.fps
    loop = asyncio.get_running_loop()

    print(f"[edge] camera open ({args.source}); connecting to {args.backend} ...", flush=True)
    try:
        async with websockets.connect(args.backend) as ws:
            print("[edge] connected — webcam pose streaming (no video leaves this node)", flush=True)
            last = loop.time()
            while True:
                ok, frame = await loop.run_in_executor(None, cap.read)
                if not ok:
                    print("[edge] camera read failed; stopping.", flush=True)
                    break

                # pose + scoring are CPU-bound -> run off the event loop
                kps = await loop.run_in_executor(None, pose.infer, frame)
                now = loop.time()
                fall_prob = detector.score(kps, max(now - last, 1e-3))
                last = now
                event = alarm.update(fall_prob)

                await ws.send(
                    json.dumps(
                        PoseFrame(
                            node_id=args.node_id,
                            ts=time.time(),
                            keypoints=kps,
                            fall_score=round(alarm.smoothed, 4),
                            event=event,
                        ).to_wire()
                    )
                )
                if event is EventType.FALL:
                    print("[edge] *** FALL detected -> alert sent ***", flush=True)

                if args.preview:
                    _draw_preview(cv2, frame.shape, kps, alarm.smoothed)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # pace to target fps
                await asyncio.sleep(max(0.0, dt - (loop.time() - now)))
    finally:
        cap.release()
        pose.close()
        if args.preview:
            cv2.destroyAllWindows()
    print("[edge] camera session ended.", flush=True)


def _draw_preview(cv2, shape, kps, score) -> None:
    """Skeleton-only debug window (no camera pixels) — proves the privacy line."""
    import numpy as np

    from eldercare.pose import _BLAZEPOSE_TO_COCO  # noqa: F401  (ensures import ok)

    h, w = shape[0], shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    color = (80, 80, 255) if score >= 0.6 else (147, 211, 54)  # BGR
    edges = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    for a, b in edges:
        ka, kb = kps[a], kps[b]
        if ka.score < 0.2 or kb.score < 0.2:
            continue
        cv2.line(canvas, (int(ka.x * w), int(ka.y * h)),
                 (int(kb.x * w), int(kb.y * h)), color, 3)
    for k in kps:
        if k.score >= 0.2:
            cv2.circle(canvas, (int(k.x * w), int(k.y * h)), 4, color, -1)
    cv2.putText(canvas, f"fall_score={score:.2f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("edge node — skeleton only (no video leaves the node)", canvas)


def main() -> None:
    args = parse_args()

    if args.synthetic:
        asyncio.run(run_synthetic(args))
        return

    if args.camera:
        asyncio.run(run_camera(args))
        return

    # Real on-device path (Phase 3). Requires the ONNX models.
    if not args.pose or not args.classifier:
        raise SystemExit("real mode needs --pose and --classifier (or use --synthetic)")
    # TODO: open capture (picamera2 / cv2.VideoCapture)
    # TODO: loop:
    #   frame = capture()          # never leaves this process
    #   kpts  = MoveNetRunner.infer(frame)
    #   window.push(kpts)
    #   if window.is_full(): prob = classifier(window.as_tensor())
    #   event = alarm.update(prob)
    #   publish(PoseFrame(node_id, ts, kpts, prob, event))  # telemetry only
    raise NotImplementedError("real camera/ONNX path not wired yet — use --synthetic")


if __name__ == "__main__":
    main()
