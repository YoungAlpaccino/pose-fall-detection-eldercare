"""FastAPI backend: WebSocket hub + REST.

Fans in skeleton telemetry from edge nodes and fans it out to caregiver
dashboards. Raw frames never reach this service — only keypoints, scores, and
events.

MVP scope: an in-memory hub + event log. Persistence (SQLModel + SQLite), JWT
auth, and alert escalation are the next layer (see TODOs) — they slot in behind
the same WS/REST surface without changing the contract.

Endpoints
---------
GET  /api/health           liveness probe
GET  /api/events           recent fall/alert events (newest first)
WS   /ws/node              fan-in: edge nodes publish PoseFrame telemetry
WS   /ws/dashboard         fan-out: caregiver dashboards receive live frames
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Make the shared core/ library importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from eldercare.schema import EventType, PoseFrame, SchemaError, validate_pose_frame  # noqa: E402

app = FastAPI(title="Eldercare Fall Detection — Backend")

# Permissive CORS for the local dashboard dev server (Vite on :5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Hub:
    """In-memory fan-in/fan-out hub plus a bounded event log."""

    def __init__(self, event_log_size: int = 200) -> None:
        self.dashboards: set[WebSocket] = set()
        self.nodes: set[WebSocket] = set()
        self.events: deque[dict] = deque(maxlen=event_log_size)

    async def broadcast(self, message: str) -> None:
        """Send a raw JSON string to every connected dashboard."""
        dead: list[WebSocket] = []
        for ws in self.dashboards:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.dashboards.discard(ws)

    def record_event(self, frame: PoseFrame) -> None:
        self.events.appendleft(
            {
                "node_id": frame.node_id,
                "ts": frame.ts,
                "event": frame.event.value,
                "fall_score": frame.fall_score,
            }
        )


hub = Hub()


@app.get("/api/health")
def health() -> dict[str, object]:
    """Liveness probe + hub stats."""
    return {
        "status": "ok",
        "nodes": len(hub.nodes),
        "dashboards": len(hub.dashboards),
        "events": len(hub.events),
    }


@app.get("/api/events")
def list_events() -> list[dict]:
    """Recent fall/alert events, newest first."""
    return list(hub.events)


@app.websocket("/ws/node")
async def ws_node(ws: WebSocket) -> None:
    """Fan-in: an edge node publishes PoseFrame telemetry here."""
    await ws.accept()
    hub.nodes.add(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                frame = PoseFrame.from_wire(msg)
                validate_pose_frame(frame)
            except (json.JSONDecodeError, KeyError, SchemaError) as exc:
                await ws.send_text(json.dumps({"error": str(exc)}))
                continue

            if frame.event is not EventType.NONE:
                hub.record_event(frame)
            # Re-broadcast the validated, canonical wire form to dashboards.
            await hub.broadcast(json.dumps(frame.to_wire()))
    except WebSocketDisconnect:
        pass
    finally:
        hub.nodes.discard(ws)


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket) -> None:
    """Fan-out: a caregiver dashboard receives live frames + events."""
    await ws.accept()
    hub.dashboards.add(ws)
    # Replay recent events so a freshly-opened dashboard has context.
    await ws.send_text(json.dumps({"type": "event_log", "events": list(hub.events)}))
    try:
        while True:
            # Dashboards are read-only; keep the socket alive / drain pings.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.dashboards.discard(ws)


# TODO (next layer, unchanged WS/REST contract):
#   - SQLModel + SQLite: persist nodes, residents, events, acks
#   - JWT auth (caregiver/admin roles) on REST + WS handshake
#   - alert escalation (page on unacked FALL) + audit log
