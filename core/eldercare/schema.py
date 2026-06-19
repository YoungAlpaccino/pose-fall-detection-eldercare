"""Keypoint and WebSocket message contracts.

Defines the 17-joint COCO keypoint schema and the compact telemetry record that
crosses the wire. NOTE: raw frames are never part of any message here — that is
the privacy invariant (see docs/PRIVACY.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# COCO 17-joint order. All pose backends normalize to this schema.
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class EventType(str, Enum):
    """Discrete events emitted by the alarm logic."""

    NONE = "none"
    FALL = "fall"
    NODE_OFFLINE = "node_offline"


@dataclass(frozen=True)
class Keypoint:
    """A single 2D joint with confidence."""

    x: float
    y: float
    score: float


# Convenience name->index lookup for the COCO order above.
KEYPOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(COCO_KEYPOINTS)}


@dataclass(frozen=True)
class PoseFrame:
    """One frame of pose telemetry. The unit that crosses the WebSocket."""

    node_id: str
    ts: float
    keypoints: list[Keypoint]
    fall_score: float
    event: EventType

    def to_wire(self) -> dict:
        """Serialize to the compact JSON record sent over the WebSocket.

        Note: this contains keypoints, score, event, timestamp — and never any
        pixel data. That omission is the privacy invariant (docs/PRIVACY.md).
        """
        return {
            "node_id": self.node_id,
            "ts": self.ts,
            # flat [x, y, score, ...] triplets keep the payload tiny on the wire
            "keypoints": [v for kp in self.keypoints for v in (kp.x, kp.y, kp.score)],
            "fall_score": self.fall_score,
            "event": self.event.value,
        }

    @classmethod
    def from_wire(cls, msg: dict) -> PoseFrame:
        """Inverse of :meth:`to_wire` (used by the backend / TS parity tests)."""
        flat = msg["keypoints"]
        kps = [
            Keypoint(flat[i], flat[i + 1], flat[i + 2])
            for i in range(0, len(flat), 3)
        ]
        return cls(
            node_id=msg["node_id"],
            ts=msg["ts"],
            keypoints=kps,
            fall_score=msg["fall_score"],
            event=EventType(msg.get("event", "none")),
        )


class SchemaError(ValueError):
    """Raised when a pose frame violates the keypoint contract."""


def validate_pose_frame(frame: PoseFrame) -> None:
    """Validate a frame against the schema (joint count, score ranges).

    Raises :class:`SchemaError` on the first violation found.
    """
    n = len(frame.keypoints)
    if n != len(COCO_KEYPOINTS):
        raise SchemaError(
            f"expected {len(COCO_KEYPOINTS)} keypoints, got {n}"
        )
    for i, kp in enumerate(frame.keypoints):
        if not (0.0 <= kp.score <= 1.0):
            raise SchemaError(
                f"keypoint {i} ({COCO_KEYPOINTS[i]}) score {kp.score} out of [0,1]"
            )
    if not (0.0 <= frame.fall_score <= 1.0):
        raise SchemaError(f"fall_score {frame.fall_score} out of [0,1]")
