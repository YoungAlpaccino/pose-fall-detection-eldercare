"""Geometric fall features (ported byte-for-byte to TypeScript).

Cheap, interpretable signals that feed both the heuristic baseline (Baseline A)
and the deep classifier's auxiliary inputs. Parity with the TS port is enforced
via shared golden vectors (tests/golden/).

All functions ignore joints below ``MIN_SCORE`` so a single dropped keypoint
cannot distort a feature (confidence gating).
"""

from __future__ import annotations

import math

from .schema import KEYPOINT_INDEX, Keypoint

MIN_SCORE = 0.2  # joints below this confidence are excluded from geometry


def _confident(keypoints: list[Keypoint]) -> list[Keypoint]:
    return [kp for kp in keypoints if kp.score >= MIN_SCORE]


def _point(keypoints: list[Keypoint], name: str) -> Keypoint | None:
    kp = keypoints[KEYPOINT_INDEX[name]]
    return kp if kp.score >= MIN_SCORE else None


def _midpoint(a: Keypoint | None, b: Keypoint | None) -> tuple[float, float] | None:
    """Midpoint of two joints, falling back to whichever one is confident."""
    if a is not None and b is not None:
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
    if a is not None:
        return (a.x, a.y)
    if b is not None:
        return (b.x, b.y)
    return None


def bounding_box_aspect_ratio(keypoints: list[Keypoint]) -> float:
    """Width / height of the keypoint bounding box. Flips when a person falls.

    Standing people are tall (ratio < 1); a fallen body is wide (ratio > 1).
    Returns 0.0 when too few joints are confident to form a box.
    """
    pts = _confident(keypoints)
    if len(pts) < 2:
        return 0.0
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if height <= 1e-6:
        return 0.0
    return width / height


def centroid(keypoints: list[Keypoint]) -> tuple[float, float] | None:
    """Mean (x, y) over confident joints, or None if none are confident."""
    pts = _confident(keypoints)
    if not pts:
        return None
    return (
        sum(p.x for p in pts) / len(pts),
        sum(p.y for p in pts) / len(pts),
    )


def centroid_vertical_velocity(
    prev: list[Keypoint], curr: list[Keypoint], dt: float
) -> float:
    """Vertical velocity of the body centroid. Spikes during a fall.

    Positive means moving *down* the image (y grows downward), in
    normalized-height units per second. Returns 0.0 if either centroid is
    undefined or ``dt`` is non-positive.
    """
    if dt <= 0.0:
        return 0.0
    c_prev = centroid(prev)
    c_curr = centroid(curr)
    if c_prev is None or c_curr is None:
        return 0.0
    return (c_curr[1] - c_prev[1]) / dt


def head_to_hip_angle(keypoints: list[Keypoint]) -> float:
    """Angle (degrees) of the hip→head vector vs vertical. ~90° => lying down.

    0° is fully upright (head directly above hips); 90° is horizontal. Returns
    0.0 when head or hips cannot be located confidently.
    """
    nose = _point(keypoints, "nose")
    hip = _midpoint(
        keypoints[KEYPOINT_INDEX["left_hip"]] if keypoints[KEYPOINT_INDEX["left_hip"]].score >= MIN_SCORE else None,
        keypoints[KEYPOINT_INDEX["right_hip"]] if keypoints[KEYPOINT_INDEX["right_hip"]].score >= MIN_SCORE else None,
    )
    if nose is None or hip is None:
        return 0.0
    dx = nose.x - hip[0]
    dy = nose.y - hip[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    # angle of the vector from vertical: 0 when pointing straight up/down.
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def torso_length(keypoints: list[Keypoint]) -> float:
    """Vertical extent from shoulders to hips. Used to scale velocities."""
    sh = _midpoint(
        keypoints[KEYPOINT_INDEX["left_shoulder"]] if keypoints[KEYPOINT_INDEX["left_shoulder"]].score >= MIN_SCORE else None,
        keypoints[KEYPOINT_INDEX["right_shoulder"]] if keypoints[KEYPOINT_INDEX["right_shoulder"]].score >= MIN_SCORE else None,
    )
    hip = _midpoint(
        keypoints[KEYPOINT_INDEX["left_hip"]] if keypoints[KEYPOINT_INDEX["left_hip"]].score >= MIN_SCORE else None,
        keypoints[KEYPOINT_INDEX["right_hip"]] if keypoints[KEYPOINT_INDEX["right_hip"]].score >= MIN_SCORE else None,
    )
    if sh is None or hip is None:
        return 0.0
    return math.hypot(sh[0] - hip[0], sh[1] - hip[1])


def sustained_low_posture(keypoints: list[Keypoint]) -> bool:
    """Whether the body is currently in a low/horizontal posture.

    True when the bounding box is wider than tall *and* the torso is tilted
    toward horizontal — the static signature of a person on the ground.
    (The "sustained" part is enforced by the alarm's confirmation window.)
    """
    return (
        bounding_box_aspect_ratio(keypoints) > 1.0
        and head_to_hip_angle(keypoints) > 55.0
    )
