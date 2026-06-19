"""Synthetic skeleton generator for development without a camera or model.

Produces a deterministic, physically-plausible COCO-17 skeleton sequence: a
person stands and sways, then falls to the floor and stays down. This is the
"fake keypoints" source the ROADMAP's Phase 0 ("hello skeleton") calls for — it
lets the whole edge → backend → dashboard pipeline run end-to-end with no
dataset, no pose model, and no hardware.

Coordinates are normalized to [0, 1] with y growing downward (image convention).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from .schema import COCO_KEYPOINTS, KEYPOINT_INDEX, Keypoint

# Standing skeleton template: (x, y) per COCO joint, y up-ish in [0,1] image.
# Roughly a person centered at x=0.5, head near top, ankles near bottom.
_STANDING: dict[str, tuple[float, float]] = {
    "nose": (0.50, 0.12),
    "left_eye": (0.48, 0.10),
    "right_eye": (0.52, 0.10),
    "left_ear": (0.46, 0.11),
    "right_ear": (0.54, 0.11),
    "left_shoulder": (0.44, 0.24),
    "right_shoulder": (0.56, 0.24),
    "left_elbow": (0.42, 0.38),
    "right_elbow": (0.58, 0.38),
    "left_wrist": (0.41, 0.50),
    "right_wrist": (0.59, 0.50),
    "left_hip": (0.46, 0.52),
    "right_hip": (0.54, 0.52),
    "left_knee": (0.46, 0.72),
    "right_knee": (0.54, 0.72),
    "left_ankle": (0.46, 0.92),
    "right_ankle": (0.54, 0.92),
}


def _rotate(p: tuple[float, float], pivot: tuple[float, float], deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    cos, sin = math.cos(rad), math.sin(rad)
    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
    return (pivot[0] + dx * cos - dy * sin, pivot[1] + dx * sin + dy * cos)


def _pose_at(progress: float, t: float) -> list[Keypoint]:
    """Build one frame.

    ``progress`` in [0, 1] interpolates standing (0) -> fallen (1). ``t`` is the
    elapsed time, used only for a small idle sway while standing.
    """
    pivot = _STANDING["left_hip"]  # rotate the body about the hip toward the floor
    sway = 0.01 * math.sin(t * 1.5) * (1.0 - progress)  # only sways upright
    angle = 80.0 * progress  # rotate up to ~horizontal as the fall completes
    # As the body goes down, drop everything toward the floor (y -> ~0.9).
    drop = 0.30 * progress

    kps: list[Keypoint] = []
    for name in COCO_KEYPOINTS:
        x, y = _STANDING[name]
        x, y = _rotate((x, y), pivot, angle)
        x += sway
        y += drop
        # Confidence dips slightly mid-fall (motion blur) but stays usable.
        score = 0.95 - 0.25 * math.exp(-((progress - 0.5) ** 2) / 0.02)
        kps.append(Keypoint(x=round(x, 5), y=round(y, 5), score=round(score, 3)))
    return kps


def synthetic_stream(
    fps: int = 30,
    stand_s: float = 3.0,
    fall_s: float = 0.6,
    down_s: float = 3.0,
) -> Iterator[tuple[float, list[Keypoint]]]:
    """Yield ``(t, keypoints)`` for one stand → fall → on-the-floor episode.

    Timings default to a ~6.6 s clip with a sharp ~0.6 s fall in the middle —
    fast enough to trip the velocity cue, then a sustained low posture after.
    """
    dt = 1.0 / fps
    total = stand_s + fall_s + down_s
    t = 0.0
    while t < total:
        if t < stand_s:
            progress = 0.0
        elif t < stand_s + fall_s:
            # ease-in-out over the fall for a realistic velocity profile
            u = (t - stand_s) / fall_s
            progress = u * u * (3 - 2 * u)
        else:
            progress = 1.0
        yield t, _pose_at(progress, t)
        t += dt


# Sanity: template must cover exactly the COCO schema.
assert set(_STANDING) == set(KEYPOINT_INDEX), "synthetic template != COCO schema"
