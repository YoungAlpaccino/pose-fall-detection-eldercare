"""Baseline A: a geometric heuristic fall detector.

Combines the interpretable features (aspect ratio, downward centroid velocity,
torso tilt) into a single per-frame fall probability in [0, 1]. This is the
rules-based baseline the ROADMAP's Phase 1 calls for — it needs no trained
weights, so it is what powers the MVP edge node out of the box.

The deep classifier (CTR-GCN, Phase 2) is a drop-in replacement: it produces the
same scalar ``fall_score`` consumed by the alarm debouncer.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import features
from .schema import Keypoint


@dataclass
class HeuristicConfig:
    """Weights/thresholds for the geometric fall score (tune before freezing)."""

    # Aspect ratio: <1 standing, >1 lying. Map [ar_lo, ar_hi] -> [0, 1].
    ar_lo: float = 0.8
    ar_hi: float = 1.6
    # Downward centroid velocity (normalized-height units/s) that reads as a fall.
    vel_hi: float = 1.2
    # Torso tilt (deg from vertical): >tilt_hi reads as horizontal.
    tilt_lo: float = 35.0
    tilt_hi: float = 75.0
    # Blend weights for the three cues (need not sum to 1; score is clamped).
    w_posture: float = 0.6   # aspect-ratio + tilt (the person is *down*)
    w_velocity: float = 0.4  # they got there *fast* (a fall, not lying down)


def _ramp(x: float, lo: float, hi: float) -> float:
    """Clamped linear ramp: 0 below lo, 1 above hi."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


class HeuristicFallDetector:
    """Stateful Baseline A scorer over a stream of keypoint frames.

    Keeps the previous frame so it can estimate downward centroid velocity.
    """

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()
        self._prev: list[Keypoint] | None = None

    def reset(self) -> None:
        self._prev = None

    def score(self, keypoints: list[Keypoint], dt: float) -> float:
        """Return the fall probability in [0, 1] for the current frame."""
        c = self.config

        ar = features.bounding_box_aspect_ratio(keypoints)
        tilt = features.head_to_hip_angle(keypoints)
        posture = max(
            _ramp(ar, c.ar_lo, c.ar_hi),
            _ramp(tilt, c.tilt_lo, c.tilt_hi),
        )

        if self._prev is None:
            velocity = 0.0
        else:
            vy = features.centroid_vertical_velocity(self._prev, keypoints, dt)
            velocity = _ramp(vy, 0.0, c.vel_hi)  # only downward motion counts

        self._prev = keypoints

        # A fall = ended up down (posture) gated by having moved down fast, but
        # we also alarm on a sustained low posture even after velocity decays,
        # so posture alone can carry the score once someone is on the floor.
        blended = c.w_posture * posture + c.w_velocity * velocity * posture
        score = max(blended, posture if posture > 0.75 else 0.0)
        return max(0.0, min(1.0, score))
