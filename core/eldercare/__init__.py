"""Eldercare fall detection — shared core library.

Single source of truth for the keypoint schema, temporal smoothing, geometric
fall features, alarm logic, and metric definitions. Reused by the edge node and
the FastAPI backend; the hot path is mirrored in TypeScript for the dashboard.
"""

from __future__ import annotations

__all__ = [
    "schema",
    "features",
    "heuristic",
    "alarm",
    "metrics",
    "pose",
    "temporal",
]
