"""Temporal utilities: sliding window buffer and EMA smoothing.

The classifier consumes a window of T=32-64 frames; low-confidence joints are
masked and temporally imputed from window context (confidence-gating).
"""

from __future__ import annotations

from collections import deque

from ..schema import Keypoint


class SlidingWindow:
    """Fixed-length rolling buffer of pose frames feeding the classifier."""

    def __init__(self, size: int = 32) -> None:
        self.size = size
        self._buf: deque[list[Keypoint]] = deque(maxlen=size)

    def push(self, keypoints: list[Keypoint]) -> None:
        """Append one frame's keypoints."""
        self._buf.append(keypoints)

    def is_full(self) -> bool:
        return len(self._buf) == self.size

    def as_tensor(self, min_score: float = 0.2):
        """Stack the window into a (T, J, C) array for the ONNX classifier.

        Low-confidence joints (score < ``min_score``) are imputed from window
        context: forward-filled from the most recent confident observation, then
        back-filled, so a brief occlusion does not punch holes in the input.
        Requires numpy; raises if the window is not yet full.
        """
        import numpy as np

        if not self.is_full():
            raise ValueError(f"window not full ({len(self._buf)}/{self.size})")

        frames = list(self._buf)
        t = len(frames)
        j = len(frames[0])
        arr = np.zeros((t, j, 3), dtype=np.float32)
        for ti, kps in enumerate(frames):
            for ji, kp in enumerate(kps):
                arr[ti, ji] = (kp.x, kp.y, kp.score)

        # Confidence-gate + temporal impute per joint (x, y channels only).
        for ji in range(j):
            confident = arr[:, ji, 2] >= min_score
            if not confident.any():
                continue
            idx = np.where(confident, np.arange(t), -1)
            # forward fill
            ff = np.maximum.accumulate(idx)
            ff[ff < 0] = idx[confident][0]
            # back fill for the leading gap
            arr[:, ji, 0] = arr[ff, ji, 0]
            arr[:, ji, 1] = arr[ff, ji, 1]
        return arr


def ema(prev: float, value: float, alpha: float) -> float:
    """Exponential moving average step (shared with the TS port)."""
    return alpha * value + (1.0 - alpha) * prev
