"""Alarm logic: threshold + temporal smoothing + confirmation window.

Converts a noisy per-frame fall probability into a single debounced alert event.
This is the principal knob for the sensitivity vs false-alarm-rate trade-off
(ported to TS for in-browser audit).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .schema import EventType
from .temporal import ema


@dataclass
class AlarmConfig:
    """Tunable alarm parameters (frozen before test)."""

    tau: float = 0.6          # probability threshold
    ema_alpha: float = 0.3    # EMA smoothing factor
    k: int = 5                # need k of last m frames above tau
    m: int = 8                # confirmation window length


class AlarmState:
    """Stateful debouncer over a stream of per-frame fall probabilities.

    Pipeline: EMA-smooth the probability, threshold it at ``tau``, then require
    ``k`` of the last ``m`` frames above threshold before emitting a single
    ``FALL``. The alarm latches until the smoothed probability falls back below
    ``tau`` (it will not re-fire while the person is still on the ground).
    """

    def __init__(self, config: AlarmConfig) -> None:
        self.config = config
        self._window: deque[bool] = deque(maxlen=config.m)
        self._ema: float = 0.0
        self._latched: bool = False

    @property
    def smoothed(self) -> float:
        """Current EMA-smoothed fall probability (for telemetry/UI)."""
        return self._ema

    def update(self, fall_prob: float) -> EventType:
        """Feed one frame's probability; return the (possibly NONE) event."""
        cfg = self.config
        self._ema = ema(self._ema, fall_prob, cfg.ema_alpha)
        above = self._ema >= cfg.tau
        self._window.append(above)

        if not above:
            # Smoothed prob dropped back below threshold -> ready to re-arm.
            self._latched = False
            return EventType.NONE

        confirmed = sum(self._window) >= cfg.k
        if confirmed and not self._latched:
            self._latched = True
            return EventType.FALL
        return EventType.NONE
