"""Evaluation metrics: sensitivity, specificity, FAR/hr, time-to-alert.

Frame-level and event-level scoring both supported. Implementations are shared
with the TS port so dashboard re-scoring matches the node exactly.

Labels are treated as binary: 1 = fall (positive), 0 = not-fall.
"""

from __future__ import annotations

from collections.abc import Sequence


def _confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p:
            if t:
                tp += 1
            else:
                fp += 1
        else:
            if t:
                fn += 1
            else:
                tn += 1
    return tp, fp, tn, fn


def sensitivity(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Recall on falls (TP / (TP + FN)). Returns 0.0 with no positives."""
    tp, _, _, fn = _confusion(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom else 0.0


def specificity(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """TN / (TN + FP). Returns 0.0 with no negatives."""
    _, fp, tn, _ = _confusion(y_true, y_pred)
    denom = tn + fp
    return tn / denom if denom else 0.0


def precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """TP / (TP + FP). Returns 0.0 with no positive predictions."""
    tp, fp, _, _ = _confusion(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom else 0.0


def f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred)
    r = sensitivity(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def false_alarms_per_hour(
    events: Sequence[float], duration_hours: float
) -> float:
    """Nuisance alerts per hour over continuous unstaged footage (target < 1).

    ``events`` is the sequence of false-alarm timestamps (or any per-alarm
    marker); only its length matters here.
    """
    if duration_hours <= 0.0:
        raise ValueError("duration_hours must be positive")
    return len(events) / duration_hours


def time_to_alert(impact_ts: float, alert_ts: float) -> float:
    """Latency from impact to alert event; report p50/p95 over a dataset."""
    return alert_ts - impact_ts
