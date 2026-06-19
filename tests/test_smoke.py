"""Core library tests: schema, features, heuristic, alarm, metrics, e2e.

These cover the MVP's actual logic (the geometric Baseline A path). The deep
classifier parity / cross-dataset eval tests remain TODO until weights exist.
"""

from __future__ import annotations

import math

import pytest

from eldercare.alarm import AlarmConfig, AlarmState
from eldercare.features import (
    bounding_box_aspect_ratio,
    centroid_vertical_velocity,
    head_to_hip_angle,
    sustained_low_posture,
)
from eldercare.heuristic import HeuristicFallDetector
from eldercare.metrics import f1, sensitivity, specificity
from eldercare.schema import (
    COCO_KEYPOINTS,
    EventType,
    Keypoint,
    PoseFrame,
    SchemaError,
    validate_pose_frame,
)
from eldercare.synthetic import synthetic_stream


def _kps(points: list[tuple[float, float]], score: float = 0.9) -> list[Keypoint]:
    return [Keypoint(x, y, score) for x, y in points]


def _standing() -> list[Keypoint]:
    # tall bounding box, head above hips
    pts = [(0.5, 0.1 + 0.05 * i) for i in range(len(COCO_KEYPOINTS))]
    return _kps(pts)


def _lying() -> list[Keypoint]:
    # wide bounding box, head beside hips (horizontal) with a little body thickness
    pts = [
        (0.1 + 0.05 * i, 0.82 + 0.03 * (i % 2)) for i in range(len(COCO_KEYPOINTS))
    ]
    return _kps(pts)


# --- schema ---------------------------------------------------------------


def test_validate_accepts_good_frame() -> None:
    frame = PoseFrame("n1", 1.0, _standing(), 0.1, EventType.NONE)
    validate_pose_frame(frame)  # should not raise


def test_validate_rejects_wrong_joint_count() -> None:
    frame = PoseFrame("n1", 1.0, _standing()[:10], 0.1, EventType.NONE)
    with pytest.raises(SchemaError):
        validate_pose_frame(frame)


def test_validate_rejects_bad_score() -> None:
    bad = _standing()
    bad[0] = Keypoint(0.5, 0.1, 1.5)
    frame = PoseFrame("n1", 1.0, bad, 0.1, EventType.NONE)
    with pytest.raises(SchemaError):
        validate_pose_frame(frame)


def test_wire_roundtrip() -> None:
    frame = PoseFrame("n1", 1.5, _standing(), 0.42, EventType.FALL)
    back = PoseFrame.from_wire(frame.to_wire())
    assert back.node_id == frame.node_id
    assert back.event is EventType.FALL
    assert back.fall_score == pytest.approx(0.42)
    assert len(back.keypoints) == len(COCO_KEYPOINTS)
    assert back.keypoints[5].x == pytest.approx(frame.keypoints[5].x)


def test_wire_carries_no_pixels() -> None:
    # Privacy invariant: the wire record has only keypoints/score/event/ts.
    wire = PoseFrame("n1", 1.0, _standing(), 0.1, EventType.NONE).to_wire()
    assert set(wire) == {"node_id", "ts", "keypoints", "fall_score", "event"}


# --- features -------------------------------------------------------------


def test_aspect_ratio_flips_on_fall() -> None:
    assert bounding_box_aspect_ratio(_standing()) < 1.0
    assert bounding_box_aspect_ratio(_lying()) > 1.0


def test_head_to_hip_angle_upright_vs_lying() -> None:
    assert head_to_hip_angle(_standing()) < 30.0
    assert head_to_hip_angle(_lying()) > 55.0


def test_centroid_velocity_down_is_positive() -> None:
    prev = _kps([(0.5, 0.2)] * len(COCO_KEYPOINTS))
    curr = _kps([(0.5, 0.6)] * len(COCO_KEYPOINTS))
    v = centroid_vertical_velocity(prev, curr, dt=0.1)
    assert v == pytest.approx(4.0, abs=1e-6)


def test_sustained_low_posture() -> None:
    assert sustained_low_posture(_lying())
    assert not sustained_low_posture(_standing())


def test_features_ignore_low_confidence() -> None:
    pts = _standing()
    pts[0] = Keypoint(99.0, 99.0, 0.0)  # garbage but zero confidence -> ignored
    assert bounding_box_aspect_ratio(pts) < 1.0


# --- heuristic ------------------------------------------------------------


def test_heuristic_low_when_standing_high_when_fallen() -> None:
    det = HeuristicFallDetector()
    standing_score = det.score(_standing(), dt=1 / 30)
    det.reset()
    det.score(_standing(), dt=1 / 30)
    fallen_score = det.score(_lying(), dt=1 / 30)
    assert standing_score < 0.3
    assert fallen_score > 0.6


# --- alarm ----------------------------------------------------------------


def test_alarm_debounces_single_spike() -> None:
    alarm = AlarmState(AlarmConfig(tau=0.6, ema_alpha=1.0, k=5, m=8))
    # one isolated high frame should not trip a k-of-m=5/8 confirmation
    events = [alarm.update(p) for p in [0.0, 0.9, 0.0, 0.0, 0.0]]
    assert EventType.FALL not in events


def test_alarm_fires_once_on_sustained_fall() -> None:
    alarm = AlarmState(AlarmConfig(tau=0.6, ema_alpha=1.0, k=5, m=8))
    events = [alarm.update(0.9) for _ in range(10)]
    assert events.count(EventType.FALL) == 1  # latches, fires exactly once


def test_alarm_rearms_after_recovery() -> None:
    alarm = AlarmState(AlarmConfig(tau=0.6, ema_alpha=1.0, k=3, m=5))
    first = [alarm.update(0.9) for _ in range(5)]
    [alarm.update(0.0) for _ in range(5)]  # recover
    second = [alarm.update(0.9) for _ in range(5)]
    assert EventType.FALL in first
    assert EventType.FALL in second


# --- metrics --------------------------------------------------------------


def test_metrics_basic() -> None:
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    assert sensitivity(y_true, y_pred) == pytest.approx(0.5)
    assert specificity(y_true, y_pred) == pytest.approx(1.0)
    assert f1(y_true, y_pred) == pytest.approx(2 / 3)


# --- end to end -----------------------------------------------------------


def test_synthetic_episode_triggers_exactly_one_fall() -> None:
    """The full Baseline A path on the synthetic stream must fire one FALL."""
    det = HeuristicFallDetector()
    alarm = AlarmState(AlarmConfig())
    dt = 1 / 30
    fired = 0
    alert_t = None
    for t, kps in synthetic_stream(fps=30):
        ev = alarm.update(det.score(kps, dt))
        if ev is EventType.FALL:
            fired += 1
            alert_t = t
    assert fired == 1, f"expected exactly one FALL, got {fired}"
    # the fall happens at stand_s=3.0s; alert should land shortly after
    assert alert_t is not None and 3.0 <= alert_t <= 5.0


def test_no_false_alarm_while_only_standing() -> None:
    det = HeuristicFallDetector()
    alarm = AlarmState(AlarmConfig())
    dt = 1 / 30
    falls = 0
    # 5 seconds of pure standing (no fall phase)
    for _, kps in synthetic_stream(fps=30, stand_s=5.0, fall_s=0.6, down_s=0.0):
        if math.isclose(0.0, 0.0):  # iterate all standing frames
            ev = alarm.update(det.score(kps, dt))
            if ev is EventType.FALL:
                falls += 1
        if _ > 4.9:
            break
    assert falls == 0
