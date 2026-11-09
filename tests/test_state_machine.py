"""Tests for the per-track fall state machine."""

import math

from fall_alert.config import FallConfig
from fall_alert.features import PoseFeatures
from fall_alert.state_machine import (
    ALERT,
    FALLEN,
    FALLING,
    UPRIGHT,
    FallStateMachine,
    MotionHistory,
    is_lying,
    is_upright,
)

STANDING = PoseFeatures(torso_angle=3.0, aspect_ratio=0.35, hip_y=300.0, box_height=500.0)
LYING = PoseFeatures(torso_angle=88.0, aspect_ratio=2.5, hip_y=650.0, box_height=180.0)
BENT = PoseFeatures(torso_angle=50.0, aspect_ratio=0.7, hip_y=320.0, box_height=420.0)
FPS = 30.0


def feed(machine, start, end, features):
    """Feeds ``features`` at 30 FPS for ``start <= t < end`` and returns the states seen."""
    states = []
    n = int(round((end - start) * FPS))
    for i in range(n):
        states.append(machine.update(start + i / FPS, features))
    return states


def test_posture_classifiers():
    cfg = FallConfig()
    assert is_upright(STANDING, cfg) and not is_lying(STANDING, cfg)
    assert is_lying(LYING, cfg) and not is_upright(LYING, cfg)
    assert not is_upright(BENT, cfg) and not is_lying(BENT, cfg)
    wide_no_angle = PoseFeatures(None, 1.5, None, 100.0)
    assert is_lying(wide_no_angle, cfg)
    narrow_no_angle = PoseFeatures(None, 0.4, None, 100.0)
    assert is_upright(narrow_no_angle, cfg)


def test_standing_stays_upright():
    m = FallStateMachine(FallConfig())
    assert set(feed(m, 0.0, 5.0, STANDING)) == {UPRIGHT}


def test_fall_reaches_alert_after_fallen_seconds():
    cfg = FallConfig(fallen_seconds=1.0)
    m = FallStateMachine(cfg)
    feed(m, 0.0, 1.0, STANDING)
    assert m.update(1.0, LYING) == FALLEN
    assert m.fall_start == 1.0
    states = feed(m, 1.0 + 1 / FPS, 1.9, LYING)
    assert set(states) == {FALLEN}
    states = feed(m, 1.9, 2.5, LYING)
    assert states[-1] == ALERT
    assert m.fall_start == 1.0


def test_recovery_from_alert_needs_upright_for_recover_seconds():
    cfg = FallConfig(recover_seconds=2.0)
    m = FallStateMachine(cfg)
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 3.0, LYING)
    assert m.state == ALERT
    states = feed(m, 3.0, 4.5, STANDING)
    assert set(states) == {ALERT}
    states = feed(m, 4.5, 5.5, STANDING)
    assert states[-1] == UPRIGHT
    assert m.fall_start is None


def test_bending_times_out_back_to_upright():
    cfg = FallConfig(falling_timeout=2.0)
    m = FallStateMachine(cfg)
    feed(m, 0.0, 1.0, STANDING)
    states = feed(m, 1.0, 2.5, BENT)
    assert set(states) == {FALLING}
    states = feed(m, 2.5, 3.5, BENT)
    assert UPRIGHT in states and ALERT not in states


def test_fast_downward_motion_starts_falling():
    cfg = FallConfig(fall_velocity=0.8)
    m = FallStateMachine(cfg)
    feed(m, 0.0, 1.0, STANDING)
    state = UPRIGHT
    t = 1.0
    for i in range(10):  # hip drops 1.5 heights/s while the torso stays vertical
        t = 1.0 + i / FPS
        f = PoseFeatures(3.0, 0.35, 300.0 + 1.5 * 500.0 * i / FPS, 500.0)
        state = m.update(t, f)
    assert state == FALLING
    assert m.velocity is not None and m.velocity > 0.8


def test_short_non_lying_gap_keeps_fallen():
    cfg = FallConfig(lying_gap=0.3, fallen_seconds=1.0)
    m = FallStateMachine(cfg)
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 1.5, LYING)
    feed(m, 1.5, 1.6, BENT)  # 0.1 s keypoint glitch
    assert m.state == FALLEN
    states = feed(m, 1.6, 2.2, LYING)
    assert states[-1] == ALERT


def test_long_non_lying_gap_returns_to_falling():
    cfg = FallConfig(lying_gap=0.3)
    m = FallStateMachine(cfg)
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 1.5, LYING)
    states = feed(m, 1.5, 2.0, BENT)
    assert states[-1] == FALLING


def test_motion_history_velocity_and_reference_height():
    cfg = FallConfig(velocity_window=0.3, height_window=2.0)
    h = MotionHistory(cfg)
    h.update(0.0, PoseFeatures(0.0, 0.3, 100.0, 400.0))
    assert h.velocity() is None
    h.update(0.1, PoseFeatures(0.0, 0.3, 120.0, 300.0))
    assert h.velocity() is None  # span 0.1 s < half the window
    h.update(0.2, PoseFeatures(0.0, 0.3, 140.0, 200.0))
    assert h.reference_height() == 400.0
    assert math.isclose(h.velocity(), (40.0 / 0.2) / 400.0)
    h.update(3.0, PoseFeatures(0.0, 0.3, None, 150.0))
    assert h.reference_height() == 150.0
    assert h.velocity() is None
