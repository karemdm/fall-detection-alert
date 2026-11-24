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


def test_object_without_body_pose_cannot_alert():
    """A wide 'person' box with almost no confident keypoints (e.g. a bin) never leaves UPRIGHT."""
    m = FallStateMachine(FallConfig())
    object_like = PoseFeatures(torso_angle=None, aspect_ratio=1.6, hip_y=None, box_height=120.0, visible_keypoints=2)
    assert set(feed(m, 0.0, 5.0, object_like)) == {UPRIGHT}
    assert not m.armed


def test_one_frame_with_a_body_pose_arms_the_track():
    m = FallStateMachine(FallConfig())
    weak_lying = PoseFeatures(torso_angle=88.0, aspect_ratio=2.5, hip_y=650.0, box_height=180.0, visible_keypoints=3)
    feed(m, 0.0, 0.1, LYING)
    assert m.armed
    assert ALERT in feed(m, 0.1, 3.0, weak_lying)


def test_pose_gate_can_be_disabled():
    m = FallStateMachine(FallConfig(min_pose_keypoints=0))
    weak_lying = PoseFeatures(torso_angle=88.0, aspect_ratio=2.5, hip_y=650.0, box_height=180.0, visible_keypoints=0)
    assert ALERT in feed(m, 0.0, 3.0, weak_lying)


def feed_at(machine, start, end, features, fps):
    """Like ``feed`` with a chosen frame rate; returns ``(time, state)`` pairs."""
    out = []
    n = int(round((end - start) * fps))
    for i in range(n):
        t = start + i / fps
        out.append((t, machine.update(t, features)))
    return out


def test_velocity_is_none_for_zero_or_tiny_time_span():
    cfg = FallConfig(velocity_window=0.0)
    h = MotionHistory(cfg)
    h.update(1.0, PoseFeatures(0.0, 0.3, 100.0, 400.0))
    h.update(1.0, PoseFeatures(0.0, 0.3, 140.0, 400.0))  # duplicate timestamp: no ZeroDivisionError
    assert h.velocity() is None


def test_velocity_needs_enough_frames_per_window():
    """Frames further apart than velocity_window leave one sample in the window: no velocity."""
    h = MotionHistory(FallConfig(velocity_window=0.3))
    for i in range(5):
        h.update(i * 0.4, PoseFeatures(0.0, 0.3, 100.0 + 50.0 * i, 400.0))
        assert h.velocity() is None


def test_velocity_ignores_frames_without_hips():
    h = MotionHistory(FallConfig(velocity_window=0.3))
    h.update(0.0, PoseFeatures(0.0, 0.3, 100.0, 400.0))
    h.update(0.1, PoseFeatures(None, 0.3, None, 400.0))
    h.update(0.2, PoseFeatures(0.0, 0.3, 140.0, 400.0))
    assert math.isclose(h.velocity(), (40.0 / 0.2) / 400.0)


def test_fast_drop_does_not_start_falling_before_the_track_is_armed():
    m = FallStateMachine(FallConfig(fall_velocity=0.8))
    for i in range(12):
        f = PoseFeatures(3.0, 0.35, 300.0 + 750.0 * i / FPS, 500.0, visible_keypoints=2)
        assert m.update(i / FPS, f) == UPRIGHT
    assert m.velocity is not None and m.velocity > 0.8 and not m.armed


def test_alert_stays_while_posture_is_neither_lying_nor_upright():
    m = FallStateMachine(FallConfig(recover_seconds=2.0))
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 3.0, LYING)
    assert m.state == ALERT
    assert set(feed(m, 3.0, 9.0, BENT)) == {ALERT}  # sitting up for a long time is not recovery


def test_interrupted_upright_does_not_recover():
    m = FallStateMachine(FallConfig(recover_seconds=2.0))
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 3.0, LYING)
    feed(m, 3.0, 4.5, STANDING)
    feed(m, 4.5, 4.6, BENT)  # the upright timer restarts
    assert set(feed(m, 4.6, 6.0, STANDING)) == {ALERT}
    assert feed(m, 6.0, 7.0, STANDING)[-1] == UPRIGHT


def test_fall_start_is_kept_when_fallen_drops_back_to_falling():
    m = FallStateMachine(FallConfig(lying_gap=0.3))
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 1.5, LYING)
    assert m.fall_start == 1.0
    feed(m, 1.5, 2.0, BENT)
    assert m.state == FALLING and m.fall_start == 1.0
    feed(m, 2.0, 4.0, LYING)
    assert m.state == ALERT and m.fall_start == 1.0


def test_falling_timeout_counts_from_the_latest_entry_after_a_drop_back():
    m = FallStateMachine(FallConfig(lying_gap=0.3, falling_timeout=2.0))
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 1.5, LYING)
    feed(m, 1.5, 3.0, BENT)  # FALLING again from ~1.9 s; 2.0 s timeout not yet reached at 3.0 s
    assert m.state == FALLING and m.fall_start == 1.0
    feed(m, 3.0, 4.5, STANDING)
    assert m.state == UPRIGHT and m.fall_start is None


def test_new_episode_after_upright_gets_a_new_fall_start():
    m = FallStateMachine(FallConfig())
    feed(m, 0.0, 1.0, STANDING)
    feed(m, 1.0, 3.0, LYING)
    feed(m, 3.0, 6.0, STANDING)
    assert m.state == UPRIGHT
    feed(m, 6.0, 7.0, STANDING)
    feed(m, 7.0, 7.5, LYING)
    assert m.fall_start == 7.0


def test_upside_down_torso_counts_as_lying():
    upside_down = PoseFeatures(torso_angle=170.0, aspect_ratio=0.5, hip_y=100.0, box_height=300.0)
    assert is_lying(upside_down, FallConfig()) and not is_upright(upside_down, FallConfig())


def test_zero_height_box_is_lying_by_aspect_but_cannot_arm_an_empty_track():
    degenerate = PoseFeatures(torso_angle=None, aspect_ratio=math.inf, hip_y=None, box_height=0.0, visible_keypoints=0)
    assert is_lying(degenerate, FallConfig())
    assert set(feed(FallStateMachine(FallConfig()), 0.0, 3.0, degenerate)) == {UPRIGHT}


def test_alert_time_does_not_depend_on_frame_rate():
    """The same fall reaches ALERT at the same video time at 5, 10, 30 and 60 FPS (one frame tolerance)."""
    alert_times = {}
    for fps in (5, 10, 30, 60):
        m = FallStateMachine(FallConfig(fallen_seconds=1.0))
        feed_at(m, 0.0, 1.0, STANDING, fps)
        seq = feed_at(m, 1.0, 4.0, LYING, fps)
        alert_times[fps] = next(t for t, s in seq if s == ALERT)
    for fps, t in alert_times.items():
        assert 2.0 <= t <= 2.0 + 1.0 / fps + 1e-9, (fps, t)


def test_sparse_frames_keep_the_lying_timer_running():
    """Observed lying frames 0.5 s apart (detector misses) still confirm after fallen_seconds."""
    m = FallStateMachine(FallConfig(fallen_seconds=1.0, lying_gap=0.3))
    feed(m, 0.0, 1.0, STANDING)
    seq = feed_at(m, 1.0, 3.0, LYING, 2)
    assert [s for _, s in seq][:2] == [FALLEN, FALLEN] and seq[2][1] == ALERT
