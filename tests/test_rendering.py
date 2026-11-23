"""Tests for drawing (needs ``cv2`` and ``numpy``; no model or video)."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from fall_alert.features import PoseFeatures  # noqa: E402
from fall_alert.monitor import AlertEvent, TrackStatus  # noqa: E402
from fall_alert.rendering import draw_skeleton, render  # noqa: E402
from fall_alert.state_machine import ALERT, UPRIGHT  # noqa: E402

BOX = (100.0, 100.0, 200.0, 400.0)
KEYPOINTS = [(150.0, 100.0 + 15.0 * i, 0.9) for i in range(17)]


def blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def status(state, angle):
    return TrackStatus(state, PoseFeatures(angle, 0.33, 250.0, 300.0), None)


def test_draw_skeleton_survives_a_short_keypoint_list():
    frame = blank()
    draw_skeleton(frame, KEYPOINTS[:5], 0.5, 2)  # a custom model with fewer than 17 keypoints
    draw_skeleton(frame, [], 0.5, 2)


def test_draw_skeleton_skips_low_confidence_points():
    frame = blank()
    draw_skeleton(frame, [(150.0, 150.0, 0.1)] * 17, 0.5, 2)
    assert not frame.any()


def test_upright_track_without_angle_is_not_drawn():
    frame = blank()
    render(frame, {1: (BOX, KEYPOINTS)}, {1: status(UPRIGHT, None)}, [], 1.0, 0.5)
    # only the time stamp (bottom-right) is drawn, nothing around the box
    assert not frame[:440, :500].any()


def test_alert_track_is_drawn_with_banner_and_stamp():
    frame = blank()
    event = AlertEvent(1, 2.0, 3.0)
    render(frame, {1: (BOX, KEYPOINTS)}, {1: status(ALERT, 80.0)}, [event], 4.0, 0.5, draw_keypoints=True)
    assert frame[10, 300].tolist() == [0, 0, 230]  # red banner (BGR) at the top
    assert frame[250, 100].tolist() == [0, 0, 230]  # box edge in the alert color


def test_render_handles_boxes_outside_the_frame():
    frame = blank(240, 320)
    render(frame, {1: ((-50.0, -20.0, 400.0, 300.0), KEYPOINTS)}, {1: status(ALERT, 80.0)}, [], 0.0, 0.5)
