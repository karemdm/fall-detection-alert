"""Tests for the keypoint feature functions."""

import math

from fall_alert.features import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    angle_from_vertical,
    aspect_ratio,
    compute_features,
    visible_midpoint,
)


def make_keypoints(shoulders, hips, conf=0.9):
    """17 keypoints with only shoulders and hips set; others have zero confidence."""
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[LEFT_SHOULDER] = (shoulders[0][0], shoulders[0][1], conf)
    kps[RIGHT_SHOULDER] = (shoulders[1][0], shoulders[1][1], conf)
    kps[LEFT_HIP] = (hips[0][0], hips[0][1], conf)
    kps[RIGHT_HIP] = (hips[1][0], hips[1][1], conf)
    return kps


def test_angle_upright_horizontal_and_upside_down():
    assert math.isclose(angle_from_vertical((0, 0), (0, 10)), 0.0)
    assert math.isclose(angle_from_vertical((0, 0), (10, 0)), 90.0)
    assert math.isclose(angle_from_vertical((0, 0), (-10, 0)), 90.0)
    assert math.isclose(angle_from_vertical((0, 0), (0, -10)), 180.0)
    assert math.isclose(angle_from_vertical((0, 0), (10, 10)), 45.0)
    assert angle_from_vertical((5, 5), (5, 5)) == 0.0


def test_visible_midpoint_gating():
    assert visible_midpoint((0, 0, 0.9), (10, 0, 0.9), 0.5) == (5.0, 0.0)
    assert visible_midpoint((0, 0, 0.9), (10, 0, 0.1), 0.5) == (0.0, 0.0)
    assert visible_midpoint((0, 0, 0.1), (10, 0, 0.1), 0.5) is None


def test_aspect_ratio():
    assert math.isclose(aspect_ratio((0, 0, 50, 100)), 0.5)
    assert math.isclose(aspect_ratio((10, 10, 210, 110)), 2.0)
    assert aspect_ratio((0, 0, 10, 0)) == math.inf


def test_compute_features_standing():
    kps = make_keypoints(((40, 20), (60, 20)), ((42, 60), (58, 60)))
    f = compute_features(kps, (30, 0, 70, 120), 0.5)
    assert math.isclose(f.torso_angle, 0.0, abs_tol=1e-9)
    assert math.isclose(f.aspect_ratio, 40 / 120)
    assert f.hip_y == 60
    assert f.box_height == 120


def test_compute_features_lying():
    kps = make_keypoints(((10, 50), (10, 55)), ((60, 50), (60, 55)))
    f = compute_features(kps, (0, 30, 120, 70), 0.5)
    assert math.isclose(f.torso_angle, 90.0)
    assert f.aspect_ratio == 3.0


def test_compute_features_low_confidence_gates_angle():
    kps = make_keypoints(((40, 20), (60, 20)), ((42, 60), (58, 60)), conf=0.2)
    f = compute_features(kps, (30, 0, 70, 120), 0.5)
    assert f.torso_angle is None
    assert f.hip_y is None
    assert math.isclose(f.aspect_ratio, 40 / 120)


def test_compute_features_short_keypoint_list():
    f = compute_features([], (0, 0, 10, 20), 0.5)
    assert f.torso_angle is None and f.hip_y is None
