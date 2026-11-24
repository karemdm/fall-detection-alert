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
    kps = make_keypoints(((40, 20), (60, 20)), ((40, 60), (60, 60)))
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


def test_keypoint_exactly_at_the_gate_is_used():
    assert visible_midpoint((0, 0, 0.5), (10, 0, 0.5), 0.5) == (5.0, 0.0)


def test_upside_down_torso_angle_is_above_90():
    kps = make_keypoints(((40, 100), (60, 100)), ((42, 40), (58, 40)))
    f = compute_features(kps, (30, 30, 70, 110), 0.5)
    assert f.torso_angle > 170.0


def test_torso_angle_ignores_the_side_of_the_lean():
    left = compute_features(make_keypoints(((40, 20), (40, 20)), ((80, 60), (80, 60))), (0, 0, 100, 100), 0.5)
    right = compute_features(make_keypoints(((80, 20), (80, 20)), ((40, 60), (40, 60))), (0, 0, 100, 100), 0.5)
    assert math.isclose(left.torso_angle, right.torso_angle) and math.isclose(left.torso_angle, 45.0)


def test_single_same_side_pair_is_enough_for_an_angle():
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[LEFT_SHOULDER] = (50.0, 20.0, 0.9)
    kps[LEFT_HIP] = (50.0, 60.0, 0.9)
    f = compute_features(kps, (30, 0, 70, 120), 0.5)
    assert math.isclose(f.torso_angle, 0.0, abs_tol=1e-9) and f.hip_y == 60.0


def test_cross_side_shoulder_and_hip_give_no_angle():
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[LEFT_SHOULDER] = (50.0, 20.0, 0.9)
    kps[RIGHT_HIP] = (50.0, 60.0, 0.9)
    f = compute_features(kps, (30, 0, 70, 120), 0.5)
    assert f.torso_angle is None and f.hip_y == 60.0


def test_both_sides_are_averaged_not_mixed_across_the_body():
    # upright person seen from the front with a wide body: same-side pairs are vertical (0 deg),
    # while the old shoulder-mid -> hip-mid on a lying body would mix sides
    kps = make_keypoints(((40, 20), (80, 20)), ((40, 60), (80, 60)))
    assert math.isclose(compute_features(kps, (30, 0, 90, 120), 0.5).torso_angle, 0.0, abs_tol=1e-9)
    # left pair leans 45 deg, right pair is vertical: average 22.5
    kps = make_keypoints(((40, 20), (80, 20)), ((80, 60), (80, 60)))
    kps[LEFT_HIP] = (80.0, 60.0, 0.9)
    kps[LEFT_SHOULDER] = (40.0, 20.0, 0.9)
    assert math.isclose(compute_features(kps, (30, 0, 90, 120), 0.5).torso_angle, 22.5)


def test_one_good_side_is_used_when_the_other_side_is_cross_only():
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[LEFT_SHOULDER] = (40.0, 20.0, 0.9)
    kps[LEFT_HIP] = (40.0, 60.0, 0.9)
    kps[RIGHT_SHOULDER] = (60.0, 20.0, 0.9)  # right hip missing: right pair unusable
    assert math.isclose(compute_features(kps, (30, 0, 70, 120), 0.5).torso_angle, 0.0, abs_tol=1e-9)


def test_only_hips_visible_gives_hip_height_but_no_angle():
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[LEFT_HIP] = (40.0, 70.0, 0.9)
    f = compute_features(kps, (30, 0, 70, 120), 0.5)
    assert f.torso_angle is None and f.hip_y == 70.0 and f.visible_keypoints == 1


def test_visible_keypoint_count_uses_the_gate():
    kps = [(0.0, 0.0, 0.9)] * 10 + [(0.0, 0.0, 0.49)] * 7
    assert compute_features(kps, (0, 0, 10, 10), 0.5).visible_keypoints == 10


def test_inverted_or_empty_boxes_do_not_crash():
    assert aspect_ratio((10, 10, 0, 0)) == math.inf  # negative size is clamped to zero height
    f = compute_features([], (10, 10, 0, 0), 0.5)
    assert f.box_height == 0.0 and f.aspect_ratio == math.inf
    assert aspect_ratio((5, 0, 5, 10)) == 0.0
