"""Tests for the frame time and frame rate helpers (no video needed)."""

import math

from fall_alert.timing import DEFAULT_FPS, frame_time, valid_fps


def test_valid_fps_accepts_normal_rates():
    assert valid_fps(25.0) == 25.0
    assert valid_fps(29.97) == 29.97


def test_valid_fps_replaces_missing_or_broken_rates():
    for bad in (0, 0.0, -5.0, None, math.nan, math.inf, -math.inf):
        assert valid_fps(bad) == DEFAULT_FPS
    assert valid_fps(0, default=10.0) == 10.0


def test_frame_time_uses_container_timestamp():
    assert frame_time(1500.0, 7, 30.0) == 1.5


def test_frame_time_falls_back_to_frame_index():
    assert frame_time(0.0, 0, 30.0) == 0.0
    assert math.isclose(frame_time(0.0, 15, 30.0), 0.5)
    assert math.isclose(frame_time(None, 30, 25.0), 1.2)
    assert math.isclose(frame_time(math.nan, 30, 30.0), 1.0)
    assert math.isclose(frame_time(-40.0, 30, 30.0), 1.0)


def test_frame_time_never_goes_backwards():
    # a backend that returns 0 for one frame, or a timestamp that jumps back
    assert frame_time(0.0, 10, 30.0, previous=5.0) == 5.0
    assert frame_time(4000.0, 11, 30.0, previous=5.0) == 5.0
    assert frame_time(5000.0, 12, 30.0, previous=5.0) == 5.0
    assert frame_time(5033.0, 13, 30.0, previous=5.0) == 5.033
