"""Tests for the command-line parser (no model or video needed)."""

import contextlib
import io

from fall_alert.cli import build_parser, config_from_args, main
from fall_alert.config import FallConfig


def test_defaults_match_config():
    cfg = config_from_args(build_parser().parse_args(["-i", "a.mp4", "-o", "b.mp4", "--no-display"]))
    assert cfg.fall == FallConfig()
    assert cfg.conf == 0.30 and cfg.imgsz == 640


def test_min_pose_keypoints_is_int():
    cfg = config_from_args(
        build_parser().parse_args(["-i", "a.mp4", "-o", "b.mp4", "--min-pose-keypoints", "8"])
    )
    assert cfg.fall.min_pose_keypoints == 8
    assert isinstance(cfg.fall.min_pose_keypoints, int)


def test_valid_defaults_pass_validation():
    cfg = config_from_args(build_parser().parse_args(["-i", "a.mp4", "-o", "b.mp4", "--no-display"]))
    assert cfg.validate() == []


def test_validation_reports_each_bad_value():
    cfg = config_from_args(build_parser().parse_args([
        "-i", "a.mp4", "-o", "b.mp4", "--lying-gap", "-1", "--velocity-window", "0", "--min-kp-conf", "1.5",
        "--conf", "2", "--imgsz", "0", "--min-pose-keypoints", "30", "--falling-angle", "nan",
    ]))
    text = " | ".join(cfg.validate())
    for name in ("lying_gap", "velocity_window", "min_kp_conf", "conf=", "imgsz", "min_pose_keypoints", "falling_angle"):
        assert name in text


def test_main_exits_with_status_2_on_invalid_values():
    err = io.StringIO()
    code = None
    with contextlib.redirect_stderr(err):
        try:
            main(["-i", "a.mp4", "-o", "b.mp4", "--fallen-seconds", "-3"])
        except SystemExit as exc:
            code = exc.code
    assert code == 2
    assert "fallen_seconds" in err.getvalue()
