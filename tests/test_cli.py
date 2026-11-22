"""Tests for the command-line parser (no model or video needed)."""

from fall_alert.cli import build_parser, config_from_args
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
