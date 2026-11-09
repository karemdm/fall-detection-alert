"""Command-line interface (``fall-alert``)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import fields
from typing import List, Optional

from fall_alert import __version__
from fall_alert.config import DEFAULT_MODEL, FallConfig, RunConfig

# Help text for every FallConfig threshold exposed on the CLI.
THRESHOLD_HELP = {
    "min_kp_conf": "Minimum keypoint confidence for shoulders/hips",
    "falling_angle": "Torso angle from vertical (deg) that starts FALLING",
    "fall_velocity": "Downward hip velocity (body heights/s) that starts FALLING",
    "velocity_window": "Time span (s) of the hip velocity estimate",
    "height_window": "Time span (s) of the reference body height (max box height)",
    "lying_angle": "Torso angle (deg) at or above which the posture is lying",
    "lying_aspect": "Box w/h at or above which the posture is lying",
    "upright_angle": "Torso angle (deg) at or below which the posture is upright",
    "upright_aspect": "Box w/h at or below which the posture is upright",
    "fallen_seconds": "Seconds the lying posture must be held before ALERT",
    "lying_gap": "Non-lying gaps (s) shorter than this do not interrupt FALLEN",
    "falling_timeout": "Seconds in FALLING without lying before returning to UPRIGHT",
    "recover_seconds": "Seconds of upright posture that end ALERT",
    "lost_seconds": "Seconds without detection after which a track is dropped",
}


def display_available() -> bool:
    """Whether a GUI display is likely available (X11/Wayland on Linux; assumed on Windows/macOS)."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def build_parser() -> argparse.ArgumentParser:
    """Creates the argument parser; threshold defaults come from :class:`FallConfig`."""
    p = argparse.ArgumentParser(
        prog="fall-alert",
        description="Rule-based fall detection on Ultralytics YOLO pose keypoints with tracking.",
    )
    p.add_argument("-i", "--input", required=True, help="Input video path.")
    p.add_argument("-o", "--output", required=True, help="Annotated output MP4 path.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Pose weights path or official name (default: %(default)s).")
    p.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold (default: %(default)s).")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: %(default)s).")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config (default: %(default)s).")
    p.add_argument("--events-json", help="Write alert events (track id, fall/alert start, end) to this JSON file.")
    p.add_argument("--trace-csv", help="Write per-frame, per-track features and states to this CSV file.")
    p.add_argument("--no-display", action="store_true", help="Do not open the preview window (implied without a display).")
    p.add_argument("--draw-skeleton", action="store_true", help="Also draw the pose keypoints and skeleton (off by default).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    g = p.add_argument_group("fall thresholds")
    for f in fields(FallConfig):
        g.add_argument(
            "--" + f.name.replace("_", "-"),
            type=float,
            default=f.default,
            help=f"{THRESHOLD_HELP[f.name]} (default: %(default)s).",
        )
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Builds a :class:`RunConfig` from parsed arguments."""
    fall = FallConfig(**{f.name: getattr(args, f.name) for f in fields(FallConfig)})
    return RunConfig(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        tracker=args.tracker,
        display=not args.no_display and display_available(),
        draw_skeleton=args.draw_skeleton,
        events_json=args.events_json,
        trace_csv=args.trace_csv,
        fall=fall,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Parses arguments and runs the pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from fall_alert.pipeline import run

    return run(config_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
