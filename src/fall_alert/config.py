"""Configuration dataclass with documented defaults (standard library only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MODEL = "weights/yolo11n-pose.pt"


@dataclass
class FallConfig:
    """Thresholds of the fall rules.

    All times are in seconds of video time. Velocities are normalized by the
    person's reference height, so ``1.0`` means "one body height per second".

    Attributes:
        min_kp_conf: Minimum keypoint confidence for a shoulder or hip to be used.
        falling_angle: Torso angle from vertical (degrees) that starts ``FALLING``.
        fall_velocity: Downward hip velocity (heights/s) that starts ``FALLING``.
        velocity_window: Time span (s) over which the hip velocity is measured.
        height_window: Time span (s) over which the reference height (max box height) is kept.
        lying_angle: Torso angle (degrees) at or above which the posture counts as lying.
        lying_aspect: Box aspect ratio w/h at or above which the posture counts as lying.
        upright_angle: Torso angle (degrees) at or below which the posture counts as upright.
        upright_aspect: Box aspect ratio w/h at or below which the posture counts as upright.
        fallen_seconds: Lying posture must be held this long in ``FALLEN`` before ``ALERT``.
        lying_gap: Non-lying frames shorter than this do not interrupt ``FALLEN``.
        falling_timeout: ``FALLING`` without lying for this long returns to ``UPRIGHT``.
        recover_seconds: Upright posture held this long ends ``ALERT``.
        lost_seconds: A track not seen for this long is dropped (its open alert is closed).
    """

    min_kp_conf: float = 0.5
    falling_angle: float = 45.0
    fall_velocity: float = 0.8
    velocity_window: float = 0.3
    height_window: float = 2.0
    lying_angle: float = 60.0
    lying_aspect: float = 1.2
    upright_angle: float = 30.0
    upright_aspect: float = 0.8
    fallen_seconds: float = 1.0
    lying_gap: float = 0.3
    falling_timeout: float = 2.0
    recover_seconds: float = 2.0
    lost_seconds: float = 2.0


@dataclass
class RunConfig:
    """Options of one video run.

    Attributes:
        input_path: Input video.
        output_path: Annotated output MP4 (written at the input FPS).
        model: Ultralytics pose weights path or official name.
        conf: Detection confidence threshold.
        imgsz: Inference image size.
        tracker: Ultralytics tracker config (``bytetrack.yaml`` or ``botsort.yaml``).
        display: Show a live preview window.
        events_json: Optional path of the alert-event JSON file.
        trace_csv: Optional path of a per-frame, per-track feature CSV (for tuning).
        fall: Fall-rule thresholds.
    """

    input_path: str
    output_path: str
    model: str = DEFAULT_MODEL
    conf: float = 0.35
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    display: bool = False
    events_json: Optional[str] = None
    trace_csv: Optional[str] = None
    fall: FallConfig = field(default_factory=FallConfig)
