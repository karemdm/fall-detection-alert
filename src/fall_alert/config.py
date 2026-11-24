"""Configuration dataclass with documented defaults (standard library only)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional

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
        banner_hold: How long (s) after its last observation an open alert is still shown in the
            video banner. Display only; does not change events.
        min_pose_keypoints: A track can fall only after at least one frame with this many
            keypoints above ``min_kp_conf``. Rejects objects detected as a person (bins, coats,
            shadows), which rarely show a body pose; ``0`` disables the check.
        stitch_distance: A new track that starts within this distance (in body heights) of a
            track lost less than ``lost_seconds`` ago continues that track's state and alert.
            Handles tracker id switches when the box changes shape during a fall, so only tracks
            that were not ``UPRIGHT`` when lost are candidates, and only for a newcomer that is tilted (at or above ``falling_angle``) or wide; ``0`` disables.
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
    banner_hold: float = 1.0
    min_pose_keypoints: int = 6
    stitch_distance: float = 1.0

    def validate(self) -> List[str]:
        """Returns a message per out-of-range value (empty if the thresholds are usable).

        ``NaN`` is rejected everywhere. Times must not be negative; the two velocity windows must
        be positive (a zero window would never produce a velocity).
        """
        problems: List[str] = []

        def check(name: str, ok: bool, rule: str) -> None:
            if not ok:
                problems.append(f"{name}={getattr(self, name)!r} must be {rule}")

        def inside(v: float, lo: float, hi: float = math.inf) -> bool:
            return lo <= v <= hi

        check("min_kp_conf", inside(self.min_kp_conf, 0.0, 1.0), "between 0 and 1")
        for name in ("falling_angle", "lying_angle", "upright_angle"):
            check(name, inside(getattr(self, name), 0.0, 180.0), "between 0 and 180 degrees")
        check("fall_velocity", inside(self.fall_velocity, 0.0), "0 or more")
        for name in ("velocity_window", "height_window", "lying_aspect", "upright_aspect"):
            check(name, getattr(self, name) > 0, "greater than 0")
        for name in ("fallen_seconds", "lying_gap", "falling_timeout", "recover_seconds", "lost_seconds",
                     "banner_hold", "stitch_distance"):
            check(name, inside(getattr(self, name), 0.0), "0 or more")
        check("min_pose_keypoints", 0 <= self.min_pose_keypoints <= 17, "between 0 and 17")
        return problems


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
        draw_skeleton: Also draw the pose keypoints and skeleton.
        events_json: Optional path of the alert-event JSON file.
        trace_csv: Optional path of a per-frame, per-track feature CSV (for tuning).
        fall: Fall-rule thresholds.
    """

    input_path: str
    output_path: str
    model: str = DEFAULT_MODEL
    conf: float = 0.30
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    display: bool = False
    draw_skeleton: bool = False
    events_json: Optional[str] = None
    trace_csv: Optional[str] = None
    fall: FallConfig = field(default_factory=FallConfig)

    def validate(self) -> List[str]:
        """Returns a message per out-of-range option, including the fall thresholds."""
        problems: List[str] = []
        if not 0.0 <= self.conf <= 1.0:
            problems.append(f"conf={self.conf!r} must be between 0 and 1")
        if self.imgsz <= 0:
            problems.append(f"imgsz={self.imgsz!r} must be greater than 0")
        return problems + self.fall.validate()
