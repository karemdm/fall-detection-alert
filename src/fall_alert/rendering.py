"""Drawing of skeletons, state-colored boxes, labels and the alert banner."""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import cv2
import numpy as np

from fall_alert.features import SKELETON, Keypoint
from fall_alert.monitor import AlertEvent, TrackStatus
from fall_alert.state_machine import ALERT, FALLEN, FALLING, UPRIGHT

# BGR colors per state.
STATE_COLORS = {
    UPRIGHT: (60, 200, 60),
    FALLING: (0, 150, 255),
    FALLEN: (0, 0, 230),
    ALERT: (0, 0, 230),
}
SKELETON_COLOR = (255, 220, 120)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_skeleton(frame: np.ndarray, keypoints: Sequence[Keypoint], min_conf: float, thickness: int) -> None:
    """Draws the COCO skeleton, skipping keypoints below ``min_conf``."""
    for a, b in SKELETON:
        if keypoints[a][2] >= min_conf and keypoints[b][2] >= min_conf:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pa, pb, SKELETON_COLOR, thickness, cv2.LINE_AA)
    for x, y, c in keypoints:
        if c >= min_conf:
            cv2.circle(frame, (int(x), int(y)), thickness + 1, SKELETON_COLOR, -1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color, scale: float, thickness: int) -> None:
    """Filled label box with white text whose bottom-left corner is at ``(x, y)``."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(y, th + base + 2)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - base - 2), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render(
    frame: np.ndarray,
    persons: Mapping[Hashable, tuple],
    statuses: Mapping[Hashable, TrackStatus],
    active: Sequence[AlertEvent],
    t: float,
    min_conf: float,
) -> None:
    """Annotates ``frame`` in place.

    Args:
        frame: BGR frame.
        persons: Track id -> ``(box, keypoints)``.
        statuses: Track id -> :class:`TrackStatus` from the monitor.
        active: Currently open alert events.
        t: Video time in seconds.
        min_conf: Keypoint confidence gate used for drawing.
    """
    h, w = frame.shape[:2]
    unit = max(1, round(h / 720))
    scale = 0.6 * unit
    for track_id, (box, keypoints) in persons.items():
        status = statuses[track_id]
        color = STATE_COLORS[status.state]
        draw_skeleton(frame, keypoints, min_conf, 2 * unit)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 * unit if status.state == ALERT else 2 * unit)
        angle = status.features.torso_angle
        angle_txt = f"{angle:.0f} deg" if angle is not None else "n/a"
        draw_label(frame, f"#{track_id} {status.state} | {angle_txt}", x1, y1 - 2, color, scale, unit)
    if active:
        ids = ", ".join(f"#{e.track_id}" for e in active)
        first = min(e.start for e in active)
        text = f"FALL ALERT {ids} since {first:.1f} s"
        cv2.rectangle(frame, (0, 0), (w, 44 * unit), STATE_COLORS[ALERT], -1)
        cv2.putText(frame, text, (12 * unit, 31 * unit), FONT, 0.9 * unit, (255, 255, 255), 2 * unit, cv2.LINE_AA)
    stamp = f"t = {t:6.2f} s"
    (tw, _), _ = cv2.getTextSize(stamp, FONT, scale, unit)
    draw_label(frame, stamp, w - tw - 16, h - 10, (40, 40, 40), scale, unit)
