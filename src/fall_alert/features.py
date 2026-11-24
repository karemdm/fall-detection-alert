"""Pose features from COCO-17 keypoint tuples (standard library only).

A keypoint is a tuple ``(x, y, confidence)`` in image pixels (y grows downward).
A box is ``(x1, y1, x2, y2)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Keypoint = Tuple[float, float, float]
Box = Tuple[float, float, float, float]
Point = Tuple[float, float]

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

# COCO-17 skeleton (0-based keypoint indices), used for drawing.
SKELETON: Tuple[Tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)


@dataclass(frozen=True)
class PoseFeatures:
    """Per-frame features of one person.

    Attributes:
        torso_angle: Angle of shoulder -> hip from vertical (same-side pairs, averaged when both
            sides are visible), degrees in ``[0, 180]`` (0 = upright, 90 = horizontal), or ``None``
            if no same-side pair passes the gate.
        aspect_ratio: Box width / height.
        hip_y: Vertical image position of the hip midpoint, or ``None`` if gated out.
        box_height: Box height in pixels.
        visible_keypoints: Number of keypoints at or above the confidence gate.
    """

    torso_angle: Optional[float]
    aspect_ratio: float
    hip_y: Optional[float]
    box_height: float
    visible_keypoints: int = 17


def visible_midpoint(a: Keypoint, b: Keypoint, min_conf: float) -> Optional[Point]:
    """Midpoint of two keypoints, using only those with confidence >= ``min_conf``.

    Args:
        a: First keypoint.
        b: Second keypoint.
        min_conf: Confidence gate.

    Returns:
        The midpoint if both pass, the single visible point if one passes, else ``None``.
    """
    pts = [(k[0], k[1]) for k in (a, b) if k[2] >= min_conf]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def angle_from_vertical(top: Point, bottom: Point) -> float:
    """Angle in degrees between the vector ``top -> bottom`` and the downward vertical.

    Returns 0 when ``bottom`` is straight below ``top``, 90 when level and 180
    when straight above (upside down).
    """
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(abs(dx), dy))


def torso_angle(keypoints: Sequence[Keypoint], min_conf: float) -> Optional[float]:
    """Torso angle from vertical using same-side shoulder/hip pairs only.

    The left pair (shoulder 5, hip 11) and the right pair (shoulder 6, hip 12) each give an
    angle when both of their points pass ``min_conf``; with both pairs the two angles are
    averaged. A shoulder of one side with only the hip of the other side (a diagonal) is not used,
    because on a lying or turned body it gives a skewed angle.

    Returns:
        The angle in degrees, or ``None`` if neither side has both points.
    """
    if len(keypoints) <= RIGHT_HIP:
        return None
    angles = []
    for s, h in ((LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)):
        if keypoints[s][2] >= min_conf and keypoints[h][2] >= min_conf:
            angles.append(angle_from_vertical(keypoints[s][:2], keypoints[h][:2]))
    return sum(angles) / len(angles) if angles else None


def aspect_ratio(box: Box) -> float:
    """Width / height of a box (``inf`` for a zero-height box)."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w / h if h > 0 else math.inf


def compute_features(keypoints: Sequence[Keypoint], box: Box, min_conf: float) -> PoseFeatures:
    """Computes the torso angle, aspect ratio and hip height of one person.

    The torso angle needs a shoulder and the hip of the same side above ``min_conf`` (see
    :func:`torso_angle`); the hip position needs at least one hip.

    Args:
        keypoints: 17 COCO keypoints ``(x, y, conf)``.
        box: Person box ``(x1, y1, x2, y2)``.
        min_conf: Keypoint confidence gate.

    Returns:
        The :class:`PoseFeatures`.
    """
    hips = None
    if len(keypoints) > RIGHT_HIP:
        hips = visible_midpoint(keypoints[LEFT_HIP], keypoints[RIGHT_HIP], min_conf)
    angle = torso_angle(keypoints, min_conf)
    return PoseFeatures(
        torso_angle=angle,
        aspect_ratio=aspect_ratio(box),
        hip_y=hips[1] if hips else None,
        box_height=max(0.0, box[3] - box[1]),
        visible_keypoints=sum(1 for k in keypoints if k[2] >= min_conf),
    )
