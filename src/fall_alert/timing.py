"""Frame timestamps and frame rate helpers (standard library only)."""

from __future__ import annotations

import math
from typing import Optional

DEFAULT_FPS = 30.0


def valid_fps(fps: Optional[float], default: float = DEFAULT_FPS) -> float:
    """Returns ``fps`` if it is a finite positive number, else ``default``.

    Containers can report ``0``, a negative number, ``NaN`` or ``inf`` as frame rate.
    """
    if fps is None or not math.isfinite(fps) or fps <= 0:
        return default
    return float(fps)


def frame_time(pos_ms: Optional[float], frame_idx: int, fps: float, previous: Optional[float] = None) -> float:
    """Video time (s) of a frame, never earlier than the previous frame.

    Uses the container timestamp ``pos_ms`` (``CAP_PROP_POS_MSEC``) when it is a finite positive
    number and ``frame_idx / fps`` otherwise (``0`` is also what the first frame reports).

    Args:
        pos_ms: Container position in milliseconds, or ``None`` / ``0`` / ``NaN`` if unavailable.
        frame_idx: Zero-based index of the frame.
        fps: Frame rate used for the fallback (must be positive, see :func:`valid_fps`).
        previous: Time of the previous frame. Timestamps that go backwards (a backend returning
            ``0`` for some frames, bad container timestamps) are held at this value, because the
            fall logic expects non-decreasing time.

    Returns:
        The time in seconds.
    """
    if pos_ms is not None and math.isfinite(pos_ms) and pos_ms > 0:
        t = pos_ms / 1000.0
    else:
        t = frame_idx / fps
    if previous is not None and t < previous:
        return previous
    return t
