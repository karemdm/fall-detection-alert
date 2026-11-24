"""Per-track fall state machine (standard library only).

Timestamps are video time in seconds and are always passed in by the caller,
so the logic is deterministic and easy to test.

States::

    UPRIGHT --(fast downward hip motion | torso angle >= falling_angle | lying)--> FALLING
    FALLING --(lying posture)--> FALLEN
    FALLING --(no lying within falling_timeout)--> UPRIGHT
    FALLEN  --(lying held for fallen_seconds)--> ALERT
    FALLEN  --(not lying for longer than lying_gap)--> FALLING
    ALERT   --(upright held for recover_seconds)--> UPRIGHT

A track can leave ``UPRIGHT`` only after it has been "armed" by one frame with at
least ``min_pose_keypoints`` confident keypoints. Objects detected as a person
(bins, coats, shadows) rarely show a body pose, so they cannot raise alerts.

``fall_start`` is the time of the first entry into ``FALLING`` of an episode (an episode ends when
the track is back in ``UPRIGHT``). A ``FALLEN -> FALLING`` drop does not move it, so it keeps
the real onset of the fall; the ``falling_timeout`` counts from the latest entry into ``FALLING``.
"""

from __future__ import annotations

from collections import deque
import statistics
from typing import Deque, Optional, Tuple

from fall_alert.config import FallConfig
from fall_alert.features import PoseFeatures

UPRIGHT = "UPRIGHT"
FALLING = "FALLING"
FALLEN = "FALLEN"
ALERT = "ALERT"
STATES = (UPRIGHT, FALLING, FALLEN, ALERT)


def is_lying(f: PoseFeatures, cfg: FallConfig) -> bool:
    """Lying posture: torso close to horizontal, or a wide box."""
    angle_lying = f.torso_angle is not None and f.torso_angle >= cfg.lying_angle
    return angle_lying or f.aspect_ratio >= cfg.lying_aspect


def is_upright(f: PoseFeatures, cfg: FallConfig) -> bool:
    """Upright posture: narrow box and (if measurable) torso close to vertical."""
    angle_ok = f.torso_angle is None or f.torso_angle <= cfg.upright_angle
    return angle_ok and f.aspect_ratio <= cfg.upright_aspect


class MotionHistory:
    """Short history of hip height and box height for one track.

    Args:
        cfg: Thresholds (uses ``velocity_window`` and ``height_window``).
    """

    def __init__(self, cfg: FallConfig) -> None:
        self.cfg = cfg
        self._hips: Deque[Tuple[float, float]] = deque()
        self._heights: Deque[Tuple[float, float]] = deque()

    def update(self, t: float, f: PoseFeatures) -> None:
        """Adds the observation at time ``t`` and drops samples outside the windows."""
        if f.hip_y is not None:
            self._hips.append((t, f.hip_y))
        self._heights.append((t, f.box_height))
        # keep the last sample before the window as an anchor, so a slow video (one frame per
        # window) still has two samples; drop the oldest only when the next one is also outside
        while len(self._hips) >= 2 and t - self._hips[1][0] > self.cfg.velocity_window:
            self._hips.popleft()
        while self._heights and t - self._heights[0][0] > self.cfg.height_window:
            self._heights.popleft()

    def reference_height(self) -> float:
        """Maximum box height within ``height_window`` (pre-fall body height proxy)."""
        return max((h for _, h in self._heights), default=0.0)

    def velocity(self) -> Optional[float]:
        """Downward hip velocity in reference heights per second (``None`` if unknown).

        The slope of the hip height is the median of all pairwise slopes (Theil-Sen) of the
        samples inside ``velocity_window`` plus one anchor sample just before the window (used only
        if it is at most twice the window before the newest sample), so a single noisy keypoint
        frame does not decide the value. The samples must span at least half of the window. The
        anchor makes the value available down to about ``1 / (2 * velocity_window)`` frames per
        second (about 1.7 FPS for the default 0.3 s window).
        """
        samples = list(self._hips)
        if len(samples) >= 2 and samples[-1][0] - samples[0][0] > 2.0 * self.cfg.velocity_window:
            samples = samples[1:]
        if len(samples) < 2:
            return None
        ref = self.reference_height()
        dt = samples[-1][0] - samples[0][0]
        if dt <= 0 or dt < 0.5 * self.cfg.velocity_window or ref <= 0:
            return None
        slopes = [
            (y1 - y0) / (t1 - t0)
            for i, (t0, y0) in enumerate(samples)
            for t1, y1 in samples[i + 1:]
            if t1 > t0
        ]
        if not slopes:
            return None
        return statistics.median(slopes) / ref


class FallStateMachine:
    """Fall state of one track.

    Args:
        cfg: Thresholds.
    """

    def __init__(self, cfg: FallConfig) -> None:
        self.cfg = cfg
        self.state = UPRIGHT
        self.motion = MotionHistory(cfg)
        self.velocity: Optional[float] = None
        self.fall_start: Optional[float] = None  # first entry into FALLING of this episode
        self._falling_since: Optional[float] = None  # latest entry into FALLING (timeout)
        self._lying_since: Optional[float] = None
        self._last_lying: Optional[float] = None
        self._upright_since: Optional[float] = None
        self.armed = cfg.min_pose_keypoints <= 0

    def update(self, t: float, f: PoseFeatures) -> str:
        """Advances the state machine with the features observed at time ``t``.

        Args:
            t: Video time in seconds (non-decreasing).
            f: Features of this track in the current frame.

        Returns:
            The new state.
        """
        cfg = self.cfg
        self.motion.update(t, f)
        self.velocity = self.motion.velocity()
        lying = is_lying(f, cfg)
        if not self.armed and f.visible_keypoints >= cfg.min_pose_keypoints:
            self.armed = True
        if lying:
            self._last_lying = t
        if is_upright(f, cfg):
            self._upright_since = t if self._upright_since is None else self._upright_since
        else:
            self._upright_since = None
        upright_held = self._upright_since is not None and t - self._upright_since >= cfg.recover_seconds

        if self.state == UPRIGHT:
            fast = self.velocity is not None and self.velocity >= cfg.fall_velocity
            tilted = f.torso_angle is not None and f.torso_angle >= cfg.falling_angle
            if self.armed and (fast or tilted or lying):
                self.state, self.fall_start, self._falling_since = FALLING, t, t
        if self.state == FALLING:
            if lying:
                self.state, self._lying_since = FALLEN, t
            elif t - self._falling_since >= cfg.falling_timeout:
                self.state = UPRIGHT
        elif self.state == ALERT and upright_held:
            self.state = UPRIGHT
        elif self.state == FALLEN:
            if not lying and t - self._last_lying > cfg.lying_gap:
                # fall_start is kept: it is the start of the whole fall episode
                self.state, self._falling_since = FALLING, t
        if self.state == FALLEN and t - self._lying_since >= cfg.fallen_seconds:
            self.state = ALERT
        if self.state == UPRIGHT:
            self.fall_start = self._falling_since = self._lying_since = None
        return self.state
