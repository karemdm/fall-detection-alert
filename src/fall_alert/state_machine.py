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
"""

from __future__ import annotations

from collections import deque
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
        while self._hips and t - self._hips[0][0] > self.cfg.velocity_window:
            self._hips.popleft()
        while self._heights and t - self._heights[0][0] > self.cfg.height_window:
            self._heights.popleft()

    def reference_height(self) -> float:
        """Maximum box height within ``height_window`` (pre-fall body height proxy)."""
        return max((h for _, h in self._heights), default=0.0)

    def velocity(self) -> Optional[float]:
        """Downward hip velocity in reference heights per second (``None`` if unknown).

        Needs samples spanning at least half of ``velocity_window``.
        """
        if len(self._hips) < 2:
            return None
        (t0, y0), (t1, y1) = self._hips[0], self._hips[-1]
        ref = self.reference_height()
        if t1 - t0 < 0.5 * self.cfg.velocity_window or ref <= 0:
            return None
        return (y1 - y0) / (t1 - t0) / ref


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
        self.fall_start: Optional[float] = None
        self._lying_since: Optional[float] = None
        self._last_lying: Optional[float] = None
        self._upright_since: Optional[float] = None

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
            if fast or tilted or lying:
                self.state, self.fall_start = FALLING, t
        if self.state == FALLING:
            if lying:
                self.state, self._lying_since = FALLEN, t
            elif t - self.fall_start >= cfg.falling_timeout:
                self.state = UPRIGHT
        elif self.state == ALERT and upright_held:
            self.state = UPRIGHT
        elif self.state == FALLEN:
            if not lying and t - self._last_lying > cfg.lying_gap:
                self.state, self.fall_start = FALLING, t
        if self.state == FALLEN and t - self._lying_since >= cfg.fallen_seconds:
            self.state = ALERT
        if self.state == UPRIGHT:
            self.fall_start = self._lying_since = None
        return self.state
