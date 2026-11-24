"""Multi-track fall monitor and alert event log (standard library only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

from fall_alert.config import FallConfig
from fall_alert.features import PoseFeatures
from fall_alert.state_machine import ALERT, UPRIGHT, FallStateMachine


def _is_stitch_newcomer_other_person(f: PoseFeatures, cfg: FallConfig) -> bool:
    """True when a new track is narrow and below the falling angle, so it is not a stitch candidate.

    A stitch repairs a box that changes shape in the middle of a fall, so the newcomer is expected
    to be tilted (torso angle at or above ``falling_angle``) or wide (aspect above
    ``upright_aspect``). A narrow newcomer below the falling angle is a different person: after a
    hard cut in a compilation, a standing person at 38.9 degrees and aspect 0.48 inherited the
    previous person's open ``ALERT``, and the next fall got no event of its own. A missing angle
    counts as not tilted.
    """
    not_tilted = f.torso_angle is None or f.torso_angle < cfg.falling_angle
    return f.aspect_ratio <= cfg.upright_aspect and not_tilted


@dataclass
class AlertEvent:
    """One alert of one track.

    Attributes:
        track_id: Tracker id of the person.
        fall_start: Time (s) the track first entered ``FALLING`` in this fall episode.
        start: Time (s) the track entered ``ALERT``.
        end: Time (s) the alert ended (recovery, track lost or end of video); ``None`` while open.
        end_reason: ``"recovered"``, ``"lost"`` or ``"end_of_video"``.
    """

    track_id: Hashable
    fall_start: Optional[float]
    start: float
    end: Optional[float] = None
    end_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-friendly dict with times rounded to milliseconds."""
        d = asdict(self)
        for key in ("fall_start", "start", "end"):
            if d[key] is not None:
                d[key] = round(d[key], 3)
        return d


@dataclass(frozen=True)
class TrackStatus:
    """State of one track after an update (for rendering)."""

    state: str
    features: PoseFeatures
    velocity: Optional[float]


class FallMonitor:
    """Keeps a :class:`FallStateMachine` per track id and logs alert events.

    Args:
        cfg: Thresholds.
    """

    def __init__(self, cfg: FallConfig) -> None:
        self.cfg = cfg
        self.events: List[AlertEvent] = []
        self._machines: Dict[Hashable, FallStateMachine] = {}
        self._last_seen: Dict[Hashable, float] = {}
        self._open: Dict[Hashable, AlertEvent] = {}
        self._last_pos: Dict[Hashable, Tuple[float, float]] = {}
        self.stitched: List[Tuple[Hashable, Hashable, float]] = []

    def update(
        self,
        t: float,
        observations: Mapping[Hashable, PoseFeatures],
        positions: Optional[Mapping[Hashable, Tuple[float, float]]] = None,
    ) -> Dict[Hashable, TrackStatus]:
        """Updates every observed track at time ``t`` and drops lost tracks.

        Args:
            t: Video time in seconds.
            observations: Features per track id in the current frame.
            positions: Optional box center per track id; enables stitching of a new track to a
                recently lost one (see ``FallConfig.stitch_distance``).

        Returns:
            Status per observed track id.
        """
        out: Dict[Hashable, TrackStatus] = {}
        for track_id, f in observations.items():
            machine = self._machines.get(track_id)
            if machine is None and positions is not None and track_id in positions:
                machine = self._stitch(t, track_id, f, positions[track_id], observations)
            if machine is None:
                machine = self._machines[track_id] = FallStateMachine(self.cfg)
            if positions is not None and track_id in positions:
                self._last_pos[track_id] = positions[track_id]
            state = machine.update(t, f)
            self._last_seen[track_id] = t
            if state == ALERT and track_id not in self._open:
                event = AlertEvent(track_id, machine.fall_start, t)
                self._open[track_id] = event
                self.events.append(event)
            elif state != ALERT and track_id in self._open:
                self._close(track_id, t, "recovered")
            out[track_id] = TrackStatus(state, f, machine.velocity)
        for track_id in [k for k, seen in self._last_seen.items() if t - seen > self.cfg.lost_seconds]:
            if track_id in self._open:
                self._close(track_id, self._last_seen[track_id], "lost")
            del self._machines[track_id], self._last_seen[track_id]
            self._last_pos.pop(track_id, None)
        return out

    def _stitch(self, t, track_id, f, pos, observations) -> Optional[FallStateMachine]:
        """Moves the nearest recently lost track (state machine and open alert) to ``track_id``.

        Candidates are tracks not observed in this frame, last seen less than ``lost_seconds``
        ago, and not ``UPRIGHT`` when last seen (``FALLING``, ``FALLEN`` or ``ALERT``): the repair
        is for the box changing shape during a fall, so a new track near a lost upright person is
        treated as a new person and starts in ``UPRIGHT``. The distance between box centers is
        divided by the larger of the old reference height and the new box height (at least 1 px)
        and must not exceed ``stitch_distance``.

        The new track must also be tilted or wide at its first observation (see
        :func:`_is_stitch_newcomer_other_person`: torso angle at or above ``falling_angle`` or box
        aspect above ``upright_aspect``; a missing angle counts as not tilted). A narrow newcomer
        below the falling angle next to a lost ``FALLEN``/``ALERT`` track is someone else, not the
        fallen person. Without a stitch the old alert closes as ``lost`` after ``lost_seconds`` and
        the new track starts unarmed in ``UPRIGHT``.

        Returns:
            The taken-over state machine, or ``None`` if no track qualifies.
        """
        if self.cfg.stitch_distance <= 0 or _is_stitch_newcomer_other_person(f, self.cfg):
            return None
        best, best_d = None, None
        for old_id, seen in self._last_seen.items():
            if old_id in observations or seen >= t or t - seen > self.cfg.lost_seconds or old_id not in self._last_pos:
                continue
            if self._machines[old_id].state == UPRIGHT:
                continue
            ref = max(self._machines[old_id].motion.reference_height(), f.box_height, 1.0)
            (x0, y0), (x1, y1) = self._last_pos[old_id], pos
            d = math.hypot(x1 - x0, y1 - y0) / ref
            if d <= self.cfg.stitch_distance and (best_d is None or d < best_d):
                best, best_d = old_id, d
        if best is None:
            return None
        machine = self._machines.pop(best)
        self._machines[track_id] = machine
        self._last_seen.pop(best)
        self._last_pos.pop(best, None)
        if best in self._open:
            self._open[track_id] = self._open.pop(best)
        self.stitched.append((best, track_id, t))
        return machine

    def finish(self) -> List[AlertEvent]:
        """Closes alerts still open at the end of the video and returns all events."""
        for track_id in list(self._open):
            self._close(track_id, self._last_seen[track_id], "end_of_video")
        return self.events

    def active_alerts(self, t: Optional[float] = None, max_age: Optional[float] = None) -> List[AlertEvent]:
        """Currently open alert events.

        Args:
            t: Current video time; needed with ``max_age``.
            max_age: If given (the pipeline passes ``banner_hold``), only alerts whose track was seen within the last ``max_age`` seconds
                (an alert stays open in the event log until the track is lost for ``lost_seconds``,
                but its person is no longer on screen, so it should not be shown).
        """
        if max_age is None or t is None:
            return list(self._open.values())
        return [e for tid, e in self._open.items() if t - self._last_seen[tid] <= max_age]

    def _close(self, track_id: Hashable, t: float, reason: str) -> None:
        """Ends the open alert of ``track_id`` at time ``t`` with ``reason``."""
        event = self._open.pop(track_id)
        event.end, event.end_reason = t, reason
