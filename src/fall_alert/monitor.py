"""Multi-track fall monitor and alert event log (standard library only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Hashable, List, Mapping, Optional

from fall_alert.config import FallConfig
from fall_alert.features import PoseFeatures
from fall_alert.state_machine import ALERT, FallStateMachine


@dataclass
class AlertEvent:
    """One alert of one track.

    Attributes:
        track_id: Tracker id of the person.
        fall_start: Time (s) the track entered ``FALLING``.
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

    def update(self, t: float, observations: Mapping[Hashable, PoseFeatures]) -> Dict[Hashable, TrackStatus]:
        """Updates every observed track at time ``t`` and drops lost tracks.

        Args:
            t: Video time in seconds.
            observations: Features per track id in the current frame.

        Returns:
            Status per observed track id.
        """
        out: Dict[Hashable, TrackStatus] = {}
        for track_id, f in observations.items():
            machine = self._machines.get(track_id)
            if machine is None:
                machine = self._machines[track_id] = FallStateMachine(self.cfg)
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
        return out

    def finish(self) -> List[AlertEvent]:
        """Closes alerts still open at the end of the video and returns all events."""
        for track_id in list(self._open):
            self._close(track_id, self._last_seen[track_id], "end_of_video")
        return self.events

    def active_alerts(self) -> List[AlertEvent]:
        """Currently open alert events."""
        return list(self._open.values())

    def _close(self, track_id: Hashable, t: float, reason: str) -> None:
        event = self._open.pop(track_id)
        event.end, event.end_reason = t, reason
