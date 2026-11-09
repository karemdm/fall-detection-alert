"""Tests for the multi-track monitor and alert events."""

from fall_alert.config import FallConfig
from fall_alert.features import PoseFeatures
from fall_alert.monitor import FallMonitor
from fall_alert.state_machine import ALERT, UPRIGHT

STANDING = PoseFeatures(torso_angle=3.0, aspect_ratio=0.35, hip_y=300.0, box_height=500.0)
LYING = PoseFeatures(torso_angle=88.0, aspect_ratio=2.5, hip_y=650.0, box_height=180.0)
DT = 0.1


def run(monitor, start, end, observations):
    out = {}
    n = int(round((end - start) / DT))
    for i in range(n):
        out = monitor.update(start + i * DT, observations)
    return out


def test_event_opened_and_closed_on_recovery():
    m = FallMonitor(FallConfig(fallen_seconds=1.0, recover_seconds=2.0))
    run(m, 0.0, 1.0, {7: STANDING})
    out = run(m, 1.0, 2.5, {7: LYING})
    assert out[7].state == ALERT
    assert len(m.active_alerts()) == 1
    event = m.events[0]
    assert event.track_id == 7 and event.fall_start == 1.0
    assert abs(event.start - 2.0) < 1e-9 and event.end is None
    out = run(m, 2.5, 5.0, {7: STANDING})
    assert out[7].state == UPRIGHT
    assert event.end is not None and event.end_reason == "recovered"
    assert m.active_alerts() == []


def test_only_fallen_track_alerts():
    m = FallMonitor(FallConfig())
    run(m, 0.0, 1.0, {1: STANDING, 2: STANDING})
    out = run(m, 1.0, 3.0, {1: STANDING, 2: LYING})
    assert out[1].state == UPRIGHT and out[2].state == ALERT
    assert [e.track_id for e in m.events] == [2]


def test_lost_track_closes_event_at_last_seen():
    m = FallMonitor(FallConfig(lost_seconds=1.0))
    run(m, 0.0, 1.0, {3: STANDING})
    run(m, 1.0, 2.5, {3: LYING})
    last_seen = 2.4
    run(m, 2.5, 4.0, {})
    event = m.events[0]
    assert event.end_reason == "lost" and abs(event.end - last_seen) < 1e-9
    assert m.active_alerts() == []


def test_finish_closes_open_events_and_serializes():
    m = FallMonitor(FallConfig())
    run(m, 0.0, 1.0, {5: STANDING})
    run(m, 1.0, 3.0, {5: LYING})
    events = m.finish()
    assert len(events) == 1 and events[0].end_reason == "end_of_video"
    d = events[0].to_dict()
    assert set(d) == {"track_id", "fall_start", "start", "end", "end_reason"}
    assert d["end"] == round(2.9, 3)
