"""Tests for the multi-track monitor and alert events."""

from fall_alert.config import FallConfig
from fall_alert.features import PoseFeatures
from fall_alert.monitor import FallMonitor
from fall_alert.state_machine import ALERT, FALLEN, UPRIGHT

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


def test_active_alerts_hide_tracks_not_on_screen():
    cfg = FallConfig()
    m = FallMonitor(cfg)
    run(m, 0.0, 1.0, {1: STANDING})
    run(m, 1.0, 3.5, {1: LYING})
    assert len(m.active_alerts()) == 1
    # the person leaves the frame: the event stays open until lost_seconds, but is not shown
    m.update(4.6, {})
    assert len(m.active_alerts()) == 1
    assert m.active_alerts(4.6, cfg.banner_hold) == []
    assert len(m.active_alerts(3.4, cfg.banner_hold)) == 1
    # still shown within banner_hold of the last observation
    assert len(m.active_alerts(4.4, cfg.banner_hold)) == 1
def run_at(monitor, start, end, observations, positions):
    out = {}
    n = int(round((end - start) / DT))
    for i in range(n):
        out = monitor.update(start + i * DT, observations, positions)
    return out


def test_id_switch_during_fall_keeps_state_and_alert():
    m = FallMonitor(FallConfig())
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (300.0, 400.0)})
    run_at(m, 1.0, 1.5, {1: LYING}, {1: (320.0, 600.0)})
    # the tracker gives the person on the floor a new id a bit further down
    out = run_at(m, 1.6, 3.0, {2: LYING}, {2: (340.0, 640.0)})
    assert out[2].state == ALERT
    assert m.stitched and m.stitched[0][:2] == (1, 2)
    assert len(m.events) == 1 and m.events[0].fall_start == 1.0


def test_far_new_track_is_not_stitched():
    m = FallMonitor(FallConfig())
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (100.0, 400.0)})
    run_at(m, 1.0, 1.5, {2: STANDING}, {2: (1200.0, 400.0)})
    assert m.stitched == []


def test_stitching_can_be_disabled():
    m = FallMonitor(FallConfig(stitch_distance=0))
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (300.0, 400.0)})
    run_at(m, 1.1, 1.5, {2: STANDING}, {2: (300.0, 400.0)})
    assert m.stitched == []


def test_track_lost_longer_than_lost_seconds_is_not_stitched():
    m = FallMonitor(FallConfig(lost_seconds=2.0))
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (300.0, 400.0)})
    run_at(m, 3.5, 4.0, {2: STANDING}, {2: (300.0, 400.0)})
    assert m.stitched == []


def fall_then_lose(monitor, tid, pos, fall_end=1.5):
    """Track ``tid`` stands until 1.0 s, then lies (FALLING/FALLEN) until ``fall_end``."""
    run_at(monitor, 0.0, 1.0, {tid: STANDING}, {tid: pos})
    run_at(monitor, 1.0, fall_end, {tid: LYING}, {tid: pos})


def test_stitch_picks_the_nearest_lost_track():
    m = FallMonitor(FallConfig())
    run_at(m, 0.0, 1.0, {1: STANDING, 2: STANDING}, {1: (100.0, 400.0), 2: (400.0, 400.0)})
    run_at(m, 1.0, 1.5, {1: LYING, 2: LYING}, {1: (100.0, 400.0), 2: (400.0, 400.0)})
    run_at(m, 1.7, 2.0, {3: LYING}, {3: (380.0, 400.0)})
    assert [s[:2] for s in m.stitched] == [(2, 3)]


def test_stitch_distance_is_in_body_heights():
    cfg = FallConfig(stitch_distance=1.0)
    # reference height 500 px: 450 px away is 0.9 heights (stitched), 600 px is 1.2 heights (not)
    near, far = FallMonitor(cfg), FallMonitor(cfg)
    for m, x in ((near, 750.0), (far, 900.0)):
        fall_then_lose(m, 1, (300.0, 400.0))
        run_at(m, 1.7, 2.0, {2: LYING}, {2: (x, 400.0)})
    assert len(near.stitched) == 1 and far.stitched == []


def test_lost_upright_track_is_not_stitched():
    m = FallMonitor(FallConfig())
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (300.0, 400.0)})
    out = run_at(m, 1.2, 1.5, {2: STANDING}, {2: (310.0, 400.0)})
    assert m.stitched == []
    assert out[2].state == UPRIGHT
    # also when the newcomer is already lying: it starts its own machine
    m = FallMonitor(FallConfig())
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (300.0, 400.0)})
    run_at(m, 1.2, 1.5, {2: LYING}, {2: (310.0, 400.0)})
    assert m.stitched == []


def test_lost_falling_or_fallen_track_is_stitched():
    m = FallMonitor(FallConfig())
    fall_then_lose(m, 1, (300.0, 600.0), fall_end=1.3)
    assert m._machines[1].state != UPRIGHT
    out = run_at(m, 1.5, 1.6, {2: LYING}, {2: (320.0, 620.0)})
    assert [s[:2] for s in m.stitched] == [(1, 2)]
    assert out[2].state != UPRIGHT
    m = FallMonitor(FallConfig())
    fall_then_lose(m, 1, (300.0, 600.0), fall_end=2.0)
    assert m._machines[1].state == FALLEN
    run_at(m, 2.2, 2.3, {2: LYING}, {2: (320.0, 620.0)})
    assert [s[:2] for s in m.stitched] == [(1, 2)]


def test_lost_alert_track_is_stitched_to_a_new_track():
    m = FallMonitor(FallConfig())
    fall_then_lose(m, 1, (300.0, 600.0), fall_end=2.5)
    assert len(m.active_alerts()) == 1
    out = run_at(m, 2.7, 3.0, {2: LYING}, {2: (320.0, 620.0)})
    assert out[2].state == ALERT and len(m.events) == 1 and m.events[0].end is None


def test_upright_newcomer_next_to_lost_alert_is_not_stitched_and_alert_closes_as_lost():
    m = FallMonitor(FallConfig(lost_seconds=1.0))
    fall_then_lose(m, 1, (300.0, 600.0), fall_end=2.5)
    out = run_at(m, 2.7, 3.0, {2: STANDING}, {2: (320.0, 620.0)})
    assert m.stitched == [] and out[2].state == UPRIGHT
    run_at(m, 3.0, 4.5, {2: STANDING}, {2: (320.0, 620.0)})
    assert len(m.events) == 1
    assert m.events[0].track_id == 1 and m.events[0].end_reason == "lost"
    assert m.active_alerts() == []


def test_upright_newcomer_next_to_lost_fallen_track_is_not_stitched():
    m = FallMonitor(FallConfig())
    fall_then_lose(m, 1, (300.0, 600.0), fall_end=2.0)
    assert m._machines[1].state == FALLEN
    run_at(m, 2.2, 2.5, {2: STANDING}, {2: (320.0, 620.0)})
    assert m.stitched == []


def test_newcomer_that_is_tilted_or_wide_is_stitched():
    bent = PoseFeatures(torso_angle=45.0, aspect_ratio=0.5, hip_y=600.0, box_height=300.0)
    wide_no_angle = PoseFeatures(torso_angle=None, aspect_ratio=0.9, hip_y=None, box_height=300.0)
    narrow_no_angle = PoseFeatures(torso_angle=None, aspect_ratio=0.35, hip_y=None, box_height=500.0)
    for newcomer, stitched in ((bent, True), (wide_no_angle, True), (narrow_no_angle, False)):
        m = FallMonitor(FallConfig())
        fall_then_lose(m, 1, (300.0, 600.0), fall_end=2.5)
        run_at(m, 2.7, 2.8, {2: newcomer}, {2: (320.0, 620.0)})
        assert bool(m.stitched) is stitched


def test_open_alert_closes_as_lost_when_nothing_stitches():
    m = FallMonitor(FallConfig(lost_seconds=1.0))
    fall_then_lose(m, 1, (300.0, 600.0), fall_end=2.5)
    # far-away newcomer and a nearby one only after the lost window: no stitch
    run_at(m, 2.7, 3.0, {2: STANDING}, {2: (1500.0, 600.0)})
    run_at(m, 3.0, 4.0, {2: STANDING}, {2: (1500.0, 600.0)})
    assert m.stitched == []
    assert len(m.events) == 1
    assert m.events[0].end_reason == "lost" and abs(m.events[0].end - 2.4) < 1e-9
    assert m.active_alerts() == []


def test_track_still_observed_is_not_stitched_to_a_new_one():
    m = FallMonitor(FallConfig())
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (300.0, 400.0)})
    m.update(1.0, {1: STANDING, 2: STANDING}, {1: (300.0, 400.0), 2: (310.0, 400.0)})
    assert m.stitched == []


def test_no_positions_means_no_stitching():
    m = FallMonitor(FallConfig())
    run(m, 0.0, 1.0, {1: STANDING})
    run(m, 1.2, 1.5, {2: STANDING})
    assert m.stitched == []


def test_stitched_alert_is_one_event_and_closes_when_the_new_track_is_lost():
    cfg = FallConfig(lost_seconds=1.0)
    m = FallMonitor(cfg)
    run_at(m, 0.0, 1.0, {1: STANDING}, {1: (300.0, 400.0)})
    run_at(m, 1.0, 2.5, {1: LYING}, {1: (320.0, 600.0)})
    assert len(m.events) == 1
    run_at(m, 2.6, 4.0, {2: LYING}, {2: (330.0, 620.0)})
    run_at(m, 4.0, 6.0, {}, {})
    assert len(m.events) == 1
    event = m.events[0]
    assert event.track_id == 1 and event.end_reason == "lost" and abs(event.end - 3.9) < 1e-9
    assert m.active_alerts() == []


def test_same_id_after_being_dropped_starts_a_new_machine_and_event():
    m = FallMonitor(FallConfig(lost_seconds=1.0, stitch_distance=0))
    run(m, 0.0, 1.0, {4: STANDING})
    run(m, 1.0, 3.0, {4: LYING})
    run(m, 3.0, 5.0, {})
    out = run(m, 5.0, 5.5, {4: STANDING})
    assert out[4].state == UPRIGHT
    run(m, 5.5, 6.5, {4: STANDING})
    run(m, 6.5, 8.5, {4: LYING})
    assert [e.end_reason for e in m.events] == ["lost", None]


def test_second_fall_after_recovery_is_a_second_event():
    m = FallMonitor(FallConfig())
    run(m, 0.0, 1.0, {1: STANDING})
    run(m, 1.0, 3.0, {1: LYING})
    run(m, 3.0, 6.0, {1: STANDING})
    run(m, 6.0, 8.0, {1: LYING})
    assert [e.end_reason for e in m.events] == ["recovered", None]


def test_empty_video_has_no_events():
    m = FallMonitor(FallConfig())
    assert m.update(0.0, {}) == {}
    assert m.finish() == [] and m.active_alerts() == []


def test_open_event_serializes_with_null_end():
    m = FallMonitor(FallConfig())
    run(m, 0.0, 1.0, {1: STANDING})
    run(m, 1.0, 3.0, {1: LYING})
    d = m.events[0].to_dict()
    assert d["end"] is None and d["end_reason"] is None
    assert d["fall_start"] == 1.0 and d["start"] == 2.0


def test_active_alerts_max_age_needs_a_time():
    m = FallMonitor(FallConfig())
    run(m, 0.0, 1.0, {1: STANDING})
    run(m, 1.0, 3.0, {1: LYING})
    assert len(m.active_alerts(max_age=0.3)) == 1
    assert len(m.active_alerts(t=2.95, max_age=0.3)) == 1


def _stitched_with(newcomer):
    m = FallMonitor(FallConfig())
    fall_then_lose(m, 1, (300.0, 600.0), fall_end=2.5)
    run_at(m, 2.7, 2.8, {2: newcomer}, {2: (320.0, 620.0)})
    return bool(m.stitched)


def test_narrow_newcomer_below_falling_angle_is_not_stitched_onto_lost_alert():
    standing_38 = PoseFeatures(torso_angle=38.9, aspect_ratio=0.48, hip_y=300.0, box_height=500.0)
    assert not _stitched_with(standing_38)


def test_tilted_newcomer_at_or_above_falling_angle_is_stitched():
    tilted = PoseFeatures(torso_angle=50.0, aspect_ratio=0.5, hip_y=600.0, box_height=300.0)
    assert _stitched_with(tilted)


def test_wide_newcomer_with_small_angle_is_stitched():
    wide = PoseFeatures(torso_angle=10.0, aspect_ratio=0.9, hip_y=600.0, box_height=300.0)
    assert _stitched_with(wide)


def test_clearly_upright_newcomer_is_still_not_stitched():
    assert not _stitched_with(STANDING)


def test_narrow_newcomer_without_angle_is_not_stitched():
    no_angle = PoseFeatures(torso_angle=None, aspect_ratio=0.48, hip_y=None, box_height=500.0)
    assert not _stitched_with(no_angle)
