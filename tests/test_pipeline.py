"""Tests for the pipeline helpers that need no model (``cv2`` is required to import the module)."""

import pytest

pytest.importorskip("cv2")

from fall_alert.config import RunConfig  # noqa: E402
from fall_alert.pipeline import _fmt, extract_persons, run  # noqa: E402


class FakeTensor:
    """Minimal stand-in for a torch tensor: ``.cpu().numpy().tolist()`` and ``.int()``."""

    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return self

    def int(self):
        return FakeTensor([int(v) for v in self.data])

    def tolist(self):
        return self.data


class FakeBoxes:
    def __init__(self, xyxy, ids):
        self.xyxy = FakeTensor(xyxy)
        self.id = None if ids is None else FakeTensor(ids)


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = FakeTensor(xy)
        self.conf = None if conf is None else FakeTensor(conf)


class FakeResult:
    def __init__(self, boxes, keypoints):
        self.boxes = boxes
        self.keypoints = keypoints


def two_people():
    xyxy = [[10.0, 20.0, 60.0, 220.0], [100.0, 40.0, 300.0, 120.0]]
    xy = [[[11.0, 21.0]] * 17, [[101.0, 41.0]] * 17]
    conf = [[0.9] * 17, [0.2] * 17]
    return xyxy, xy, conf


def test_extract_persons_maps_track_id_to_box_and_keypoints():
    xyxy, xy, conf = two_people()
    persons = extract_persons(FakeResult(FakeBoxes(xyxy, [3.0, 8.0]), FakeKeypoints(xy, conf)))
    assert set(persons) == {3, 8}
    box, kps = persons[3]
    assert box == (10.0, 20.0, 60.0, 220.0)
    assert len(kps) == 17 and kps[0] == (11.0, 21.0, 0.9)
    assert persons[8][1][5] == (101.0, 41.0, 0.2)


def test_extract_persons_without_track_ids_or_keypoints_is_empty():
    xyxy, xy, conf = two_people()
    assert extract_persons(FakeResult(FakeBoxes(xyxy, None), FakeKeypoints(xy, conf))) == {}
    assert extract_persons(FakeResult(FakeBoxes(xyxy, [1, 2]), None)) == {}
    assert extract_persons(FakeResult(None, FakeKeypoints(xy, conf))) == {}


def test_extract_persons_without_keypoint_confidence_assumes_visible():
    xyxy, xy, _ = two_people()
    persons = extract_persons(FakeResult(FakeBoxes(xyxy, [1, 2]), FakeKeypoints(xy, None)))
    assert persons[1][1][0][2] == 1.0


def test_fmt_formats_numbers_and_missing_values():
    assert _fmt(None) == ""
    assert _fmt(12.34567) == "12.3457"
    assert _fmt(float("inf")) == "inf"


def test_run_returns_1_for_missing_input_without_loading_a_model(tmp_path):
    cfg = RunConfig(input_path=str(tmp_path / "missing.mp4"), output_path=str(tmp_path / "out.mp4"))
    assert run(cfg) == 1
    assert not (tmp_path / "out.mp4").exists()
