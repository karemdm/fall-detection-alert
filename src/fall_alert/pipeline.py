"""Video loop: pose tracking, fall monitoring, rendering and event export."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from typing import Dict, Hashable, List, Tuple

import cv2

from fall_alert.config import RunConfig
from fall_alert.features import Box, Keypoint, compute_features
from fall_alert.monitor import FallMonitor
from fall_alert.rendering import render

logger = logging.getLogger(__name__)

WINDOW_NAME = "Fall Detection Alert"
TRACE_FIELDS = ["time", "track_id", "state", "torso_angle", "aspect_ratio", "velocity", "box_height"]

Person = Tuple[Box, List[Keypoint]]


def extract_persons(result) -> Dict[Hashable, Person]:
    """Converts an Ultralytics pose tracking result into ``{track_id: (box, keypoints)}``.

    Detections without a track id (not yet confirmed by the tracker) are skipped.
    """
    boxes = result.boxes
    if boxes is None or boxes.id is None or result.keypoints is None:
        return {}
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    ids = boxes.id.int().cpu().numpy().tolist()
    kxy = result.keypoints.xy.cpu().numpy().tolist()
    kconf = result.keypoints.conf
    kconf = kconf.cpu().numpy().tolist() if kconf is not None else [[1.0] * len(k) for k in kxy]
    persons: Dict[Hashable, Person] = {}
    for box, track_id, pts, confs in zip(xyxy, ids, kxy, kconf):
        keypoints = [(p[0], p[1], c) for p, c in zip(pts, confs)]
        persons[int(track_id)] = ((box[0], box[1], box[2], box[3]), keypoints)
    return persons


def _fmt(value) -> str:
    return "" if value is None else f"{value:.4f}"


def run(cfg: RunConfig) -> int:
    """Processes the input video and writes the annotated video and optional files.

    Args:
        cfg: Run configuration.

    Returns:
        Process exit code (``0`` on success).
    """
    from ultralytics import YOLO

    cap = cv2.VideoCapture(cfg.input_path)
    if not cap.isOpened():
        logger.error("Cannot open input video: %s", cfg.input_path)
        return 1
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Input: %dx%d @ %.2f FPS, %d frames", width, height, fps, total)

    for path in (cfg.output_path, cfg.events_json, cfg.trace_csv):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(cfg.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    logger.info("Loading pose model %s", cfg.model)
    model = YOLO(cfg.model)
    monitor = FallMonitor(cfg.fall)
    trace_file = open(cfg.trace_csv, "w", newline="", encoding="utf-8") if cfg.trace_csv else None
    trace = csv.writer(trace_file) if trace_file else None
    if trace:
        trace.writerow(TRACE_FIELDS)
    if cfg.display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx, start = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else frame_idx / fps
            frame_idx += 1
            result = model.track(
                frame, persist=True, conf=cfg.conf, imgsz=cfg.imgsz, tracker=cfg.tracker, classes=[0], verbose=False
            )[0]
            persons = extract_persons(result)
            features = {tid: compute_features(kps, box, cfg.fall.min_kp_conf) for tid, (box, kps) in persons.items()}
            statuses = monitor.update(t, features)
            for event in monitor.events:
                if event.start == t:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    logger.info("ALERT track #%s at %.2f s (falling since %.2f s)", event.track_id, t, event.fall_start)
            if trace:
                for tid, st in statuses.items():
                    f = st.features
                    trace.writerow([f"{t:.3f}", tid, st.state, _fmt(f.torso_angle), _fmt(f.aspect_ratio),
                                    _fmt(st.velocity), f"{f.box_height:.1f}"])
            render(frame, persons, statuses, monitor.active_alerts(), t, cfg.fall.min_kp_conf)
            writer.write(frame)

            elapsed = time.time() - start
            sys.stdout.write(f"\rFrame {frame_idx}/{total} | {frame_idx / max(elapsed, 1e-6):.1f} FPS")
            sys.stdout.flush()
            if cfg.display:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopped by user")
                    break
    finally:
        sys.stdout.write("\n")
        cap.release()
        writer.release()
        if trace_file:
            trace_file.close()
        if cfg.display:
            cv2.destroyAllWindows()

    events = monitor.finish()
    logger.info("Output saved to %s (%d alert event(s))", cfg.output_path, len(events))
    for e in events:
        logger.info("  track #%s: fall %.2f s, alert %.2f-%.2f s (%s)", e.track_id, e.fall_start, e.start, e.end,
                    e.end_reason)
    if cfg.events_json:
        payload = {"input": os.path.basename(cfg.input_path), "events": [e.to_dict() for e in events]}
        with open(cfg.events_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Events saved to %s", cfg.events_json)
    return 0
