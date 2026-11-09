# Results

Annotated videos produced by this project, re-encoded to H.264 720p without audio.

- **Fall mix** (`fall_mix_alert.mp4`): current defaults (`weights/yolo11n-pose.pt`, `--conf 0.30`, `--imgsz 1280`,
  ByteTrack, default `FallConfig`; upright tracks with an `n/a` torso angle are not drawn).
- **Single clips** (below): produced with the previous default `--conf 0.35`, `imgsz=640`, before the `n/a` boxes
  were hidden; re-running them with the current defaults can give slightly different times.

- `gmdcsa24-s1-fall-05_alert.mp4` + `gmdcsa24-s1-fall-05_events.json`: walking, then a forward fall to hands and
  knees and lying on the floor.
- `gmdcsa24-s1-fall-11_alert.mp4` + `gmdcsa24-s1-fall-11_events.json`: standing facing the camera, then a backward
  fall ending on the side.
- `snow-pexels-20595440_alert.mp4` + `snow-pexels-20595440_events.json`: outdoor, man kicks a tree, falls backward
  and lies in deep snow; the clip ends ~1 s after he lands (**no alert**, see below).
- `fall_mix_alert.mp4` + `fall_mix_events.json`: the 55.2 s fall mix (eight GMDCSA24 falls of three subjects plus
  the snow fall), run with **`--imgsz 1280`**; see [Fall mix](#fall-mix) below.

Reproduce (from the repo root):

```bash
fall-alert -i samples/fall_mix_input.mp4 -o outputs/fall_mix_alert.mp4 --events-json results/fall_mix_events.json --trace-csv outputs/fall_mix_trace.csv --imgsz 1280 --no-display
fall-alert -i samples/gmdcsa24-s1-fall-05.mp4 -o outputs/fall-05.mp4 --events-json results/gmdcsa24-s1-fall-05_events.json --no-display
fall-alert -i samples/gmdcsa24-s1-fall-11.mp4 -o outputs/fall-11.mp4 --events-json results/gmdcsa24-s1-fall-11_events.json --no-display
fall-alert -i samples/snow-pexels-20595440.mp4 -o outputs/snow-20595440.mp4 --events-json results/snow-pexels-20595440_events.json --no-display
```

## Fall mix

Input: [`../samples/fall_mix_input.mp4`](../samples/README.md#fall-mix-fall_mix_inputmp4) (segment table, sources
and licenses there). Default thresholds, `weights/yolo11n-pose.pt`, ByteTrack; nothing was tuned on the mix. The
mix was run twice, at the default `--imgsz 640` and at `--imgsz 1280`; the published video and events are the
**1280** run because it missed fewer falls (8/9 vs 6/9 hits). Times are mix seconds, from `fall_mix_events.json`
and the `--trace-csv` output.

Rule used: **hit** if an `ALERT` starts inside `[labelled fall start, labelled fall end + 2 s]` of the segment
(clipped to the segment end), **miss** otherwise; a **false alert** is an `ALERT` starting outside every such
window.

Columns "(1280)" are the current defaults (`--conf 0.30`, published video); "(640)" is the earlier run at `--conf 0.35`.

| # | Segment | Labelled fall (mix s) | `FALLING` (1280) | `ALERT` (1280) | Result 1280 | `ALERT` (640) | Result 640 |
|---|---|---|---|---|---|---|---|
| 1 | S1 05, walking, right side fall | 1.80–5.00 | 2.20 | 3.27 (+1.47 s) | hit | 3.30 | hit |
| 2 | S2 02, walking, right side fall | 7.87–13.17 | 8.33 | 11.87 (+4.00 s) | hit (late) | none | miss |
| 3 | S3 04, walking, backward fall | 14.80–18.60 | 15.07 | 18.83 (+4.03 s) | hit (late) | none | miss |
| 4 | S2 17, walking, forward fall | 21.07–24.57 | 21.70 | 22.73 (+1.66 s) | hit | 22.77 | hit |
| 5 | S1 11, standing, backward fall | 26.07–31.07 | 27.13 (`FALLEN`) | 28.13 (+2.06 s) | hit | 28.23 | hit |
| 6 | S2 13, night, backward fall (side view) | 33.17–38.67 | 34.37 | 35.67 (+2.50 s) | hit | 35.57 | hit |
| 7 | S3 16, night, walking, left side fall | 41.40–45.67 | 41.57 | 44.00 (+2.60 s) | hit | none | miss |
| 8 | S2 23, standing, sideways fall | 47.67–51.07 | 48.27 | 49.60 (+1.93 s) | hit | 49.43 | hit |
| 9 | Pexels snow, backward fall (visual estimate) | ~53.60–54.30 | 53.23 | 55.13 (+1.53 s), 0.1 s before the end | hit | 55.20 (last frame) | hit |

Summary (own measurement on one compilation, not a validation and not an accuracy figure):

| `--imgsz` | Hits | Misses | False alerts |
|---|---|---|---|
| 1280 (published) | 8 / 9 | 1 (segment 3) | 0 |
| 640 (default) | 6 / 9 | 3 (segments 2, 3, 7) | 0 |

Observations (from the trace and frames of the annotated videos):

- **Misses are detection losses on the floor, not rule failures.** In segments 2, 3 and 7 the person was tracked
  `UPRIGHT` → `FALLING` (torso angle crossing 45°), but once lying on the patterned floor mat YOLO11n-pose
  detected the body only in short bursts. Every new detection started a new track id, so the 1.0 s
  `fallen_seconds` confirmation restarted again and again. At 1280 the detections on the floor were stable enough
  in segments 2 and 7; in segment 3 (lying on the back across the frame, arms spread) the body was
  detected only for a few frames at 18.03–19.00 s, too late and too short for the confirmation before the cut.
- Segment 2 at 1280 is a **marginal** hit: after several id switches (#5 → #11 → #16) track #16 reached `ALERT`
  at 12.10 s and was not detected again, so the alert lasted one frame in the events (the banner stays up to
  `lost_seconds` = 2 s in the video).
- Segment 9 alerted at 1280 and at 640, unlike the separate snow clip at 640 (no alert, see below). The mix
  resamples the 60 fps clip to 30 fps and pads it into a 1280x720 frame, which changed the tracking (no id switch
  on the snow); the alert fires only 0.1 s before the end of the mix, so this is a fragile result.
- **No false alerts.** Non-fall postures that briefly entered `FALLING` / `FALLEN` (bending over at the start of
  segments 6 and 8 at 640) returned to `UPRIGHT` without an alert. The hanging clothes on the left of the Subject 2
  / Subject 3 room were sometimes detected as an extra `UPRIGHT` person; these tracks never left `UPRIGHT`.
- **Cuts between segments.** The previous person's track is only dropped after `lost_seconds` (2 s) without a
  detection, so the red banner of the last alert stays on for up to ~2 s into the next segment (visible in the demo
  GIF), and at 31.20–31.33 s the alerting track #30 of segment 5 was briefly matched to the new person of
  segment 6. These are artefacts of the hard cuts, not new alert events.
- Delays from the labelled fall start to `ALERT` were 1.5–2.6 s on the clean hits; they include the time to reach
  the floor and the 1.0 s confirmation by design.

The demo GIF (`docs/assets/demo.gif`) is cut from this video: mix time 21.0–29.8 s (segments 4 and 5, 8 fps,
720 px wide).

## What was measured

Own measurement on **two clips of one subject**. This is a sanity check of the pipeline, not a validation and not
an accuracy figure. Times are video seconds; labels come from the dataset CSV.

| Clip | Labelled | Last `UPRIGHT` frame | `FALLING` | `FALLEN` | `ALERT` | Alert after label start |
|---|---|---|---|---|---|---|
| fall-05 | walking 0–1.8 s, falling 1.8–5.0 s | 2.24 s | 2.27 s | 2.30 s | 3.31 s | +1.51 s |
| fall-11 | standing 0–1.0 s, falling 1.0–6.0 s | 2.00 s | 2.03 s | 2.14 s | 3.15 s | +2.15 s |

Observations (from the `--trace-csv` output and frames of the annotated videos):

- Both persons were `UPRIGHT` from the first frame until the fall (during the labelled walking / standing phase:
  torso angle ≤ 7°, box w/h 0.26–0.34),
  and both alerts fired inside the labelled fall window. The alerts stayed on until the end of each clip, where the
  person is still on the floor (`end_reason: end_of_video`).
- `FALLING` was triggered by the torso angle crossing 45° in both clips. The hip velocity peaked at 0.70 H/s
  (fall-05) and 0.60 H/s (fall-11), below the 0.8 H/s default, so the velocity cue did not contribute.
- `FALLING` lasted only 1–4 frames before the lying rule was met, so the orange box is barely visible.
- fall-11: the torso angle first rose to ~43° at 1.57 s, then dropped to ~25° while the body fell backward toward
  the camera (foreshortened torso); it crossed 45° at 2.03 s. Most of the delay between the label start (1.0 s)
  and the alert comes from this phase plus the 1.0 s `fallen_seconds` confirmation.
- The label start marks the beginning of the motion, so a delay of at least `fallen_seconds` after the person
  reaches the floor is expected by design.
- No threshold was changed after looking at these clips; the values in `FallConfig` are the initial defaults.

An earlier demo GIF was made from a 12 s concatenation of these two clips; the cut between them produced a new
track id (#3) and a second alert (9.00 s, i.e. 3.13 s into the second clip). The current GIF comes from the
[fall mix](#fall-mix).

## Snow clips (outdoor, Pexels)

Same default settings, no threshold changed. Fall times are a visual estimate from frame mosaics (no dataset
label); state times come from `--trace-csv`.

| Clip | Fall (visual estimate) | `FALLING` | `FALLEN` | `ALERT` | Upright before? |
|---|---|---|---|---|---|
| snow-pexels-20595440 | upright until ~2.9 s, lying on the snow ~3.5–3.6 s | 3.35 s | 3.45 s | none (clip ends at 4.55 s) | yes (0–3.30 s) |

Observations (from the trace and frames of the annotated videos):

- **No false alerts** on the snow clip: the person was `UPRIGHT` before the fall, including the kicking motion (torso angle up to ~40° before the fall,
  below the 45° `falling_angle`).
- **20595440 — detected but not confirmed before the clip ends.** The torso angle crossed 45° at 3.35 s
  (`FALLING`) and 60° at 3.45 s (`FALLEN`), about when the person reaches the snow. The 1.0 s `fallen_seconds`
  confirmation would end at ~4.45 s, but track #1 was last seen at 4.08 s (body partly hidden in deep snow, likely cause) and a new track #2
  appeared in `FALLEN` at 4.12 s, restarting the timer; the clip ends at 4.55 s. In a single exploratory run with
  `--fallen-seconds 0.5` (not kept, not a default change) track #1 alerted at 3.92 s.
- Possible changes, **not applied**: a shorter `fallen_seconds` for short clips (more alerts on brief lying
  postures), or carrying the `FALLEN` timer over when a new track appears at the same place. Neither was tested on
  more data.

## Source footage

GMDCSA24 dataset (Ekram Alam et al.), MIT License — see [../samples/README.md](../samples/README.md) for links,
license notice and citation. The subjects (Subjects 1, 2 and 3 in the fall mix) are adults who, according to the
dataset authors, consented to the recording. The outputs are shown only to illustrate the method. The UniData clips
(CC BY-NC-ND) are not used in the fall mix or in any annotated output.

Snow clip: Pexels (Ovais Ibn farooq), [Pexels License](https://www.pexels.com/license/) — see
[../samples/README.md](../samples/README.md#snow-clips-pexels) for links.
