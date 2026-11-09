# Results

Annotated videos produced by this project with the default settings
(`weights/yolo11n-pose.pt`, `conf=0.35`, `imgsz=640`, ByteTrack, default `FallConfig`), re-encoded to
H.264 720p without audio.

- `gmdcsa24-s1-fall-05_alert.mp4` + `gmdcsa24-s1-fall-05_events.json`: walking, then a forward fall to hands and
  knees and lying on the floor.
- `gmdcsa24-s1-fall-11_alert.mp4` + `gmdcsa24-s1-fall-11_events.json`: standing facing the camera, then a backward
  fall ending on the side.
- `snow-pexels-20595440_alert.mp4` + `snow-pexels-20595440_events.json`: outdoor, man kicks a tree, falls backward
  and lies in deep snow; the clip ends ~1 s after he lands (**no alert**, see below).

Reproduce (from the repo root):

```bash
fall-alert -i samples/gmdcsa24-s1-fall-05.mp4 -o outputs/fall-05.mp4 --events-json results/gmdcsa24-s1-fall-05_events.json --no-display
fall-alert -i samples/gmdcsa24-s1-fall-11.mp4 -o outputs/fall-11.mp4 --events-json results/gmdcsa24-s1-fall-11_events.json --no-display
fall-alert -i samples/snow-pexels-20595440.mp4 -o outputs/snow-20595440.mp4 --events-json results/snow-pexels-20595440_events.json --no-display
```

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

The demo GIF (`docs/assets/demo.gif`) is the output on a 12 s concatenation of the same two clips; the cut between
them produced a new track id (#3) and a second alert (9.00 s, i.e. 3.13 s into the second clip).

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
license notice and citation. The subject is an adult who, according to the dataset authors, consented to the
recording. The outputs are shown only to illustrate the method.

Snow clips: Pexels (Gustavo Fring; Ovais Ibn farooq), [Pexels License](https://www.pexels.com/license/) — see
[../samples/README.md](../samples/README.md#snow-clips-pexels) for links.
