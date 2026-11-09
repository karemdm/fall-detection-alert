# Sample clips

Two indoor test clips from the **GMDCSA24** human fall detection dataset (copied unmodified, original audio
included), one outdoor snow clip from **Pexels** (see [Snow clips (Pexels)](#snow-clips-pexels)) and a
**fall mix** compiled from GMDCSA24 and Pexels footage (see [Fall mix](#fall-mix-fall_mix_inputmp4)).

## Fall mix (`fall_mix_input.mp4`)

A 55.2 s compilation of nine falls: eight GMDCSA24 clips (three subjects, day and night recordings, side,
forward and backward falls, different rooms and camera positions) followed by the Pexels snow fall. Built with
ffmpeg:

- each segment runs from ~2 s before the labelled fall start (or the clip start) to ~2.5 s after the labelled fall
  end (or the clip end); in practice the labelled fall lasts until the end of almost every GMDCSA24 clip, so the
  segments end at the clip end;
- each segment is converted to 30 fps and fitted into 1280x720 with black padding (the 640x480 night clips get
  black side bars, the portrait snow clip gets wide black side bars), audio removed;
- the segments are concatenated without transitions and encoded with H.264 (`-crf 23`), 1657 frames.

The six Subject 2 / Subject 3 clips were downloaded from the GMDCSA24 repository only to build this mix and are
not stored separately in this repository. The UniData clips are **not** used in the mix.

| # | Mix time (s) | Source file | Source in–out (s) | Label (`Subject N/Fall.csv`) | Labelled fall (source s) | Labelled fall (mix s) |
|---|---|---|---|---|---|---|
| 1 | 0.00–5.87 | GMDCSA24 `Subject 1/Fall/05.mp4` | 0.00–5.86 (whole clip) | Walking to falling, right side fall | 1.8–5.0 | 1.80–5.00 |
| 2 | 5.87–13.60 | GMDCSA24 `Subject 2/Fall/02.mp4` | 0.70–8.43 | Right side fall on the ground after walking | 2.7–8.0 | 7.87–13.17 |
| 3 | 13.60–19.07 | GMDCSA24 `Subject 3/Fall/04.mp4` | 0.00–5.47 | Walking and then falling backward on the floor | 1.2–5.0 | 14.80–18.60 |
| 4 | 19.07–25.07 | GMDCSA24 `Subject 2/Fall/17.mp4` | 0.50–6.50 | Walking then falling forward | 2.5–6.0 | 21.07–24.57 |
| 5 | 25.07–31.17 | GMDCSA24 `Subject 1/Fall/11.mp4` | 0.00–6.10 (whole clip) | Standing position to falling (backward) | 1.0–6.0 | 26.07–31.07 |
| 6 | 31.17–39.60 | GMDCSA24 `Subject 2/Fall/13.mp4` (night, 640x480) | 0.50–8.92 | Falling backward (side view) | 2.5–8.0 | 33.17–38.67 |
| 7 | 39.60–45.67 | GMDCSA24 `Subject 3/Fall/16.mp4` (night) | 0.00–6.08 (whole clip) | Walking and then falling left side on the floor | 1.8–7.0 (clip ends at 6.08) | 41.40–45.67 |
| 8 | 45.67–51.60 | GMDCSA24 `Subject 2/Fall/23.mp4` | 1.60–7.53 | Standing then falling sideways | 3.6–7.0 | 47.67–51.07 |
| 9 | 51.60–55.23 | Pexels `snow-pexels-20595440.mp4` (portrait) | 0.90–4.55 | Kicks a tree, falls backward into deep snow (no label) | ~2.9–3.6 (visual estimate) | ~53.60–54.30 |

Labels and times for segments 1–8 come from the dataset CSV files (`Subject 1/Fall.csv`, `Subject 2/Fall.csv`,
`Subject 3/Fall.csv`); segment 9 is a visual estimate (see below). Mix times are rounded to 0.01 s.

Licenses of the mix: the GMDCSA24 footage is under the MIT License (notice and citation below; the license allows
modified copies as long as the notice is kept), the snow footage is under the Pexels License (modification
allowed). Source repository for segments 1–8:
https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos.

## GMDCSA24 clips

| File | Original file | Duration | Ground truth (`Subject 1/Fall.csv`) |
|---|---|---|---|
| `gmdcsa24-s1-fall-05.mp4` | `Subject 1/Fall/05.mp4` | 5.86 s, 1280x720 | "Walking to falling. Right side fall": walking 0–1.8 s, falling 1.8–5 s |
| `gmdcsa24-s1-fall-11.mp4` | `Subject 1/Fall/11.mp4` | 6.10 s, 1280x720 | "Standing position to falling": standing 0–1 s, falling (backward) 1–6 s |

- Dataset: GMDCSA24 — A Dataset for Human Fall Detection in Videos.
- Repository: https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
- Zenodo: https://doi.org/10.5281/zenodo.12921216
- Article (Data in Brief, CC BY 4.0): https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611/
- Recording: fixed 720p, ~30 fps webcam, indoor room. According to the authors, the subjects gave consent.

### License

The GMDCSA24 clips are distributed under the MIT License of the dataset repository:

```
MIT License

Copyright (c) 2024 Ekram Alam
```

Full text: https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/blob/master/LICENSE
(the permission notice must be kept with copies of the files).

### Citation

If you use these clips, cite the dataset:

> Ekram Alam et al., "GMDCSA24: A dataset for human fall detection in videos", *Data in Brief*, 2024.
> https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611/ — dataset DOI: https://doi.org/10.5281/zenodo.12921216

## Snow clips (Pexels)

One outdoor clip of an adult falling in snow, downloaded from Pexels (1080p file) and copied with the audio track
removed (`ffmpeg -c copy -an`, no re-encode, no trim). It is not a slip on ice: no suitable Pexels clip of an
everyday slip on an icy walkway was found, so this is the closest real-time, full-body falls in snow.

| File | Source | Author | Duration, format | Content | Approx. fall (visual estimate) |
|---|---|---|---|---|---|
| `snow-pexels-20595440.mp4` | [fun in snow, snow fight](https://www.pexels.com/video/fun-in-snow-snow-fight-20595440/) | [Ovais Ibn farooq](https://www.pexels.com/@ovais-ibn-farooq-454781010/) | 4.55 s, 1080x1920 (portrait), 60 fps | Man standing in deep snow kicks a snow-covered tree, loses balance and falls backward, ending lying in the snow. | upright (kicking) until ~2.9 s, on the snow ~3.5–3.6 s, lying until the end (~1 s) |

Fall times are a visual estimate from 2–4 fps frame mosaics, not a ground-truth label. The clip looks real-time
(no slow motion or time-lapse visible).

### License

The clip is under the [Pexels License](https://www.pexels.com/license/): free to use and modify, attribution not
required (given above as a courtesy). The license does not allow selling unaltered copies or implying endorsement by
the people shown.

## UniData "Fall Detection" samples (CC BY-NC-ND 4.0)

| File | Source | License | Content (visual check) |
|---|---|---|---|
| `unidata-sample_1.mp4` | [ud-smart-city/fall-detection](https://huggingface.co/datasets/ud-smart-city/fall-detection), `sample_1.mp4` | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | Portrait 1080x1920, 13.3 s. A man walks across a snowy parking lot, slips, lies on the snow and gets up. |
| `unidata-sample_2.mp4` | [ud-smart-city/fall-detection](https://huggingface.co/datasets/ud-smart-city/fall-detection), `sample_2.mp4` | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | 1920x1080, 13.8 s. A person walks along an icy sidewalk next to a fence, slips, falls and gets up. |

**These two files are redistributed unmodified under their own license (CC BY-NC-ND 4.0, © UniData),
not under the repository's AGPL-3.0, for non-commercial study only.** Because the license does not allow
derivatives, no annotated video, crop or GIF of them is included in this repository, and they are not part of
the fall mix or of any other derivative; only the measured alert times are reported in the README.
