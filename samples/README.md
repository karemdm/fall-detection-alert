# Sample clips

Two indoor test clips from the **GMDCSA24** human fall detection dataset (copied unmodified, original audio
included) and two outdoor snow clips from **Pexels** (see [Snow clips (Pexels)](#snow-clips-pexels)).

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

Both clips are under the [Pexels License](https://www.pexels.com/license/): free to use and modify, attribution not
required (given above as a courtesy). The license does not allow selling unaltered copies or implying endorsement by
the people shown.
