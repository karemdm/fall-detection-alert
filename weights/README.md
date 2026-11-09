# Weights

| File / model | Size | Where it comes from | Role |
|---|---|---|---|
| `weights/yolo11n-pose.pt` | ~6.0 MB | Official Ultralytics YOLO11n-pose (COCO keypoints), downloaded from the Ultralytics assets release (`v8.4.0`) and shipped in this folder. | **Default** pose model |
| `yolo11s-pose.pt`, `yolo11m-pose.pt`, ... | 20–120 MB | Official Ultralytics names; auto-downloaded to the working directory on first use. | Optional, more accurate keypoints, slower |

The default path is relative (`weights/yolo11n-pose.pt`), so run the command from the repository root or pass
`--model` explicitly. If the file is missing, `--model yolo11n-pose.pt` lets Ultralytics download it again.

## Usage

```bash
# default (shipped file)
fall-alert -i input.mp4 -o outputs/out.mp4

# larger official model, auto-downloaded
fall-alert -i input.mp4 -o outputs/out.mp4 --model yolo11s-pose.pt
```

## License

- YOLO11 pose weights: Ultralytics, AGPL-3.0 (an enterprise license is available from Ultralytics).
- Trained on COCO-Pose (17 keypoints, person class only). No fine-tuning was done for this project.
