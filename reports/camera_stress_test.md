# Camera-condition stress test

This is a descriptive sensitivity check on the **same 115 held-out MVTec AD metal nut test images** used in the main evaluation. The saved PatchCore model and its threshold of **14.85**, calibrated on 44 unchanged normal validation images, remained fixed. Each test image was scored unchanged, then with three prespecified synthetic changes applied at inference: brightness ×0.8, brightness ×1.2, and Gaussian blur with radius 1.5 pixels on the source image. These transformations approximate limited exposure and focus changes; they are not samples from another production camera.

| Test condition | Defects found | Good parts rejected | Image AUROC | Paired decision change from original |
|---|---:|---:|---:|---|
| Original | 86 / 93 | 1 / 22 | 0.988 | Reference |
| 20% dimmer | 85 / 93 | 1 / 22 | 0.990 | Scratch 022 became a miss |
| 20% brighter | 87 / 93 | 2 / 22 | 0.978 | Color 018 became detected; good 007 became a false reject |
| Gaussian blur, radius 1.5 | 86 / 93 | 1 / 22 | 0.984 | No accept/reject changes |

The brightened condition illustrates why an improved defect count alone does not settle the operating choice: it also rejected another good part. The unchanged blur counts show only that this **particular mild transform** left decisions unchanged. The small set of 22 good images still makes the estimated false-reject rate uncertain.

The unchanged images were rescored in this script as a consistency check; their scores match the saved evaluation to within **0.000009**. No transformed score was used to fit the model or change its threshold. The full paired scores and condition definitions are in ignored local artifact `artifacts/camera_stress_test.json`.

Reproduce locally after the saved model exists:

```bash
.venv/bin/python scripts/stress_test_camera.py
```

A real camera qualification would need newly collected images across shifts, lighting, focus, surface finish, and part batches, with a separately set acceptance criterion. Synthetic changes on one benchmark camera cannot establish that performance.
