# Metal nut inspection: locked test comparison

Both methods use the same [MVTec AD metal nut](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) manifest: 176 normal images for fitting, 44 normal images for threshold calibration, and the official test set of 22 good plus 93 defective images. Each threshold targets a 5% validation false-reject rate and was set without test labels. The model configurations were chosen for CPU feasibility before the test comparison.

| Test measure | Global ResNet18 baseline | PatchCore (0.5% coreset) |
|---|---:|---:|
| Defects found | 65 / 93 (69.9%) | **86 / 93 (92.5%)** |
| Good parts rejected | 4 / 22 (18.2%) | **1 / 22 (4.5%)** |
| Image AUROC | 0.793 | **0.988** |
| Pixel AUROC | — | 0.981 |
| Pixel average precision | — | 0.848 |

PatchCore's approximate 95% Wilson intervals are **85.3%–96.3%** for defect recall and **0.8%–21.8%** for false-reject rate. The second interval is wide because there are only 22 good test images. These are benchmark estimates, not production guarantees.

## Where the difference occurred

| Supplied defect type | Baseline found | PatchCore found |
|---|---:|---:|
| Bent | 8 / 25 | **22 / 25** |
| Color | 13 / 22 | **19 / 22** |
| Flip | 23 / 23 | 23 / 23 |
| Scratch | 21 / 23 | 22 / 23 |

PatchCore still missed seven defects: three bent, three color, and one scratch. The good image `test/good/016.png` was the single false reject. The workbench exposes these cases for inspection.

## CPU and artifact details

PatchCore uses [Anomalib 2.6.2](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html), a pretrained ResNet18 backbone, 224 × 224 inputs, layers 2 and 3, nine neighbors, and a seeded 689-patch coreset. Fitting the memory bank took **47.3 seconds** on this machine. Its measured forward-pass time across validation and test images was **48.8 ms per image** at batch size 8; that excludes image loading, preprocessing, and UI work. The saved model is 12 MB and the compressed test heatmaps are 6.8 MB. A separate, identical batch-1 timing check on 20 preprocessed test images (three repetitions each, 60 measurements per model) found median score times of **35.3 ms** for the baseline and **35.7 ms** for PatchCore on this CPU with four PyTorch threads. Their respective 95th-percentile times were 46.7 and 48.2 ms. These timings exclude image I/O, preprocessing, and UI rendering; the [benchmark script](../scripts/benchmark_latency.py) records the exact boundary.

## Decision scenario

The [illustrative scenario](../scenarios/illustrative.json) assumes 10,000 parts with 1% defects and capacity to review 150 flagged parts. The baseline projects 30.4 missed defects and 1,657.1 good parts rejected; PatchCore projects 8.8 and 326.8 respectively. These numbers depend strongly on assumed prevalence, reviewer accuracy, capacity, and the uncertain false-reject estimates. They are for exploring decisions, not a measured factory return on investment.

## Conclusion

PatchCore found more defects and rejected fewer good parts than the global baseline on this test set. Its anomaly maps also make individual results inspectable. The workbench includes both methods, the seven PatchCore misses, the false reject, and an illustrative review scenario. The [camera-condition stress test](camera_stress_test.md) probes small synthetic exposure and blur changes. Validation on images from a production camera is still needed.
