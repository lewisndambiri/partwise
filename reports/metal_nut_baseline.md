# Metal nut baseline: first measured result

## Experiment contract

- Dataset: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), `metal_nut` category, local copy excluded from Git.
- Seed: 42. Of 220 defect-free training images, 176 fit the reference bank and 44 calibrate the threshold. The official test set remains separate: 22 good and 93 defective images.
- Method: fixed ImageNet ResNet18 global embeddings; anomaly score is `1 − cosine similarity` to the nearest normal training image.
- Decision rule: flag a part if its score is **greater than 0.028159**. This threshold is the 95th percentile of normal validation scores, selected before test evaluation.
- Tested environment: Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.5.3.

## Test result

| Measure | Result |
|---|---:|
| Defects found | 65 / 93 (69.9%) |
| Defects missed | 28 / 93 |
| Good parts rejected | 4 / 22 (18.2%) |
| Good parts accepted | 18 / 22 |
| Image AUROC | 0.793 |
| Approximate 95% Wilson interval, defect recall | 59.9%–78.3% |
| Approximate 95% Wilson interval, false reject rate | 7.3%–38.5% |

Two of the 44 normal validation images (4.5%) were flagged. The higher test false-reject rate may reflect sampling variation, a shift between images, or both. The 22 good test images do not support a precise factory false-reject estimate.

### Defects found by supplied type

| Type | Found / total |
|---|---:|
| Bent | 8 / 25 |
| Color | 13 / 22 |
| Flip | 23 / 23 |
| Scratch | 21 / 23 |

The baseline handles flipped nuts well and misses many bent and color defects. This pattern motivated the [PatchCore comparison](model_comparison.md), which tests local patch features on the same split.

The benchmark contains 93 defective images among 115 test images. Its measured precision (94.2%) therefore must **not** be presented as expected precision in production, where defects may be much rarer.

## Illustrative production scenario

The local [scenario](../scenarios/illustrative.json) assumes 10,000 parts, 1% defects, a 150-part human review capacity, 95% reviewer sensitivity, and 99% reviewer specificity. Applying the **measured point estimates** gives about 30 missed defects, 1,657 good parts rejected, and 150 reviewed. These are scenario projections, not measured factory outcomes. The large rejection count is driven by the baseline's false-reject rate and the limited review capacity.

## Follow-up

The [PatchCore comparison](model_comparison.md) measures bent and color recall, false rejects, defect localization, and CPU inference time under the same split and threshold protocol.
