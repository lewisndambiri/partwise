"""Render README graphics and a short silent tour from measured local outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from partwise.decisions import Scenario, project_decisions
from partwise.visualization import heat_overlay


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data/mvtec_ad"
OUT = ROOT / "docs/assets"
W, H = 1600, 900
BG = "#101a28"
PANEL = "#182a3c"
BORDER = "#345064"
WHITE = "#eff6f4"
MUTED = "#acc3ce"
TEAL = "#61d0b5"
ORANGE = "#f4a35a"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def canvas(height: int = H) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, height), BG)
    return image, ImageDraw.Draw(image)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *,
          size: int = 28, color: str = WHITE, bold: bool = False) -> None:
    draw.text(xy, text, font=font(size, bold), fill=color)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=26, fill=PANEL, outline=BORDER, width=2)


def asset_image(path: str) -> Image.Image:
    with Image.open(DATA / path) as source:
        return source.convert("RGB")


def fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def source_footer(draw: ImageDraw.ImageDraw, y: int) -> None:
    label(draw, (72, y), "MVTec AD metal nut · CC BY-NC-SA 4.0 · Local benchmark, not a factory trial",
          size=20, color="#8ca7b5")


def load_inputs() -> tuple[dict, dict, dict, dict[str, np.ndarray], tuple[float, float]]:
    manifest = json.loads((ARTIFACTS / "metal_nut_manifest.json").read_text())
    baseline = json.loads((ARTIFACTS / "metal_nut_baseline.json").read_text())
    patch = json.loads((ARTIFACTS / "metal_nut_patchcore.json").read_text())
    with np.load(ARTIFACTS / "metal_nut_patchcore_maps.npz") as bundle:
        maps = {
            str(path): anomaly_map.astype(np.float32)
            for path, anomaly_map in zip(bundle["image_paths"], bundle["anomaly_maps"])
        }
    low, high = np.percentile(np.stack(list(maps.values())), [75, 99])
    return manifest, baseline, patch, maps, (float(low), float(high))


def draw_hero(patch: dict, maps: dict[str, np.ndarray], heat_range: tuple[float, float]) -> Image.Image:
    image, d = canvas(760)
    d.rounded_rectangle((0, 0, W, 760), radius=0, fill="#112131")
    d.ellipse((-200, -320, 830, 640), fill="#183f4c")
    label(d, (72, 55), "◉  PARTWISE  /  COMPUTER VISION QUALITY INSPECTION", size=25, color=TEAL, bold=True)
    label(d, (72, 154), "See the defect.", size=65, bold=True)
    label(d, (72, 236), "Understand the decision.", size=54, bold=True)
    label(d, (74, 350), "Anomaly detection for manufactured metal nuts, paired", size=26, color=MUTED)
    label(d, (74, 390), "with failure analysis and a production decision model.", size=26, color=MUTED)

    cards = [
        ("86 / 93", "defects found"),
        ("1 / 22", "good parts rejected"),
        ("0.988", "image AUROC"),
    ]
    for i, (value, caption) in enumerate(cards):
        x = 72 + i * 245
        panel(d, (x, 540, x + 222, 660))
        label(d, (x + 18, 557), value, size=34, bold=True)
        label(d, (x + 18, 611), caption, size=19, color=MUTED)

    path = "metal_nut/test/bent/000.png"
    original = asset_image(path)
    overlay = heat_overlay(original, maps[path], *heat_range)
    for x, title, content in (
        (868, "Original", original),
        (1202, "Anomaly overlay", overlay),
    ):
        panel(d, (x - 14, 100, x + 318, 588))
        image.paste(fit_tile(content, (304, 360)), (x, 156))
        label(d, (x + 9, 115), title, size=23, bold=True)
    label(d, (893, 614), "Detected bend · score 17.41 > threshold 14.85", size=23, color=ORANGE, bold=True)
    source_footer(d, 708)
    return image


def draw_portfolio_card(
    maps: dict[str, np.ndarray], heat_range: tuple[float, float]
) -> Image.Image:
    """Render paired original and overlay views for the 4:3 thumbnail."""
    path = "metal_nut/test/bent/000.png"
    original = asset_image(path)
    overlay = heat_overlay(original, maps[path], *heat_range)
    image = Image.new("RGB", (1200, 900), BG)
    d = ImageDraw.Draw(image)
    label(d, (56, 43), "PARTWISE  /  COMPUTER VISION", size=28, color=TEAL, bold=True)
    for x, title, content in (
        (56, "Original", original),
        (624, "Anomaly overlay", overlay),
    ):
        d.rounded_rectangle((x - 8, 118, x + 528, 700), radius=20, fill=PANEL, outline=BORDER, width=2)
        image.paste(fit_tile(content, (512, 512)), (x, 154))
        label(d, (x + 8, 124), title, size=22, bold=True)
    label(d, (56, 737), "BENT DEFECT  /  FLAGGED FOR REVIEW", size=30, color=ORANGE, bold=True)
    label(d, (56, 811), "MVTec AD metal nut · CC BY-NC-SA 4.0 · Adapted image and overlay", size=17, color="#8ca7b5")
    return image


def draw_inspection(
    manifest: dict, patch: dict, maps: dict[str, np.ndarray], heat_range: tuple[float, float]
) -> Image.Image:
    image, d = canvas()
    label(d, (72, 48), "01  /  INSPECT A PART", size=23, color=TEAL, bold=True)
    label(d, (72, 100), "The model can point. The misses still matter.", size=43, bold=True)
    label(d, (72, 165), "A detected bend beside a bend that fell just below the fixed image threshold.", size=24, color=MUTED)
    masks = {item["image"]: item["mask"] for item in manifest["test_defective"]}
    scores = {
        item["image"]: item["score"]
        for split in ("test_normal", "test_defective")
        for item in patch["scores"][split]
    }
    for row, (name, path, accent) in enumerate((
        ("Detected bend", "metal_nut/test/bent/000.png", TEAL),
        ("Missed bend", "metal_nut/test/bent/006.png", ORANGE),
    )):
        y = 230 + row * 308
        panel(d, (72, y, 1528, y + 292))
        original = asset_image(path)
        overlay = heat_overlay(original, maps[path], *heat_range)
        with Image.open(DATA / masks[path]) as source:
            mask = source.convert("RGB")
        label(d, (98, y + 13), name, size=23, color=accent, bold=True)
        label(d, (1250, y + 17), f"Score {scores[path]:.2f}", size=20, color=WHITE)
        for i, (title, content) in enumerate((
            ("Original", original), ("Anomaly overlay", overlay), ("Ground truth", mask)
        )):
            x = 98 + i * 478
            image.paste(fit_tile(content, (350, 218)), (x, y + 47))
            label(d, (x, y + 267), title, size=18, color=MUTED)
    label(d, (72, 842), "Both cases use the same validation-set threshold; visible color is not the decision rule.", size=19, color=MUTED)
    label(d, (72, 872), "MVTec AD metal nut · CC BY-NC-SA 4.0 · Images resized and overlaid", size=16, color="#8ca7b5")
    return image


def progress_bar(d: ImageDraw.ImageDraw, x: int, y: int, width: int, value: float, color: str) -> None:
    d.rounded_rectangle((x, y, x + width, y + 28), radius=14, fill="#294357")
    d.rounded_rectangle((x, y, x + max(8, int(width * value)), y + 28), radius=14, fill=color)


def draw_evaluation(baseline: dict, patch: dict) -> Image.Image:
    image, d = canvas()
    label(d, (72, 48), "02  /  LOCKED TEST COMPARISON", size=23, color=TEAL, bold=True)
    label(d, (72, 103), "Local patches catch what whole-image features miss.", size=43, bold=True)
    label(d, (72, 168), "Same 115 test images. Thresholds set on 44 normal validation images.", size=25, color=MUTED)
    specs = [
        (72, "Defects found", "higher is better", "true_positive", "n_defective"),
        (820, "Good parts rejected", "lower is better", "false_positive", "n_good"),
    ]
    for x, title, direction, numerator, denominator in specs:
        panel(d, (x, 255, x + 708, 690))
        label(d, (x + 35, 295), title, size=33, bold=True)
        label(d, (x + 35, 344), direction.upper(), size=18, color=TEAL)
        for i, (name, result, color) in enumerate((
            ("Global baseline", baseline, "#93a9b8"),
            ("PatchCore", patch, TEAL),
        )):
            y = 410 + i * 125
            m = result["metrics"]
            ratio = m[numerator] / m[denominator]
            label(d, (x + 35, y), name, size=22, color=MUTED)
            label(d, (x + 520, y), f"{m[numerator]} / {m[denominator]}", size=25, color=WHITE, bold=True)
            progress_bar(d, x + 35, y + 44, 622, ratio, color)
    label(d, (75, 740), "PatchCore image AUROC 0.988  ·  Pixel AUROC 0.981  ·  Pixel AP 0.848", size=27, color=WHITE, bold=True)
    label(d, (75, 795), "False-reject uncertainty remains wide: only 22 good test images.", size=24, color=ORANGE)
    source_footer(d, 858)
    return image


def draw_decisions(baseline: dict, patch: dict) -> Image.Image:
    scenario = Scenario.from_dict(json.loads((ROOT / "scenarios/illustrative.json").read_text()))
    base = project_decisions(
        baseline["metrics"]["defect_recall"], baseline["metrics"]["false_reject_rate"], scenario
    )
    local = project_decisions(
        patch["metrics"]["defect_recall"], patch["metrics"]["false_reject_rate"], scenario
    )
    image, d = canvas()
    label(d, (72, 48), "03  /  DECISION STUDIO", size=23, color=TEAL, bold=True)
    label(d, (72, 103), "A model score is only part of the operating choice.", size=43, bold=True)
    label(d, (72, 169), "Illustrative assumptions: 10,000 parts · 1% defects · 150 reviews available.", size=24, color=MUTED)
    for i, (name, result, color) in enumerate((
        ("Global baseline", base, "#93a9b8"),
        ("PatchCore", local, TEAL),
    )):
        x = 72 + i * 748
        panel(d, (x, 260, x + 708, 700))
        label(d, (x + 35, 297), name, size=32, color=color, bold=True)
        label(d, (x + 35, 390), f"{result['missed_defective']:.1f}", size=53, bold=True)
        label(d, (x + 35, 456), "expected missed defects", size=23, color=MUTED)
        label(d, (x + 35, 540), f"{result['good_rejected']:.1f}", size=53, bold=True)
        label(d, (x + 35, 606), "expected good parts rejected", size=23, color=MUTED)
    label(d, (72, 752), "Scenario outputs, not measured factory savings.", size=29, color=ORANGE, bold=True)
    label(d, (72, 807), "Prevalence, review capacity, reviewer accuracy, and costs are editable in the local app.", size=22, color=MUTED)
    source_footer(d, 858)
    return image


def render_video(slides: list[Image.Image]) -> None:
    output = OUT / "walkthrough.mp4"
    fps = 12
    frames_per_slide = 60
    process = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for index, slide in enumerate(slides):
            for frame in range(frames_per_slide):
                if frame >= frames_per_slide - 8 and index + 1 < len(slides):
                    blend = (frame - (frames_per_slide - 8) + 1) / 9
                    image = Image.blend(slide, slides[index + 1], blend)
                else:
                    image = slide
                process.stdin.write(image.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg could not encode the walkthrough")
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(output),
            "-vf", "fps=4,scale=800:-1:flags=lanczos",
            "-t", "20", str(OUT / "walkthrough.gif"),
        ],
        check=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, baseline, patch, maps, heat_range = load_inputs()
    hero = draw_hero(patch, maps, heat_range)
    inspection = draw_inspection(manifest, patch, maps, heat_range)
    evaluation = draw_evaluation(baseline, patch)
    decisions = draw_decisions(baseline, patch)
    hero.save(OUT / "hero.png", optimize=True)
    draw_portfolio_card(maps, heat_range).save(OUT / "portfolio-card.png", optimize=True)
    inspection.save(OUT / "inspection.png", optimize=True)
    evaluation.save(OUT / "evaluation.png", optimize=True)
    decisions.save(OUT / "decision.png", optimize=True)
    render_video([
        ImageOps.pad(hero, (W, H), method=Image.Resampling.LANCZOS, color=BG),
        inspection, evaluation, decisions,
    ])
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
