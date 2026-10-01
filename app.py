"""Partwise local inspection workbench. Run with `streamlit run app.py`."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from partwise.decisions import Scenario, project_decisions
from partwise.inference import load_patchcore_model, predict_image
from partwise.visualization import heat_overlay


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data/mvtec_ad"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="Partwise · Inspection Workbench",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      .stApp {background: radial-gradient(circle at 85% 0%, #203b4e 0, #0d1826 42%, #0b1420 100%);}
      .block-container {max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem;}
      h1, h2, h3 {letter-spacing: -0.035em;}
      .eyebrow {color:#78d6c3; font-size:.76rem; letter-spacing:.18em; font-weight:700;}
      .hero {padding:1.7rem 2rem; border:1px solid #355265; border-radius:20px;
             background:linear-gradient(115deg, #183c4b, #132435 58%, #3b2a2c); margin-bottom:1.3rem;}
      .hero h1 {font-size:3rem; margin:.25rem 0 .4rem; color:#f2f7f7;}
      .hero p {color:#bacdd3; font-size:1.08rem; max-width:760px; margin:0;}
      .status-pill {display:inline-block; padding:.35rem .7rem; border-radius:999px;
                    background:#1b473c; color:#a2ead1; font-size:.83rem; font-weight:700;}
      .status-pill.alert {background:#5b3828; color:#ffd09a;}
      div[data-testid="stMetric"] {border:1px solid #30475b; border-radius:14px;
          padding:.7rem 1rem; background:#122437;}
      div[data-testid="stTabs"] button {font-weight:650;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_runs() -> tuple[dict, dict, dict, dict[str, np.ndarray], tuple[float, float]]:
    manifest = json.loads((ARTIFACTS / "metal_nut_manifest.json").read_text())
    baseline = json.loads((ARTIFACTS / "metal_nut_baseline.json").read_text())
    patch = json.loads((ARTIFACTS / "metal_nut_patchcore.json").read_text())
    with np.load(ARTIFACTS / "metal_nut_patchcore_maps.npz") as bundle:
        maps = {
            str(path): anomaly_map.astype(np.float32)
            for path, anomaly_map in zip(bundle["image_paths"], bundle["anomaly_maps"])
        }
    heat_low, heat_high = np.percentile(np.stack(list(maps.values())), [75, 99])
    return manifest, baseline, patch, maps, (float(heat_low), float(heat_high))


@st.cache_resource(show_spinner="Loading inspection model…")
def load_model():
    return load_patchcore_model(ARTIFACTS / "metal_nut_patchcore_model.pt")


def inspect_upload(image: Image.Image, image_size: int) -> tuple[float, np.ndarray]:
    return predict_image(load_model(), image, image_size)


def case_group(item: dict, threshold: float) -> str:
    flagged = item["score"] > threshold
    if item["label"]:
        return "Found defects" if flagged else "Missed defects"
    return "False rejects" if flagged else "Accepted good parts"


try:
    manifest, baseline, patch, maps, heat_range = load_runs()
except (OSError, KeyError, ValueError) as exc:
    st.error("Run the data audit, baseline, and PatchCore commands before opening the workbench.")
    st.code(".venv/bin/python -m partwise audit --data-root data/mvtec_ad --output artifacts/metal_nut_manifest.json\n"
            ".venv/bin/python -m partwise baseline --data-root data/mvtec_ad --manifest artifacts/metal_nut_manifest.json --output artifacts/metal_nut_baseline.json\n"
            ".venv/bin/python -m partwise patchcore --data-root data/mvtec_ad --manifest artifacts/metal_nut_manifest.json --output artifacts/metal_nut_patchcore.json --model-output artifacts/metal_nut_patchcore_model.pt --maps-output artifacts/metal_nut_patchcore_maps.npz")
    st.caption(str(exc))
    st.stop()

st.markdown(
    """<div class="hero"><div class="eyebrow">PARTWISE / COMPUTER VISION QUALITY INSPECTION</div>
    <h1>Find defects. Understand decisions.</h1>
    <p>A local workbench for metal nut inspection: see what the model flagged,
    where it looked, and what its errors could mean for a production line.</p></div>""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Defects found", f"{patch['metrics']['true_positive']} / {patch['metrics']['n_defective']}")
m2.metric("Good parts rejected", f"{patch['metrics']['false_positive']} / {patch['metrics']['n_good']}")
m3.metric("Image AUROC", f"{patch['metrics']['image_auroc']:.3f}")
m4.metric("Test cases", str(patch["metrics"]["n_good"] + patch["metrics"]["n_defective"]))

inspect_tab, compare_tab, decisions_tab = st.tabs([
    "Inspect a part", "Compare the models", "Decision studio"
])

with inspect_tab:
    st.subheader("Inspect a part")
    st.caption("Browse official test cases or upload a metal nut photo for exploratory inference.")
    upload = st.file_uploader("Upload a metal nut image", type=["png", "jpg", "jpeg"])
    score_lookup = {
        item["image"]: item
        for split in ("test_normal", "test_defective")
        for item in patch["scores"][split]
    }
    baseline_lookup = {
        item["image"]: item
        for split in ("test_normal", "test_defective")
        for item in baseline["scores"][split]
    }
    mask_lookup = {item["image"]: item["mask"] for item in manifest["test_defective"]}

    if upload is not None:
        with Image.open(upload) as source:
            image = source.convert("RGB")
        with st.spinner("Inspecting uploaded image…"):
            score, anomaly_map = inspect_upload(image, patch["configuration"]["image_size"])
        selected_path = None
        label_text = "Uploaded image · ground truth unknown"
        baseline_text = "Baseline comparison is available for benchmark images only."
    else:
        all_cases = list(score_lookup.values())
        filter_choice = st.selectbox(
            "Case filter",
            ["All cases", "Missed defects", "False rejects", "Found defects", "Accepted good parts"],
        )
        filtered = [
            item for item in all_cases
            if filter_choice == "All cases" or case_group(item, patch["threshold"]) == filter_choice
        ]
        selected_path = st.selectbox(
            "Benchmark image",
            [item["image"] for item in filtered],
            format_func=lambda path: f"{path.split('/')[-2].replace('_', ' ').title()} · {path.split('/')[-1]}",
        )
        selected = score_lookup[selected_path]
        with Image.open(DATA_ROOT / selected_path) as source:
            image = source.convert("RGB")
        anomaly_map = maps[selected_path]
        score = selected["score"]
        label_text = case_group(selected, patch["threshold"])
        old = baseline_lookup[selected_path]
        baseline_text = (
            f"Global baseline: {'flagged' if old['score'] > baseline['threshold'] else 'accepted'}"
            f" · score {old['score']:.4f} (threshold {baseline['threshold']:.4f})"
        )

    status = "FLAGGED FOR REVIEW" if score > patch["threshold"] else "ACCEPTED"
    status_class = " alert" if status == "FLAGGED FOR REVIEW" else ""
    st.markdown(f"<span class='status-pill{status_class}'>{status}</span>", unsafe_allow_html=True)
    st.caption(f"{label_text} · PatchCore score {score:.2f} · threshold {patch['threshold']:.2f}")
    st.caption(baseline_text)
    has_mask = selected_path in mask_lookup
    columns = st.columns(3 if has_mask else 2)
    columns[0].image(image, caption="Original", width="stretch")
    columns[1].image(
        heat_overlay(image, anomaly_map, *heat_range),
        caption="Anomaly heatmap overlay",
        width="stretch",
    )
    if has_mask:
        with Image.open(DATA_ROOT / mask_lookup[selected_path]) as source:
            mask = source.convert("L")
        columns[2].image(
            mask,
            caption="Dataset defect mask",
            width="stretch",
        )
    st.caption("Overlay intensity uses the 75th–99th percentile range of benchmark patch scores. The image decision uses the score above, not an overlay cutoff.")
    if upload is not None:
        st.warning("Uploaded photos may differ from the benchmark camera setup. Treat this result as exploratory.")

with compare_tab:
    st.subheader("Two models, one test set")
    st.caption("Both thresholds came from the same 44 normal validation images. Defective test images were held out until evaluation.")
    rows = []
    for name, run in (("Global baseline", baseline), ("PatchCore", patch)):
        m = run["metrics"]
        rows.append({
            "Model": name,
            "Defects found": f"{m['true_positive']} / {m['n_defective']}",
            "Good parts rejected": f"{m['false_positive']} / {m['n_good']}",
            "Image AUROC": round(m["image_auroc"], 3),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("#### Defects found by type")
    defect_types = {item["image"]: item["defect_type"] for item in manifest["test_defective"]}
    chart_rows = []
    for name, run in (("Global baseline", baseline), ("PatchCore", patch)):
        counts = {}
        for item in run["scores"]["test_defective"]:
            defect_type = defect_types[item["image"]]
            found, total = counts.get(defect_type, (0, 0))
            counts[defect_type] = (found + int(item["score"] > run["threshold"]), total + 1)
        for defect_type, (found, total) in sorted(counts.items()):
            chart_rows.append({"Type": defect_type.title(), "Model": name, "Recall": found / total, "Found": found, "Total": total})
    chart = alt.Chart(pd.DataFrame(chart_rows)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Type:N", title=None),
        xOffset="Model:N",
        y=alt.Y("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["Global baseline", "PatchCore"], range=["#889eaf", "#5ad0b3"])),
        tooltip=["Type", "Model", "Found", "Total", alt.Tooltip("Recall:Q", format=".1%")],
    ).properties(height=310)
    st.altair_chart(chart, width="stretch")
    st.info("PatchCore missed 7 defects and rejected 1 good image. Browse those cases in the Inspect tab.")
    st.caption(
        f"PatchCore pixel AUROC {patch['metrics']['pixel_auroc']:.3f}; pixel average precision "
        f"{patch['metrics']['pixel_average_precision']:.3f}. Model forward pass averaged "
        f"{patch['runtime']['inference_ms_per_image']:.1f} ms/image at batch size 8 on this CPU."
    )
    st.caption("False-reject uncertainty is substantial: PatchCore's approximate 95% interval is 0.8%–21.8% from only 22 good test images.")

with decisions_tab:
    st.subheader("Decision studio")
    st.caption("Change the factory assumptions. Both models use their measured test recall and false-reject rate; costs are illustrative units per part.")
    left, middle, right = st.columns(3)
    with left:
        volume = st.number_input("Parts in the scenario", min_value=100, max_value=1_000_000, value=10_000, step=100)
        prevalence_pct = st.slider("Defect prevalence (%)", 0.1, 10.0, 1.0, 0.1)
        review_capacity = st.number_input("Human review capacity", min_value=0, max_value=int(volume), value=min(150, int(volume)), step=10)
    with middle:
        missed_cost = st.number_input("Cost of a missed defect", min_value=0.0, value=100.0, step=10.0)
        false_reject_cost = st.number_input("Cost of rejecting a good part", min_value=0.0, value=5.0, step=1.0)
        review_cost = st.number_input("Cost of one review", min_value=0.0, value=1.0, step=0.5)
    with right:
        reviewer_sensitivity_pct = st.slider("Reviewer defect catch rate (%)", 50, 100, 95)
        reviewer_specificity_pct = st.slider("Reviewer good-part acceptance (%)", 50, 100, 99)
        policy = st.selectbox("Flagged-part policy", ["review_first", "auto_reject"], format_func=lambda p: "Review then reject overflow" if p == "review_first" else "Reject all flagged parts")
    scenario = Scenario.from_dict({
        "volume": int(volume),
        "prevalence": prevalence_pct / 100,
        "missed_defect_cost": float(missed_cost),
        "false_reject_cost": float(false_reject_cost),
        "review_cost": float(review_cost),
        "review_capacity": int(review_capacity),
        "reviewer_sensitivity": reviewer_sensitivity_pct / 100,
        "reviewer_specificity": reviewer_specificity_pct / 100,
        "policy": policy,
    })
    base_projection = project_decisions(baseline["metrics"]["defect_recall"], baseline["metrics"]["false_reject_rate"], scenario)
    patch_projection = project_decisions(patch["metrics"]["defect_recall"], patch["metrics"]["false_reject_rate"], scenario)
    st.markdown("#### Expected outcomes")
    projection_rows = []
    for name, value in (("Global baseline", base_projection), ("PatchCore", patch_projection)):
        projection_rows.append({
            "Model": name,
            "Missed defects": round(value["missed_defective"], 1),
            "Good parts rejected": round(value["good_rejected"], 1),
            "Parts reviewed": round(value["reviewed"], 1),
            "Total cost units": round(value["total_cost"], 1),
        })
    st.dataframe(pd.DataFrame(projection_rows), hide_index=True, width="stretch")
    cost_rows = []
    for name, value in (("Global baseline", base_projection), ("PatchCore", patch_projection)):
        for label, key in (("Missed defects", "cost_misses"), ("False rejects", "cost_false_rejects"), ("Human review", "cost_review")):
            cost_rows.append({"Model": name, "Source": label, "Cost units": value[key]})
    cost_chart = alt.Chart(pd.DataFrame(cost_rows)).mark_bar().encode(
        x=alt.X("Model:N", title=None),
        y=alt.Y("Cost units:Q", title="Expected cost units"),
        color=alt.Color("Source:N", scale=alt.Scale(range=["#f5a75a", "#ee6d70", "#5ad0b3"])),
        tooltip=["Model", "Source", alt.Tooltip("Cost units:Q", format=",.1f")],
    ).properties(height=290)
    st.altair_chart(cost_chart, width="stretch")
    st.info("This is a scenario calculation, not measured factory savings. The false-reject estimate is based on 22 good test images; real prevalence and costs must be measured locally.")

st.divider()
st.caption("Dataset: MVTec AD, CC BY-NC-SA 4.0 · Local benchmark images are not bundled with source code · Results are from a public benchmark, not a production camera.")
