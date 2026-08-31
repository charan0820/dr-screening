"""
Owner: P6
Streamlit GUI. This is the only file P6 owns for the interface — imports
run_pipeline() from main.py rather than calling individual src/ modules
directly, so integration stays in one place.

Color scheme:
  Dominant  #f3d9e5 (60%)
  Contrast  #4a3267 (30%)
  Highlight #de638a (10%)
Fonts: Orbitron (headings), Abel (subheadings), Cardo (body)
"""
import sys
import os
import time
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_checkpoint.keras")
MODEL_INPUT_SIZE = (224, 224)

sys.path.append(PROJECT_ROOT)
from main import run_pipeline  # noqa: E402
from src.model import load_trained_model  # noqa: E402
from src.quality import get_recapture_message  # noqa: E402

st.set_page_config(page_title="DR Screening Prototype", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Abel&family=Cardo&display=swap');

html, body, [class*="css"] {
    background-color: #f3d9e5;
    font-family: 'Cardo', serif;
}
h1 { font-family: 'Orbitron', sans-serif; color: #4a3267; }
h2, h3 { font-family: 'Abel', sans-serif; color: #4a3267; }
.stButton>button {
    background-color: #de638a;
    color: white;
    border: none;
}
.recommendation-banner {
    padding: 12px;
    border-radius: 8px;
    font-family: 'Abel', sans-serif;
    font-size: 1.1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("DR Screening Prototype")
st.caption("AI-assisted screening/referral support — not an autonomous diagnostic device.")


def _rl_image_from_array(image_array: np.ndarray, width_pt: float):
    """Converts a numpy RGB array to a reportlab Image flowable, preserving
    aspect ratio at the given target width.

    NOTE: pass the raw BytesIO buffer directly to RLImage — NOT wrapped in
    an ImageReader. This reportlab version's Image flowable calls
    os.path.splitext() on whatever it's given internally (to sniff JPEG
    vs other formats), which throws a TypeError on an ImageReader object
    since that isn't a str/path. A raw file-like buffer works correctly.
    """
    pil_img = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    aspect = pil_img.height / pil_img.width
    return RLImage(buffer, width=width_pt, height=width_pt * aspect)


def build_pdf_report(result: dict, original_image: np.ndarray, image_name: str = "N/A") -> bytes:
    """
    Built with reportlab's Platypus API (auto-flowing layout) rather than
    manual canvas coordinates — the previous canvas-based version left a
    mostly-empty second page because image positions were hardcoded rather
    than sized to fit the actual content.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Italic"], textColor=colors.HexColor("#4a3267"), fontSize=9
    )
    label_style = ParagraphStyle(
        "ImgLabel", parent=styles["Normal"], alignment=1, fontName="Helvetica-Bold", fontSize=9
    )
    story = []

    story.append(Paragraph("DR Screening Report", styles["Title"]))
    story.append(Paragraph(
        "AI-assisted screening output — ophthalmologist confirmation required.",
        subtitle_style
    ))
    story.append(Spacer(1, 12))

    q = result["quality"]
    pred = result["prediction"]
    unc = result["uncertainty"]
    lesions = result["lesions"]
    rec = result["recommendation"]

    info_rows = [
        ["Image", image_name],
        ["Quality Score", f"{q['score']}/100 ({q['status']})"],
        ["DR Grade", str(pred["dr_grade"]) if pred["dr_grade"] is not None else "N/A"],
        ["Referable DR", "Yes" if pred["referable"] else ("No" if pred["referable"] is not None else "N/A")],
        ["Uncertainty", unc["uncertainty_level"] if unc["uncertainty_level"] is not None else "N/A"],
        ["Microaneurysm candidates", str(lesions.get("microaneurysm_count", "N/A"))],
        ["Hemorrhage candidates", str(lesions.get("hemorrhage_count", "N/A"))],
        ["Exudate candidates", str(lesions.get("exudate_count", "N/A"))],
    ]
    info_table = Table(info_rows, colWidths=[2.2 * inch, 4.5 * inch])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3d9e5")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4a3267")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 8))

    rec_colors = {"routine": "#7fbf7f", "review": "#de638a", "recapture": "#c0392b"}
    rec_color = rec_colors.get(rec, "#4a3267")
    rec_table = Table([[f"RECOMMENDATION: {rec.upper()}"]], colWidths=[6.7 * inch])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(rec_color)),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Visual Evidence", styles["Heading2"]))
    story.append(Spacer(1, 4))

    img_width = 2.35 * inch
    top_labels = [Paragraph("Original", label_style), Paragraph("Enhanced", label_style)]
    top_images = [
        _rl_image_from_array(original_image, img_width),
        _rl_image_from_array(result["enhanced_image"], img_width),
    ]
    bottom_labels = [Paragraph("Grad-CAM", label_style), Paragraph("Lesion Overlay", label_style)]
    bottom_images = [
        _rl_image_from_array(result["gradcam"], img_width),
        _rl_image_from_array(lesions.get("overlay_image", result["gradcam"]), img_width),
    ]

    image_grid = Table(
        [top_labels, top_images, bottom_labels, bottom_images],
        colWidths=[img_width + 0.2 * inch] * 2,
    )
    image_grid.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
        ("TOPPADDING", (0, 3), (-1, 3), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
    ]))
    # KeepTogether forces this whole grid to move to the next page as a
    # single unit rather than letting Platypus split it mid-table — without
    # this, a label row (e.g. "Grad-CAM") could land at the bottom of one
    # page while its actual image flowed to the next, which is exactly what
    # testing caught before this fix.
    story.append(KeepTogether(image_grid))
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "Grad-CAM and lesion overlays are interpretability aids, not standalone "
        "clinical evidence. This is a screening/referral-support prototype, not "
        "an autonomous diagnostic device, and has not undergone clinical validation.",
        subtitle_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_resource(show_spinner="Loading DR model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_trained_model(MODEL_PATH)


model = load_model()
if model is None:
    st.warning(
        f"No trained model checkpoint found at {MODEL_PATH}. "
        "Predictions and Grad-CAM will use placeholder fallbacks."
    )
else:
    st.caption(f"Model loaded: {os.path.relpath(MODEL_PATH, PROJECT_ROOT)}")

if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0
if "last_ungradable_info" not in st.session_state:
    st.session_state["last_ungradable_info"] = None

uploaded_file = st.file_uploader(
    "Upload a fundus image",
    type=["png", "jpg", "jpeg"],
    key=f"fundus_upload_{st.session_state['uploader_version']}",
)

if uploaded_file is not None:
    # Day 5: review-time measurement, per spec Section 26 target (<30s/case).
    # Timer starts the moment a new image is uploaded.
    if st.session_state.get("current_file_id") != uploaded_file.file_id:
        st.session_state["current_file_id"] = uploaded_file.file_id
        st.session_state["review_start_time"] = time.time()
        st.session_state["decision_made"] = None
        st.session_state["modify_grade_placeholder"] = False
        st.session_state["recapture_requested"] = False
        st.session_state["last_ungradable_info"] = None

    original_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(original_image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS))
    result = run_pipeline(image, model=model)

    if result["quality"]["status"] == "ungradable":
        # Below the ungradable threshold, main.py's run_pipeline() skips
        # prediction/Grad-CAM entirely and just returns the input unchanged
        # (by design — don't run inference on an image that isn't fit to
        # grade). Previously this rendered three identical images with no
        # explanation. Now: redirect straight to a clear recapture prompt
        # and reset the uploader for a fresh image, instead of showing
        # placeholder-looking output.
        q = result["quality"]
        st.session_state["last_ungradable_info"] = {
            "score": q["score"],
            "reason": q.get("failure_reason"),
            "message": get_recapture_message(q.get("failure_reason")),
        }
        st.session_state["uploader_version"] += 1
        st.session_state["recapture_requested"] = True
        st.session_state.pop("current_file_id", None)
        st.session_state.pop("review_start_time", None)
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Enhanced")
        st.image(result["enhanced_image"], use_container_width=True)
    with col3:
        st.subheader("Grad-CAM")
        st.image(result["gradcam"], use_container_width=True)

    st.subheader("Prediction")
    pred = result["prediction"]
    unc = result["uncertainty"]
    q = result["quality"]

    # Day 2: guard against the ungradable path, where main.py sets these to
    # None rather than fabricating a grade for an image that can't be graded.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DR Grade", pred["dr_grade"] if pred["dr_grade"] is not None else "N/A")
    m2.metric("Referable", "Yes" if pred["referable"] else ("No" if pred["referable"] is not None else "N/A"))
    m3.metric("Quality Score", f"{q['score']}/100")
    m4.metric("Uncertainty", unc["uncertainty_level"] if unc["uncertainty_level"] is not None else "N/A")

    rec = result["recommendation"]
    color = {"routine": "#7fbf7f", "review": "#de638a", "recapture": "#c0392b"}.get(rec, "#4a3267")
    st.markdown(
        f'<div class="recommendation-banner" style="background-color:{color};color:white;">'
        f'Recommendation: {rec.upper()}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Full report"):
        st.text(result["report"])

    # Day 5: human-in-the-loop action buttons (spec Section 26). These are
    # UI-only for the hackathon scope — no backend persistence — but they
    # do stop the review-time timer, which is the actual research metric
    # (<30s/case target) worth measuring even without full case storage.
    st.subheader("Ophthalmologist Review")
    b1, b2, b3 = st.columns(3)

    if b1.button("Modify DR grade"):
        st.session_state["decision_made"] = "Modify DR grade"
        st.session_state["modify_grade_placeholder"] = True

    if b2.button("Request recapture"):
        st.session_state["uploader_version"] += 1
        st.session_state["decision_made"] = "Request recapture"
        st.session_state["modify_grade_placeholder"] = False
        st.session_state["recapture_requested"] = True
        st.session_state["last_ungradable_info"] = None
        st.session_state.pop("current_file_id", None)
        st.session_state.pop("review_start_time", None)
        st.rerun()

    b3.download_button(
        "Save report",
        data=build_pdf_report(result, image, image_name=uploaded_file.name),
        file_name="dr_screening_report.pdf",
        mime="application/pdf",
    )

    if st.session_state.get("modify_grade_placeholder"):
        st.info(
            "Modify DR grade placeholder: in the full system, this action would "
            "open an editable field where the ophthalmologist can override the "
            "AI-predicted DR grade with their own clinical assessment. This "
            "prototype records the intended workflow only."
        )

    if st.session_state.get("decision_made") and st.session_state.get("review_start_time"):
        elapsed = time.time() - st.session_state["review_start_time"]
        st.success(f"Decision recorded: **{st.session_state['decision_made']}** "
                   f"(review time: {elapsed:.1f}s)")
        if elapsed > 30:
            st.caption("Note: this exceeds the <30s/case research target — "
                       "fine during testing, worth tracking in the real review-time study.")
else:
    if st.session_state.get("recapture_requested"):
        info = st.session_state.get("last_ungradable_info")
        if info:
            st.error(
                f"**Recapture required** — quality score {info['score']}/100"
                + (f" ({info['reason'].replace('_', ' ')})" if info["reason"] else "")
            )
            st.info(info["message"])
        else:
            st.info("Recapture requested by reviewer.")
        st.caption("Please upload a new fundus image using the uploader above.")
    else:
        st.info("Upload a fundus image to begin.")
    st.session_state.pop("current_file_id", None)
