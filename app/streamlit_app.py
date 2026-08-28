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
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import run_pipeline  # noqa: E402

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

uploaded_file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    result = run_pipeline(image)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Enhanced")
        st.image(result["enhanced_image"], use_column_width=True)
    with col3:
        st.subheader("Grad-CAM")
        st.image(result["gradcam"], use_column_width=True)

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

    # TODO(P6): wire in Accept / Modify / Ungradable / Recapture / Refer / Save
    # action buttons here once the review-time study is in scope (Day 5+).
else:
    st.info("Upload a fundus image to begin.")
