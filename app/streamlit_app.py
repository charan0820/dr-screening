"""
P6 - Streamlit GUI for DR Screening.

The GUI communicates with the rest of the system only through
main.run_pipeline().
"""

import os
import sys

import numpy as np
import streamlit as st
from PIL import Image


# Allow importing main.py from the project root
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from main import run_pipeline  # noqa: E402


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="DR Screening Prototype",
    layout="wide"
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("DR Screening Prototype")

st.caption(
    "AI-assisted screening and referral support — "
    "not an autonomous diagnostic device."
)


# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a fundus image",
    type=["png", "jpg", "jpeg"]
)


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

if uploaded_file is not None:

    image = np.array(
        Image.open(uploaded_file).convert("RGB")
    )

    st.subheader("Processing...")

    result = run_pipeline(image)


    # -----------------------------------------------------
    # Images
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Enhanced")
        st.image(
            result["enhanced_image"],
            use_container_width=True
        )

    with col3:
        st.subheader("Grad-CAM")
        st.image(
            result["gradcam"],
            use_container_width=True
        )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    st.subheader("Screening Results")

    prediction = result["prediction"]
    uncertainty = result["uncertainty"]
    quality = result["quality"]

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "DR Grade",
            prediction["dr_grade"]
        )

    with m2:
        st.metric(
            "Referable",
            "Yes" if prediction["referable"] else "No"
        )

    with m3:
        st.metric(
            "Quality Score",
            f"{quality['score']}/100"
        )

    with m4:
        st.metric(
            "Uncertainty",
            uncertainty["uncertainty_level"]
        )


    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    recommendation = result["recommendation"]

    st.subheader("Recommendation")

    if recommendation == "routine":
        st.success("ROUTINE")

    elif recommendation == "review":
        st.warning("REVIEW BY OPHTHALMOLOGIST")

    elif recommendation == "recapture":
        st.error("RECAPTURE IMAGE")

    else:
        st.info(recommendation.upper())


    # -----------------------------------------------------
    # Lesion information
    # -----------------------------------------------------

    lesions = result["lesions"]

    st.subheader("Lesion Evidence")

    l1, l2, l3 = st.columns(3)

    with l1:
        st.metric(
            "Microaneurysm Candidates",
            lesions.get("microaneurysm_count", "N/A")
        )

    with l2:
        st.metric(
            "Hemorrhage Candidates",
            lesions.get("hemorrhage_count", "N/A")
        )

    with l3:
        st.metric(
            "Exudate Candidates",
            lesions.get("exudate_count", "N/A")
        )


    # -----------------------------------------------------
    # Full report
    # -----------------------------------------------------

    with st.expander("Full Screening Report"):
        st.text(result["report"])

else:

    st.info(
        "Upload a fundus image to begin screening."
    )