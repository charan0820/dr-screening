"""
OCULUS AI Streamlit interface.

The screening pipeline remains the source of truth for image quality,
enhancement, classification, uncertainty, explainability, lesions, and
recommendations. This module owns the presentation layer and report delivery.
"""
import base64
import html
import json
import os
import re
import smtplib
import subprocess
import sys
import time
import urllib.parse
from datetime import date
from email.message import EmailMessage
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_checkpoint.keras")
MODEL_INPUT_SIZE = (224, 224)
ASSET_DIR = os.path.join(PROJECT_ROOT, "attached_assets")

sys.path.append(PROJECT_ROOT)
from main import run_pipeline  # noqa: E402
from src.model import load_trained_model  # noqa: E402
from src.quality import get_recapture_message  # noqa: E402


st.set_page_config(
    page_title="OCULUS AI | Retinal Screening",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PALETTE = {
    "navy": "#001C30",
    "teal": "#176B87",
    "aqua": "#64CCC5",
    "mist": "#DAFFFB",
    "ink": "#082A3D",
    "muted": "#55717C",
    "line": "#B3DCD9",
    "coral": "#D97A62",
    "green": "#267E68",
    "amber": "#B77B28",
    "red": "#A84843",
}


def _asset_data_uri(filename: str, *, remove_dark_background: bool = False) -> str:
    """Prepare a small self-contained asset for the CSS background layers."""
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(path):
        return ""

    try:
        image = Image.open(path).convert("RGBA")
        image.thumbnail((1100, 700), Image.Resampling.LANCZOS)
        pixels = np.array(image)

        if filename.startswith("bg_eye1"):
            # The supplied PNG has a checkerboard baked into the image rather
            # than transparency. Remove only neutral, light checker squares.
            rgb = pixels[:, :, :3]
            neutral = (
                (np.max(rgb, axis=2) - np.min(rgb, axis=2) <= 40)
                & (np.mean(rgb, axis=2) > 80)
            )
            pixels[neutral, 3] = 0
            image = Image.fromarray(pixels)
        elif remove_dark_background:
            rgb = pixels[:, :, :3]
            dark = np.max(rgb, axis=2) < 95
            pixels[dark, 3] = 0
            image = Image.fromarray(pixels)

        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except (OSError, ValueError):
        return ""


def _raw_asset_data_uri(filename: str) -> str:
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as asset:
            encoded = base64.b64encode(asset.read()).decode("ascii")
        extension = os.path.splitext(filename)[1].lower().lstrip(".")
        mime = "image/avif" if extension == "avif" else f"image/{extension}"
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return ""


EYE_ONE = _asset_data_uri("bg_eye1_1788438058172.png")
EYE_TWO = _asset_data_uri("bg_eye2_1788438058173.jpeg", remove_dark_background=True)
BUFFERING_EYE = _raw_asset_data_uri("buffering_eye_1788438058173.avif")

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {{
    --navy: {PALETTE["navy"]};
    --teal: {PALETTE["teal"]};
    --aqua: {PALETTE["aqua"]};
    --mist: {PALETTE["mist"]};
    --ink: {PALETTE["ink"]};
    --muted: {PALETTE["muted"]};
    --line: {PALETTE["line"]};
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--mist);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stAppViewContainer"] > .main {{
    position: relative;
    z-index: 1;
}}

[data-testid="stMainBlockContainer"] {{
    max-width: 1420px;
    padding: 1.15rem clamp(1rem, 3vw, 3.5rem) 4rem;
}}

.eye-background {{
    position: fixed;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
    z-index: 0;
    background: radial-gradient(circle at 74% 26%, rgba(100, 204, 197, .14), transparent 34%);
}}

.eye-layer {{
    position: absolute;
    right: -9vw;
    top: 4vh;
    width: min(62vw, 860px);
    height: min(62vw, 860px);
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    border-radius: 50%;
    clip-path: ellipse(44% 47% at 50% 50%);
    opacity: .10;
    animation: eyeCycle 24s ease-in-out infinite;
}}
.eye-layer.one {{ background-image: url("{EYE_ONE}"); animation-delay: 0s; }}
.eye-layer.one {{ clip-path: ellipse(35% 50% at 50% 50%); }}
.eye-layer.two {{
    background-image: url("{EYE_TWO}");
    animation-delay: -8s;
    clip-path: ellipse(35% 50% at 50% 50%);
}}
.eye-layer.three {{
    background-image: url("{BUFFERING_EYE}");
    animation-delay: -16s;
    opacity: .06;
}}
@keyframes eyeCycle {{
    0%, 28% {{ opacity: .10; transform: scale(1) rotate(-3deg); }}
    34%, 61% {{ opacity: 0; transform: scale(1.04) rotate(2deg); }}
    68%, 95% {{ opacity: .08; transform: scale(1.02) rotate(-1deg); }}
    100% {{ opacity: .10; transform: scale(1) rotate(-3deg); }}
}}

.brand-mark {{
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.2rem, 2vw, 1.6rem);
    font-weight: 800;
    letter-spacing: .08em;
    color: var(--mist);
}}
.brand-mark span {{ color: var(--aqua); }}
.brand-subtitle {{
    color: rgba(218, 255, 251, .72);
    font-size: .74rem;
    letter-spacing: .05em;
    margin-top: .15rem;
}}
.topbar {{
    background: var(--navy);
    border-radius: 18px;
    padding: .9rem 1.2rem;
    margin-bottom: 1.35rem;
    box-shadow: 0 12px 30px rgba(0, 28, 48, .12);
}}
.topbar [data-testid="stHorizontalBlock"] {{ align-items: center; }}
.brand-wrap {{
    background: var(--navy);
    border-radius: 16px;
    padding: .7rem .9rem;
}}

.hero-kicker {{
    color: var(--teal);
    font-size: .74rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin: .9rem 0 .5rem;
}}
.hero-title {{
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2rem, 5vw, 4.25rem);
    font-weight: 800;
    letter-spacing: -.065em;
    line-height: .98;
    max-width: 820px;
    margin: 0;
}}
.hero-copy {{
    color: var(--muted);
    font-size: clamp(1rem, 1.5vw, 1.18rem);
    line-height: 1.55;
    max-width: 610px;
    margin: 1rem 0 1.35rem;
}}
.fact-carousel {{
    position: relative;
    min-height: 2.2rem;
    color: var(--teal);
    font-size: .88rem;
    font-weight: 600;
    max-width: 500px;
    overflow: hidden;
}}
.fact-carousel span {{
    position: absolute;
    inset: 0;
    opacity: 0;
    animation: factCycle 20s linear infinite;
}}
.fact-carousel span:nth-child(2) {{ animation-delay: 5s; }}
.fact-carousel span:nth-child(3) {{ animation-delay: 10s; }}
.fact-carousel span:nth-child(4) {{ animation-delay: 15s; }}
@keyframes factCycle {{
    0%, 5% {{ opacity: 0; transform: translateY(8px); }}
    8%, 22% {{ opacity: 1; transform: translateY(0); }}
    25%, 100% {{ opacity: 0; transform: translateY(-8px); }}
}}

.section-label {{
    color: var(--teal);
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: .55rem;
}}
.section-title {{
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.3rem, 2vw, 1.8rem);
    font-weight: 800;
    letter-spacing: -.035em;
    margin: 0 0 .4rem;
}}
.section-copy {{ color: var(--muted); line-height: 1.55; }}
.card {{
    background: rgba(255, 255, 255, .76);
    border: 1px solid rgba(23, 107, 135, .14);
    border-radius: 22px;
    padding: clamp(1rem, 2vw, 1.45rem);
    box-shadow: 0 14px 40px rgba(0, 28, 48, .07);
    backdrop-filter: blur(10px);
}}
.upload-card {{
    background: var(--navy);
    border-radius: 22px;
    color: var(--mist);
    padding: clamp(1rem, 2vw, 1.45rem);
    box-shadow: 0 18px 45px rgba(0, 28, 48, .13);
}}
.upload-card .section-title, .upload-card .section-copy {{ color: var(--mist); }}
.upload-card .section-copy {{ opacity: .75; }}
.preview-card {{
    border: 1px solid rgba(100, 204, 197, .34);
    border-radius: 16px;
    overflow: hidden;
    background: rgba(218, 255, 251, .06);
    margin-top: .9rem;
}}
.preview-card img {{ display: block; width: 100%; }}
.preview-caption {{
    display: flex;
    justify-content: space-between;
    gap: .5rem;
    padding: .65rem .75rem;
    color: rgba(218, 255, 251, .78);
    font-size: .77rem;
}}

.result-hero {{
    border-radius: 24px;
    padding: clamp(1.1rem, 2.4vw, 1.75rem);
    background: var(--navy);
    color: var(--mist);
    box-shadow: 0 18px 45px rgba(0, 28, 48, .16);
}}
.result-hero h2 {{ color: var(--mist); margin: 0; }}
.result-hero p {{ color: rgba(218, 255, 251, .72); margin: .35rem 0 0; }}
.result-pill {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: .38rem .7rem;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .11em;
    margin-bottom: .8rem;
}}
.result-pill.routine {{ color: #d8fff0; background: rgba(38, 126, 104, .4); }}
.result-pill.review {{ color: #fff0d7; background: rgba(183, 123, 40, .46); }}
.result-pill.recapture {{ color: #ffe3e0; background: rgba(168, 72, 67, .5); }}
.grade-number {{
    color: var(--aqua);
    font-family: 'Manrope', sans-serif;
    font-size: clamp(3rem, 7vw, 5.8rem);
    font-weight: 800;
    letter-spacing: -.08em;
    line-height: .9;
}}
.grade-name {{ font-size: 1.1rem; font-weight: 700; margin-top: .4rem; }}
.metric-card {{
    background: rgba(255,255,255,.83);
    border: 1px solid rgba(23, 107, 135, .13);
    border-radius: 16px;
    padding: 1rem;
    height: 100%;
}}
.metric-label {{
    color: var(--muted);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}}
.metric-value {{
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: .38rem;
}}
.recommendation {{
    border-radius: 18px;
    padding: 1rem 1.15rem;
    border-left: 5px solid var(--aqua);
    background: rgba(100, 204, 197, .16);
}}
.recommendation.review {{ border-left-color: #D29A49; background: rgba(210, 154, 73, .14); }}
.recommendation.recapture {{ border-left-color: #B9534A; background: rgba(185, 83, 74, .13); }}
.recommendation strong {{ color: var(--navy); font-family: 'Manrope', sans-serif; }}
.evidence-label {{
    color: var(--navy);
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .07em;
    text-transform: uppercase;
    margin-bottom: .45rem;
}}
.disclaimer {{
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: .77rem;
    line-height: 1.5;
    margin-top: 2.5rem;
    padding-top: 1rem;
}}
.processing-card {{
    align-items: center;
    background: rgba(23, 107, 135, .13);
    border: 1px solid rgba(23, 107, 135, .18);
    border-radius: 16px;
    display: flex;
    gap: .8rem;
    padding: .8rem 1rem;
}}
.processing-card img {{
    animation: spin 1.5s linear infinite;
    height: 48px;
    object-fit: contain;
    width: 48px;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

.stTextInput input, .stNumberInput input, .stDateInput input {{
    border-radius: 10px;
}}
.stButton > button, .stDownloadButton > button {{
    border: 0;
    border-radius: 11px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    min-height: 2.8rem;
    transition: transform .18s ease, box-shadow .18s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    box-shadow: 0 8px 18px rgba(23, 107, 135, .18);
    transform: translateY(-1px);
}}
.upload-card + div .stButton > button {{
    background: var(--aqua);
    color: var(--navy);
}}
.topbar .stTextInput label, .topbar .stSelectbox label {{ color: rgba(218,255,251,.72); }}
.topbar .stTextInput input {{ background: rgba(255,255,255,.1); color: var(--mist); border-color: rgba(218,255,251,.22); }}
.topbar .stSelectbox div[data-baseweb="select"] > div {{ background: rgba(255,255,255,.1); color: var(--mist); border-color: rgba(218,255,251,.22); }}
.topbar [data-testid="stWidgetLabel"] p {{ color: rgba(218,255,251,.72); }}

@media (max-width: 700px) {{
    [data-testid="stMainBlockContainer"] {{ padding: .75rem .75rem 3rem; }}
    .topbar {{ border-radius: 14px; padding: .8rem; }}
    .eye-layer {{ right: -24vw; top: 18vh; width: 100vw; opacity: .07; }}
    .hero-title {{ font-size: 2.55rem; }}
    .result-hero {{ border-radius: 18px; }}
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _render_background() -> None:
    st.markdown(
        f"""
        <div class="eye-background" aria-hidden="true">
            <div class="eye-layer one"></div>
            <div class="eye-layer two"></div>
            <div class="eye-layer three"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _rl_image_from_array(image_array: np.ndarray, width_pt: float):
    """Convert an RGB array into a reportlab image while preserving its ratio."""
    pil_img = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    aspect = pil_img.height / pil_img.width
    return RLImage(buffer, width=width_pt, height=width_pt * aspect)


def build_pdf_report(
    result: dict,
    original_image: np.ndarray,
    image_name: str = "N/A",
    patient_id: str = "N/A",
    age: int | None = None,
    screening_date: date | None = None,
    eye: str = "Not specified",
) -> bytes:
    """Build the existing rule-based screening result as a downloadable PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Italic"],
        textColor=colors.HexColor(PALETTE["teal"]),
        fontSize=9,
    )
    label_style = ParagraphStyle(
        "ImgLabel",
        parent=styles["Normal"],
        alignment=1,
        fontName="Helvetica-Bold",
        fontSize=9,
    )
    story = [
        Paragraph("OCULUS AI — Screening Report", styles["Title"]),
        Paragraph(
            "AI-assisted screening output — ophthalmologist confirmation required.",
            subtitle_style,
        ),
        Spacer(1, 12),
    ]

    q = result["quality"]
    pred = result["prediction"]
    unc = result["uncertainty"]
    lesions = result["lesions"]
    rec = result["recommendation"]
    age_value = str(age) if age is not None else "Not provided"
    date_value = screening_date.isoformat() if screening_date else "Not provided"
    referable = (
        "Yes"
        if pred["referable"]
        else ("No" if pred["referable"] is not None else "N/A")
    )
    info_rows = [
        ["Patient ID", patient_id or "Not provided"],
        ["Age", age_value],
        ["Screening date", date_value],
        ["Eye", eye],
        ["Image", image_name],
        ["Quality score", f"{q['score']}/100 ({q['status']})"],
        ["DR grade", str(pred["dr_grade"]) if pred["dr_grade"] is not None else "N/A"],
        ["Referable DR", referable],
        ["Uncertainty", unc["uncertainty_level"] or "N/A"],
        ["Microaneurysm candidates", str(lesions.get("microaneurysm_count", "N/A"))],
        ["Hemorrhage candidates", str(lesions.get("hemorrhage_count", "N/A"))],
        ["Exudate candidates", str(lesions.get("exudate_count", "N/A"))],
    ]
    info_table = Table(info_rows, colWidths=[2.2 * inch, 4.5 * inch])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(PALETTE["mist"])),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor(PALETTE["teal"])),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B3DCD9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([info_table, Spacer(1, 8)])

    rec_colors = {"routine": "#267E68", "review": "#B77B28", "recapture": "#A84843"}
    rec_table = Table(
        [[f"RECOMMENDATION: {rec.upper()}"]],
        colWidths=[6.7 * inch],
    )
    rec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(rec_colors.get(rec, PALETTE["teal"]))),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([rec_table, Spacer(1, 10), Paragraph("Visual evidence", styles["Heading2"]), Spacer(1, 4)])

    img_width = 2.35 * inch
    image_grid = Table(
        [
            [Paragraph("Original", label_style), Paragraph("Enhanced", label_style)],
            [
                _rl_image_from_array(original_image, img_width),
                _rl_image_from_array(result["enhanced_image"], img_width),
            ],
            [Paragraph("Grad-CAM", label_style), Paragraph("Lesion overlay", label_style)],
            [
                _rl_image_from_array(result["gradcam"], img_width),
                _rl_image_from_array(
                    lesions.get("overlay_image", result["gradcam"]), img_width
                ),
            ],
        ],
        colWidths=[img_width + 0.2 * inch] * 2,
    )
    image_grid.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 1), (-1, 1), 4),
                ("TOPPADDING", (0, 3), (-1, 3), 4),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
            ]
        )
    )
    story.extend(
        [
            KeepTogether(image_grid),
            Spacer(1, 14),
            Paragraph(
                "Grad-CAM and lesion overlays are interpretability aids, not standalone clinical evidence. "
                "This is a screening/referral-support prototype, not an autonomous diagnostic device, "
                "and has not undergone clinical validation.",
                subtitle_style,
            ),
        ]
    )
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_resource(show_spinner="Loading the screening model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_trained_model(MODEL_PATH)


def _is_valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def _send_pdf_email(
    recipient: str,
    pdf_bytes: bytes,
    *,
    patient_id: str,
    image_name: str,
) -> tuple[bool | None, str]:
    """
    Send through configured SMTP or the attached Resend connector.

    SMTP values are read only by the running app. When SMTP_HOST is absent,
    the Resend connector bridge uses Replit-managed authentication instead of
    requiring a provider API key in the app. If neither path is configured,
    return a clear fallback state instead of pretending delivery happened.
    """
    recipient = recipient.strip()
    if not _is_valid_email(recipient):
        return False, "Enter a valid recipient email address."

    host = os.getenv("SMTP_HOST", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip()
    if not host and not sender:
        return None, "Email service is not configured. Download the PDF and use the email client link below."

    if not host:
        return _send_pdf_via_resend(
            recipient,
            pdf_bytes,
            sender=sender,
            patient_id=patient_id,
            image_name=image_name,
        )

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        username = os.getenv("SMTP_USERNAME", "").strip()
        password = os.getenv("SMTP_PASSWORD", "")
        smtp_sender = sender or username
        if not smtp_sender:
            return False, "Email service is missing a sender address."

        message = EmailMessage()
        message["Subject"] = f"OCULUS AI screening report — {patient_id or image_name}"
        message["From"] = smtp_sender
        message["To"] = recipient
        message.set_content(
            "Attached is the OCULUS AI diabetic retinopathy screening report. "
            "This screening output requires ophthalmologist confirmation."
        )
        message.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename="oculus-ai-screening-report.pdf",
        )

        with smtplib.SMTP(host, port, timeout=15) as smtp:
            if os.getenv("SMTP_USE_TLS", "true").lower() not in {"0", "false", "no"}:
                smtp.starttls()
            if username:
                smtp.login(username, password)
            smtp.send_message(message)
        return True, f"Report sent to {recipient}."
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        return False, f"Report could not be sent: {exc}"


def _send_pdf_via_resend(
    recipient: str,
    pdf_bytes: bytes,
    *,
    sender: str,
    patient_id: str,
    image_name: str,
) -> tuple[bool, str]:
    """Send a PDF through the Replit-managed Resend connector."""
    payload = {
        "from": sender,
        "to": [recipient],
        "subject": f"OCULUS AI screening report — {patient_id or image_name}",
        "text": (
            "Attached is the OCULUS AI diabetic retinopathy screening report. "
            "This screening output requires ophthalmologist confirmation."
        ),
        "attachments": [
            {
                "filename": "oculus-ai-screening-report.pdf",
                "content": base64.b64encode(pdf_bytes).decode("ascii"),
            }
        ],
    }

    try:
        completed = subprocess.run(
            ["node", os.path.join(PROJECT_ROOT, "scripts", "send_resend_email.cjs")],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Report could not be sent: {exc}"

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "The Resend connector could not process the request."
        return False, f"Report could not be sent: {detail}"

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, "Report could not be sent: the Resend connector returned an invalid response."

    if response.get("ok") is True:
        return True, f"Report sent to {recipient}."

    error_body = response.get("body")
    if isinstance(error_body, dict):
        detail = error_body.get("message") or error_body.get("error")
    else:
        detail = None
    detail = detail or f"Resend returned HTTP {response.get('status', 'unknown')}."
    return False, f"Report could not be sent: {detail}"


def _mailto_link(recipient: str, result: dict) -> str:
    subject = "OCULUS AI screening report"
    body = (
        "The OCULUS AI screening report is ready.\n\n"
        f"Recommendation: {result.get('recommendation', 'N/A').upper()}\n"
        "Download the PDF from the screening page and attach it to this email."
    )
    query = urllib.parse.urlencode({"subject": subject, "body": body})
    return f"mailto:{urllib.parse.quote(recipient.strip())}?{query}"


def _safe_upload_image(uploaded_file) -> Image.Image | None:
    try:
        uploaded_file.seek(0)
        return Image.open(uploaded_file).convert("RGB")
    except (OSError, ValueError):
        return None


def _grade_name(grade) -> str:
    names = {
        0: "No diabetic retinopathy",
        1: "Mild diabetic retinopathy",
        2: "Moderate diabetic retinopathy",
        3: "Severe diabetic retinopathy",
        4: "Proliferative diabetic retinopathy",
    }
    return names.get(grade, "Unable to grade")


def _risk_label(prediction: dict) -> str:
    grade = prediction.get("dr_grade")
    if grade is None:
        return "Not gradable"
    return "Referable risk" if int(grade) >= 2 else "Lower risk"


def _recommendation_copy(recommendation: str, failure_reason: str | None = None) -> tuple[str, str]:
    if recommendation == "routine":
        return "Routine screening", "No immediate referral indicated. Continue routine screening."
    if recommendation == "review":
        return "Clinical review recommended", "Clinical examination or referral is recommended."
    return "Image recapture required", get_recapture_message(failure_reason)


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def _reset_analysis(*, clear_upload: bool = False) -> None:
    st.session_state["analysis_result"] = None
    st.session_state["analysis_image"] = None
    st.session_state["analysis_pdf"] = None
    st.session_state["email_status"] = None
    st.session_state["analysis_patient"] = None
    if clear_upload:
        st.session_state["uploader_version"] += 1


def _render_menu_content(menu_choice: str) -> None:
    content = {
        "About us": (
            "About OCULUS AI",
            "A screening and referral-support prototype designed to help PHC teams make faster, clearer decisions with retinal images.",
        ),
        "Explore our model": (
            "Explore our model",
            "The existing workflow assesses image quality, enhances suitable images, classifies five DR grades, estimates uncertainty, and provides visual evidence for clinical review.",
        ),
        "Our codebase": (
            "Our codebase",
            "This project keeps the model and clinical logic separate from the Streamlit interface so the screening workflow remains inspectable.",
        ),
        "References": (
            "References",
            "The prototype uses the APTOS 2019 Blindness Detection dataset and includes a clear clinical-validation disclaimer.",
        ),
    }
    if menu_choice in content:
        title, copy = content[menu_choice]
        st.info(f"**{title}**\n\n{copy}")


_render_background()

for key, default in {
    "uploader_version": 0,
    "selected_file_id": None,
    "analysis_result": None,
    "analysis_image": None,
    "analysis_pdf": None,
    "analysis_patient": None,
    "email_status": None,
    "recapture_requested": False,
    "decision_made": None,
    "modify_grade_placeholder": False,
}.items():
    st.session_state.setdefault(key, default)


with st.container():
    nav_col, menu_col, email_col = st.columns([2.1, 1.25, 1.7], vertical_alignment="center")
    with nav_col:
        st.markdown(
            '<div class="brand-wrap"><div class="brand-mark">OCULUS <span>AI</span></div>'
            '<div class="brand-subtitle">DIABETIC RETINOPATHY SCREENING</div></div>',
            unsafe_allow_html=True,
        )
    with menu_col:
        menu_choice = st.selectbox(
            "Menu",
            ["Menu", "About us", "Explore our model", "Our codebase", "References"],
            key="menu_choice",
            label_visibility="collapsed",
        )
    with email_col:
        st.text_input(
            "Report email",
            placeholder="Receiver email",
            key="receiver_email",
            label_visibility="collapsed",
        )

_render_menu_content(menu_choice)

st.markdown('<div class="hero-kicker">Rural PHC screening support</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">A clearer view of retinal health.</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-copy">AI-assisted retinal screening for early detection and referral — '
    "designed for confident decisions when specialist access is limited.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="fact-carousel">'
    "<span>Regular screening can help identify changes before vision is affected.</span>"
    "<span>Good image quality helps the screening model make a safer assessment.</span>"
    "<span>Diabetic retinopathy can progress without noticeable symptoms.</span>"
    "<span>Every result is referral support, not a final diagnosis.</span>"
    "</div>",
    unsafe_allow_html=True,
)

model = load_model()
if model is None:
    st.warning("No trained checkpoint was found. The existing placeholder fallback will be used.")

st.markdown("<br>", unsafe_allow_html=True)
input_col, details_col = st.columns([1.18, 0.82], gap="large")

with input_col:
    st.markdown('<div class="section-label">Step 01 · Image input</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Upload a retinal image</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Use a clear, centered fundus photograph. '
        "The image will be checked before any DR prediction is made.</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=["png", "jpg", "jpeg"],
        key=f"fundus_upload_{st.session_state['uploader_version']}",
        help="Accepted formats: PNG, JPG, JPEG.",
    )

    if uploaded_file is not None:
        file_id = getattr(
            uploaded_file,
            "file_id",
            f"{uploaded_file.name}:{getattr(uploaded_file, 'size', '')}",
        )
        if st.session_state["selected_file_id"] != file_id:
            st.session_state["selected_file_id"] = file_id
            _reset_analysis()
            st.session_state["recapture_requested"] = False

        preview = _safe_upload_image(uploaded_file)
        if preview is None:
            st.error("This file could not be opened as an image. Please choose a PNG or JPG fundus image.")
        else:
            preview_buffer = BytesIO()
            preview.save(preview_buffer, format="JPEG", quality=86)
            preview_uri = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode("ascii")
            st.markdown(
                f'<div class="preview-card"><img src="{preview_uri}" alt="Selected retinal image">'
                f'<div class="preview-caption"><span>{html.escape(uploaded_file.name)}</span>'
                "<span>Image ready</span></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Analyze image",
                type="primary",
                width="stretch",
                disabled=st.session_state["analysis_result"] is not None,
            ):
                st.session_state["email_status"] = None
                processing = st.empty()
                progress = st.progress(8, text="Preparing image quality checks…")
                processing.markdown(
                    f'<div class="processing-card"><img src="{BUFFERING_EYE}" alt="">'
                    "<div><strong>Analysis in progress</strong><br>"
                    '<span style="color:#55717C">Checking quality, evidence, and referral risk…</span></div></div>',
                    unsafe_allow_html=True,
                )
                try:
                    image = np.array(preview.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS))
                    progress.progress(30, text="Assessing image quality…")
                    result = run_pipeline(image, model=model)
                    progress.progress(82, text="Preparing clinical summary…")
                    patient = {
                        "patient_id": st.session_state.get("patient_id", "").strip(),
                        "age": st.session_state.get("patient_age", 45),
                        "screening_date": st.session_state.get("screening_date", date.today()),
                        "eye": st.session_state.get("screening_eye", "Right"),
                    }
                    pdf = build_pdf_report(
                        result,
                        image,
                        image_name=uploaded_file.name,
                        patient_id=patient["patient_id"],
                        age=patient["age"],
                        screening_date=patient["screening_date"],
                        eye=patient["eye"],
                    )
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_image"] = image
                    st.session_state["analysis_pdf"] = pdf
                    st.session_state["analysis_patient"] = patient
                    recipient = st.session_state.get("receiver_email", "").strip()
                    if recipient:
                        st.session_state["email_status"] = _send_pdf_email(
                            recipient,
                            pdf,
                            patient_id=patient["patient_id"],
                            image_name=uploaded_file.name,
                        )
                    progress.progress(100, text="Analysis ready")
                    time.sleep(.25)
                    st.rerun()
                except Exception as exc:
                    progress.empty()
                    processing.error(f"Processing failed: {exc}")


with details_col:
    st.markdown('<div class="section-label">Step 02 · Screening information</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Patient context</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Use the minimum information needed to identify this screening case.</p>',
        unsafe_allow_html=True,
    )
    st.text_input("Patient ID", placeholder="e.g. PHC-2026-001", key="patient_id")
    st.number_input("Age", min_value=1, max_value=120, value=45, step=1, key="patient_age")
    st.date_input("Screening date", value=date.today(), key="screening_date")
    st.radio("Eye being screened", ["Right", "Left"], horizontal=True, key="screening_eye")

    if uploaded_file is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><div class="section-label">Ready when you are</div>'
            '<h2 class="section-title">Start with one image</h2>'
            '<p class="section-copy">Upload a fundus image on the left. The analysis button will appear after the image is ready.</p></div>',
            unsafe_allow_html=True,
        )


result = st.session_state.get("analysis_result")
analysis_image = st.session_state.get("analysis_image")
analysis_patient = st.session_state.get("analysis_patient") or {}

if result is not None and analysis_image is not None:
    pred = result["prediction"]
    unc = result["uncertainty"]
    quality = result["quality"]
    recommendation = result["recommendation"]
    rec_title, rec_copy = _recommendation_copy(recommendation, quality.get("failure_reason"))
    grade = pred.get("dr_grade")
    grade_display = str(grade) if grade is not None else "—"
    risk = _risk_label(pred)
    pill_label = {
        "routine": "ROUTINE SCREENING",
        "review": "CLINICAL REVIEW RECOMMENDED",
        "recapture": "IMAGE RECAPTURE REQUIRED",
    }.get(recommendation, recommendation.upper())

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Step 03 · Screening result</div>', unsafe_allow_html=True)
    hero_left, hero_right = st.columns([1.2, 1], gap="large")
    with hero_left:
        st.markdown(
            f'<div class="result-hero"><div class="result-pill {recommendation}">{pill_label}</div>'
            f'<div class="grade-number">{html.escape(grade_display)}</div>'
            f'<div class="grade-name">{html.escape(_grade_name(grade))}</div>'
            '<p>The model’s five-class result is shown above. Use the recommendation as referral support, '
            "then confirm clinically.</p></div>",
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        _metric_card("Risk category", risk)
        st.markdown("<br>", unsafe_allow_html=True)
        _metric_card("Image quality", f"{quality['score']}/100")
        st.markdown("<br>", unsafe_allow_html=True)
        _metric_card("Uncertainty", str(unc.get("uncertainty_level") or "N/A").title())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    rec_class = "" if recommendation == "routine" else recommendation
    st.markdown(
        f'<div class="recommendation {rec_class}"><strong>{html.escape(rec_title)}</strong>'
        f'<br><span>{html.escape(rec_copy)}</span></div>',
        unsafe_allow_html=True,
    )

    if recommendation == "recapture":
        st.warning(
            f"Image quality is {quality['score']}/100. "
            f"{get_recapture_message(quality.get('failure_reason'))}"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Quality and evidence</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">See what the system saw</h2>', unsafe_allow_html=True)
    image_col1, image_col2, image_col3 = st.columns(3, gap="medium")
    evidence = [
        (image_col1, "Original image", analysis_image),
        (image_col2, "Enhanced image", result["enhanced_image"]),
        (image_col3, "Grad-CAM", result["gradcam"]),
    ]
    for column, label, image in evidence:
        with column:
            st.markdown(f'<div class="evidence-label">{label}</div>', unsafe_allow_html=True)
            st.image(image, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Why did the AI make this prediction?</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Highlighted regions indicate areas that contributed to the AI prediction. '
            "They are visual aids for review, not standalone clinical evidence.</p>",
            unsafe_allow_html=True,
        )
        explain_col1, explain_col2 = st.columns([1, 1.15], gap="large")
        with explain_col1:
            st.image(result["gradcam"], width="stretch")
        with explain_col2:
            st.markdown(
                f'<div class="card"><div class="metric-label">Model interpretation</div>'
                f'<div class="metric-value">{html.escape(_grade_name(grade))}</div>'
                f'<p class="section-copy">The Grad-CAM view helps a reviewer see the retinal regions '
                "that influenced this five-class prediction. Always review the original and enhanced images together.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Clinical recommendation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="recommendation {rec_class}"><strong>{html.escape(recommendation.upper())}</strong>'
        f'<br><span>{html.escape(rec_copy)}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Results summary</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Case summary</h2>', unsafe_allow_html=True)
    summary_left, summary_right = st.columns([.9, 1.6], gap="large")
    with summary_left:
        _metric_card("Patient ID", analysis_patient.get("patient_id") or "Not provided")
        st.markdown("<br>", unsafe_allow_html=True)
        _metric_card("Image quality", f"{quality['score']}/100")
        st.markdown("<br>", unsafe_allow_html=True)
        _metric_card("DR grade", grade_display)
        st.markdown("<br>", unsafe_allow_html=True)
        _metric_card("Risk level", risk)
        st.markdown("<br>", unsafe_allow_html=True)
        _metric_card("Referral recommendation", recommendation.title())
    with summary_right:
        summary_images = st.columns(2, gap="medium")
        summary_evidence = [
            ("Original image", analysis_image),
            ("Enhanced image", result["enhanced_image"]),
            ("Grad-CAM", result["gradcam"]),
            ("Lesion overlay", result["lesions"].get("overlay_image", result["gradcam"])),
        ]
        for column, (label, image) in zip(summary_images * 2, summary_evidence):
            with column:
                st.markdown(f'<div class="evidence-label">{label}</div>', unsafe_allow_html=True)
                st.image(image, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns([1, 1, 1.35], gap="medium")
    if action_col1.button("Modify DR grade", width="stretch"):
        st.session_state["decision_made"] = "Modify DR grade"
        st.session_state["modify_grade_placeholder"] = True
    if action_col2.button("Request recapture", width="stretch"):
        st.session_state["decision_made"] = "Request recapture"
        st.session_state["recapture_requested"] = True
        _reset_analysis(clear_upload=True)
        st.rerun()
    with action_col3:
        st.download_button(
            "Download PDF report",
            data=st.session_state["analysis_pdf"],
            file_name="oculus-ai-screening-report.pdf",
            mime="application/pdf",
            width="stretch",
        )

    if st.session_state.get("modify_grade_placeholder"):
        st.info(
            "Clinician override is not persisted in this prototype. The displayed grade remains the model prediction."
        )

    email_status = st.session_state.get("email_status")
    if email_status:
        sent, message = email_status
        if sent is True:
            st.success(message)
        elif sent is False:
            st.error(message)
        else:
            st.info(message)

    recipient = st.session_state.get("receiver_email", "").strip()
    if recipient and _is_valid_email(recipient):
        email_col1, email_col2 = st.columns([1, 1.4], gap="medium")
        with email_col1:
            if st.button("Send report by email", width="stretch"):
                status = _send_pdf_email(
                    recipient,
                    st.session_state["analysis_pdf"],
                    patient_id=analysis_patient.get("patient_id", ""),
                    image_name=uploaded_file.name if uploaded_file else "screening",
                )
                st.session_state["email_status"] = status
                st.rerun()
        with email_col2:
            st.markdown(
                f'<a href="{_mailto_link(recipient, result)}" target="_blank" '
                'style="color:#176B87;font-weight:700;line-height:2.8rem;">'
                "Open email client</a>",
                unsafe_allow_html=True,
            )

    with st.expander("View full rule-based report"):
        st.text(result["report"])

elif uploaded_file is None:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.get("recapture_requested"):
        st.warning("Please upload a new fundus image to continue the screening.")

st.markdown(
    '<div class="disclaimer">OCULUS AI is a screening and referral-support prototype, not an autonomous '
    "diagnostic device. Results require ophthalmologist confirmation and this system has not undergone clinical validation. "
    "Keep patient information private and follow your PHC’s clinical workflow.</div>",
    unsafe_allow_html=True,
)