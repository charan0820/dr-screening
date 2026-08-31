"""
Owner: P6
Rule-based, templated report generator. No LLM calls — every field comes
directly from the pipeline_output dict. This is a hard requirement:
never let anything invent or infer a finding that isn't in the data.
See CONTRACTS.md for the expected pipeline_output shape.
"""
import io
import numpy as np
from PIL import Image as PILImage


def generate_report(pipeline_output: dict) -> str:
    """
    TODO(P6): Fill out the template below using pipeline_output's fields.
    Keep it templated (f-strings / .format), not model-generated text.
    """
    q = pipeline_output.get("quality", {})
    pred = pipeline_output.get("prediction", {})
    unc = pipeline_output.get("uncertainty", {})
    lesions = pipeline_output.get("lesions", {})
    rec = pipeline_output.get("recommendation", "review")

    report = f"""
AI-ASSISTED SCREENING OUTPUT — OPHTHALMOLOGIST CONFIRMATION REQUIRED

IMAGE QUALITY
  Score: {q.get('score', 'N/A')}/100
  Status: {q.get('status', 'N/A')}

DR GRADE: {pred.get('dr_grade', 'N/A')}
REFERABLE DR: {'Yes' if pred.get('referable') else 'No'}

UNCERTAINTY: {unc.get('uncertainty_level', 'N/A')}

LESION EVIDENCE (heuristic, not clinically validated)
  Microaneurysm candidates: {lesions.get('microaneurysm_count', 'N/A')}
  Hemorrhage candidates: {lesions.get('hemorrhage_count', 'N/A')}
  Exudate candidates: {lesions.get('exudate_count', 'N/A')}

RECOMMENDATION: {rec}

This is a screening/referral-support prototype, not an autonomous
diagnostic device. This is a hackathon prototype trained on a small
subsample of data and has not undergone clinical validation.
"""
    return report.strip()


def _np_to_reportlab_image(array: np.ndarray, width_pt: float):
    """Converts a numpy RGB uint8 array into a reportlab-embeddable Image
    flowable, preserving aspect ratio at the given target width."""
    from reportlab.platypus import Image as RLImage

    pil_img = PILImage.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    aspect = pil_img.height / pil_img.width
    return RLImage(buf, width=width_pt, height=width_pt * aspect)


def generate_pdf_report(pipeline_output: dict, original_image: np.ndarray, patient_id: str = "N/A") -> bytes:
    """
    Builds a PDF version of the same rule-based report text, with the
    original/enhanced/Grad-CAM/lesion-overlay images embedded alongside it.
    Returns raw PDF bytes (caller — e.g. Streamlit — decides whether to
    write to disk, offer a download button, etc.).

    Still fully rule-based: every field is pulled directly from
    pipeline_output, nothing is generated or inferred here.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    story = []

    q = pipeline_output.get("quality", {})
    pred = pipeline_output.get("prediction", {})
    unc = pipeline_output.get("uncertainty", {})
    lesions = pipeline_output.get("lesions", {})
    rec = pipeline_output.get("recommendation", "review")

    story.append(Paragraph("AI-Assisted DR Screening Report", styles["Title"]))
    story.append(Paragraph(
        "Ophthalmologist confirmation required — not an autonomous diagnostic device.",
        styles["Italic"]
    ))
    story.append(Spacer(1, 12))

    table_data = [
        ["Patient/Image ID", str(patient_id)],
        ["Image Quality", f"{q.get('score', 'N/A')}/100 ({q.get('status', 'N/A')})"],
        ["DR Grade", str(pred.get("dr_grade", "N/A"))],
        ["Referable DR", "Yes" if pred.get("referable") else "No"],
        ["Uncertainty", str(unc.get("uncertainty_level", "N/A"))],
        ["Microaneurysm candidates", str(lesions.get("microaneurysm_count", "N/A"))],
        ["Hemorrhage candidates", str(lesions.get("hemorrhage_count", "N/A"))],
        ["Exudate candidates", str(lesions.get("exudate_count", "N/A"))],
        ["Recommendation", str(rec).upper()],
    ]
    table = Table(table_data, colWidths=[2.3 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3d9e5")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4a3267")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Visual Evidence", styles["Heading2"]))
    story.append(Spacer(1, 6))

    img_width = 2.1 * inch
    labels = ["Original", "Enhanced", "Grad-CAM"]
    images = [original_image, pipeline_output.get("enhanced_image", original_image),
              pipeline_output.get("gradcam", original_image)]
    row = [_np_to_reportlab_image(img, img_width) for img in images]
    img_table = Table([row, labels], colWidths=[img_width + 0.2 * inch] * 3)
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Lesion Evidence Overlay (heuristic, not clinically validated)", styles["Heading2"]))
    story.append(Spacer(1, 6))
    lesion_img = lesions.get("overlay_image", original_image)
    story.append(_np_to_reportlab_image(lesion_img, 4.5 * inch))
    story.append(Spacer(1, 16))

    story.append(Paragraph(
        "This is a screening/referral-support prototype, not an autonomous "
        "diagnostic device. Trained on a small hackathon subsample and has "
        "not undergone clinical validation.",
        styles["Italic"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
