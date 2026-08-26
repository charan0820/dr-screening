"""
P6 - Rule-based report generator.

This module generates a screening report from the output
of the complete pipeline.

No LLM calls are used.
"""

def generate_report(pipeline_output: dict) -> str:
    """
    Generate a rule-based screening report.

    Args:
        pipeline_output: Dictionary returned by main.run_pipeline()

    Returns:
        Formatted report as a string.
    """

    quality = pipeline_output.get("quality", {})
    prediction = pipeline_output.get("prediction", {})
    uncertainty = pipeline_output.get("uncertainty", {})
    lesions = pipeline_output.get("lesions", {})
    recommendation = pipeline_output.get(
        "recommendation",
        "review"
    )

    report = f"""
AI-ASSISTED SCREENING OUTPUT
OPHTHALMOLOGIST CONFIRMATION REQUIRED

IMAGE QUALITY
  Score: {quality.get("score", "N/A")}/100
  Status: {quality.get("status", "N/A")}

DIABETIC RETINOPATHY
  DR Grade: {prediction.get("dr_grade", "N/A")}
  Referable DR: {
        "Yes" if prediction.get("referable") else "No"
    }

UNCERTAINTY
  Level: {uncertainty.get("uncertainty_level", "N/A")}

LESION EVIDENCE
  Microaneurysm candidates: {
        lesions.get("microaneurysm_count", "N/A")
    }
  Hemorrhage candidates: {
        lesions.get("hemorrhage_count", "N/A")
    }
  Exudate candidates: {
        lesions.get("exudate_count", "N/A")
    }

RECOMMENDATION
  {recommendation.upper()}

This is a screening/referral-support prototype, not an
autonomous diagnostic device. Results require confirmation
by a qualified ophthalmologist.
"""

    return report.strip()