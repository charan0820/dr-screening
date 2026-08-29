"""
Owner: P1 (Admin)
Integration entrypoint. This is the ONLY file that imports from every src/
module and wires them together. Nobody else edits this file directly —
if your module's signature changes, tell P1 (or open a PR that only touches
your own file plus this one, and flag it for review).

Run with dummy stub data:
    python main.py --dummy
"""
import argparse
import os
import numpy as np

from src.data_pipeline import load_image
from src.quality import compute_quality_score, get_recapture_message
from src.enhancement import enhance_image
from src.model import load_trained_model  # noqa: F401 (used in main())
from src.train import predict
from src.gradcam import generate_gradcam
from src.uncertainty import mc_dropout_predict
from src.lesion_overlay import detect_lesions, combine_evidence
from src.report_generator import generate_report


def run_pipeline(image: np.ndarray, model=None) -> dict:
    """
    Full pipeline: quality -> enhance -> classify -> uncertainty -> Grad-CAM
    -> lesion overlay -> report -> recommendation.
    Returns the PIPELINE_OUTPUT_SHAPE dict documented in CONTRACTS.md.
    """
    quality = compute_quality_score(image)

    if quality["status"] == "ungradable":
        recommendation = "recapture"
        enhanced = image
        pred = {"dr_grade": None, "referable": None, "probabilities": None}
        unc = {"mean_probs": None, "entropy": None, "uncertainty_level": None}
        gradcam_img = image
        lesions = {"microaneurysm_count": None, "hemorrhage_count": None,
                   "exudate_count": None, "overlay_image": image}
    else:
        enhanced = enhance_image(image, quality.get("failure_reason"))
        pred = predict(model, enhanced)
        unc = mc_dropout_predict(model, enhanced)
        gradcam_img = generate_gradcam(model, enhanced, pred["dr_grade"])
        lesions = detect_lesions(enhanced)

        if unc["uncertainty_level"] == "high":
            recommendation = "review"
        elif pred["referable"]:
            recommendation = "review"
        else:
            recommendation = "routine"

    output = {
        "quality": quality,
        "enhanced_image": enhanced,
        "prediction": pred,
        "uncertainty": unc,
        "gradcam": gradcam_img,
        "lesions": lesions,
        "recommendation": recommendation,
    }
    # Day 5: combined side-by-side evidence image (original + Grad-CAM + lesion
    # overlay), per spec Section 23. Skipped for the ungradable path since
    # there's no meaningful Grad-CAM/lesion evidence to show for it.
    if quality["status"] != "ungradable":
        output["evidence_image"] = combine_evidence(enhanced, gradcam_img, lesions["overlay_image"])
    else:
        output["evidence_image"] = image

    output["report"] = generate_report(output)

    if quality["status"] == "ungradable":
        output["report"] += "\n\n" + get_recapture_message(quality.get("failure_reason"))

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Run with a fake random image")
    parser.add_argument("--image", type=str, help="Path to a real fundus image")
    parser.add_argument("--checkpoint", type=str, default="models/best_checkpoint.keras",
                         help="Path to a trained model checkpoint (Day 3+)")
    args = parser.parse_args()

    model = None
    if os.path.exists(args.checkpoint):
        print(f"Loading trained checkpoint: {args.checkpoint}")
        model = load_trained_model(args.checkpoint)
    else:
        print(f"No checkpoint found at {args.checkpoint} — using placeholder predictions.")

    if args.image:
        image = load_image(args.image)
    else:
        # --dummy or default: fake image so the pipeline can be smoke-tested today
        image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)

    result = run_pipeline(image, model=model)
    print(result["report"])


if __name__ == "__main__":
    main()
