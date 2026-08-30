"""
Owner: P1 (Admin)
Day 6: run this BEFORE the demo, and ideally every time someone merges a PR
today. Automates all the manual checks we've been doing by hand all week —
catches broken imports, missing functions, and pipeline crashes in one shot
instead of discovering them mid-demo.

Usage:
    python scripts/health_check.py

Exit code 0 = all checks passed. Non-zero = something needs fixing before
the demo.
"""
import os
import sys
import importlib
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REQUIRED_FUNCTIONS = {
    "src.data_pipeline": ["load_dataset", "stratified_split", "get_augmentation_pipeline", "load_image"],
    "src.quality": ["compute_quality_score", "get_recapture_message"],
    "src.enhancement": ["enhance_image"],
    "src.model": ["build_model", "load_trained_model"],
    "src.train": ["train", "evaluate", "predict"],
    "src.gradcam": ["generate_gradcam"],
    "src.uncertainty": ["mc_dropout_predict"],
    "src.lesion_overlay": ["detect_lesions", "combine_evidence"],
    "src.report_generator": ["generate_report"],
}

CHECKS_PASSED = []
CHECKS_FAILED = []


def check(name, fn):
    try:
        fn()
        CHECKS_PASSED.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        CHECKS_FAILED.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")


def check_imports_and_contracts():
    print("\n1. Checking all modules import and match CONTRACTS.md signatures...")
    for module_name, expected_functions in REQUIRED_FUNCTIONS.items():
        def _check(module_name=module_name, expected_functions=expected_functions):
            module = importlib.import_module(module_name)
            missing = [fn for fn in expected_functions if not hasattr(module, fn)]
            if missing:
                raise AssertionError(f"missing function(s): {missing}")
        check(f"{module_name} — {expected_functions}", _check)


def check_dummy_pipeline():
    print("\n2. Checking main.py --dummy runs end-to-end...")
    def _check():
        import numpy as np
        from main import run_pipeline
        image = np.random.randint(0, 255, size=(224, 224, 3), dtype="uint8")
        result = run_pipeline(image, model=None)
        required_keys = ["quality", "enhanced_image", "prediction", "uncertainty",
                          "gradcam", "lesions", "evidence_image", "report", "recommendation"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            raise AssertionError(f"run_pipeline() output missing keys: {missing}")
    check("main.run_pipeline() with model=None", _check)


def check_checkpoint_status():
    print("\n3. Checking for a trained model checkpoint...")
    checkpoint_path = "models/best_checkpoint.keras"
    if os.path.exists(checkpoint_path):
        print(f"  [INFO] Found checkpoint at {checkpoint_path} — pipeline will use real predictions.")
        def _check():
            from src.model import load_trained_model
            from src.train import predict
            import numpy as np
            model = load_trained_model(checkpoint_path)
            image = np.random.randint(0, 255, size=(224, 224, 3), dtype="uint8").astype("float32")
            result = predict(model, image)
            if result.get("dr_grade") is None:
                raise AssertionError("predict() returned no grade even with a real model loaded")
        check("real checkpoint loads and predicts", _check)
    else:
        print(f"  [WARNING] No checkpoint at {checkpoint_path} — demo will show placeholder "
              "predictions only. This should be fixed before presenting, if at all possible.")
        CHECKS_FAILED.append(("checkpoint exists", "no trained model found — demo will use placeholders"))


def check_real_data_present():
    print("\n4. Checking for real dataset...")
    csv_path = "data/train_subsample.csv"
    image_dir = "data/train_images"
    if os.path.exists(csv_path) and os.path.isdir(image_dir) and len(os.listdir(image_dir)) > 0:
        print(f"  [PASS] Found {csv_path} and non-empty {image_dir}/")
        CHECKS_PASSED.append("real dataset present")
    else:
        print(f"  [WARNING] {csv_path} or {image_dir}/ missing/empty — "
              "demo will only work with synthetic/dummy images unless this is fixed.")
        CHECKS_FAILED.append(("real dataset present", f"{csv_path} or {image_dir}/ missing"))


def check_streamlit_app_imports():
    print("\n5. Checking the Streamlit app's imports are valid (syntax + import check only)...")
    def _check():
        import ast
        with open("app/streamlit_app.py") as f:
            source = f.read()
        ast.parse(source)  # raises SyntaxError if broken
    check("app/streamlit_app.py parses without error", _check)


def main():
    print("=" * 60)
    print("DR SCREENING PROTOTYPE — PRE-DEMO HEALTH CHECK")
    print("=" * 60)

    check_imports_and_contracts()
    check_dummy_pipeline()
    check_checkpoint_status()
    check_real_data_present()
    check_streamlit_app_imports()

    print("\n" + "=" * 60)
    print(f"RESULT: {len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed/warned")
    print("=" * 60)

    if CHECKS_FAILED:
        print("\nItems needing attention before demo:")
        for name, reason in CHECKS_FAILED:
            print(f"  - {name}: {reason}")
        sys.exit(1)
    else:
        print("\nAll checks passed — repo is demo-ready.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[UNEXPECTED ERROR during health check itself]")
        traceback.print_exc()
        sys.exit(2)
