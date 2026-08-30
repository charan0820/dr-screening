# Function Contracts

This is the agreement everyone codes against. If you change a signature here,
you MUST message the team before doing it — someone else's code depends on it.

All images are passed around as `numpy.ndarray`, shape `(H, W, 3)`, RGB, dtype
`uint8`, unless stated otherwise.

---

## P2 — `src/data_pipeline.py`

```python
def load_dataset(csv_path: str, image_dir: str) -> pd.DataFrame:
    """Returns a DataFrame with columns: ['image_id', 'filepath', 'label']"""

def stratified_split(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42) -> dict:
    """Returns {'train': df, 'val': df, 'test': df}"""

def get_augmentation_pipeline(training: bool = True):
    """Returns an albumentations.Compose (or equivalent) transform object."""

def load_image(filepath: str) -> np.ndarray:
    """Returns RGB uint8 image array."""
```

## P3 — `src/quality.py` and `src/enhancement.py`

```python
# quality.py
def compute_quality_score(image: np.ndarray) -> dict:
    """
    Returns:
    {
        'score': float,            # 0-100
        'status': str,             # 'good' | 'borderline' | 'ungradable'
        'failure_reason': str|None # 'blur' | 'low_illumination' | 'low_contrast' |
                                    # 'incomplete_field' | 'glare' | None
    }
    """

def get_recapture_message(failure_reason: str) -> str:
    """Returns a human-readable recapture instruction string."""
```

```python
# enhancement.py
def enhance_image(image: np.ndarray, failure_reason: str | None) -> np.ndarray:
    """Applies CLAHE / illumination norm / etc. based on failure_reason.
    Returns enhanced RGB uint8 image, same shape as input."""
```

## P4 — `src/model.py` and `src/train.py`

```python
# model.py
def build_model(num_classes: int = 5, backbone: str = "efficientnet_b0"):
    """Returns a compiled/uncompiled model object (framework-agnostic name)."""

def load_trained_model(checkpoint_path: str):
    """Returns a loaded model ready for inference."""
```

```python
# train.py
def train(model, train_data, val_data, epochs: int = 15) -> dict:
    """Returns {'history': dict, 'best_checkpoint_path': str}"""

def evaluate(model, test_data) -> dict:
    """
    Returns:
    {
        'accuracy': float, 'macro_f1': float, 'qwk': float,
        'confusion_matrix': np.ndarray, 'per_class': dict
    }
    """

def predict(model, image: np.ndarray) -> dict:
    """
    Returns:
    {
        'dr_grade': int,           # 0-4
        'referable': bool,
        'probabilities': list[float]  # length 5, softmax output
    }
    """
```

## P5 — `src/gradcam.py`, `src/uncertainty.py`, `src/lesion_overlay.py`

```python
# gradcam.py
def generate_gradcam(model, image: np.ndarray, class_idx: int) -> np.ndarray:
    """Returns heatmap overlay, same H,W as input image, RGB uint8."""
```

```python
# uncertainty.py
def mc_dropout_predict(model, image: np.ndarray, n_passes: int = 10) -> dict:
    """
    Returns:
    {
        'mean_probs': list[float],
        'entropy': float,
        'uncertainty_level': str   # 'low' | 'medium' | 'high'
    }
    """
```

```python
# lesion_overlay.py
def detect_lesions(image: np.ndarray) -> dict:
    """
    Heuristic (non-trained) detector for demo overlay purposes.
    Returns:
    {
        'microaneurysm_count': int,
        'hemorrhage_count': int,
        'exudate_count': int,
        'overlay_image': np.ndarray   # RGB uint8, lesions marked
    }
    """
```

## P6 — `src/report_generator.py` and `app/streamlit_app.py`

```python
# report_generator.py
def generate_report(pipeline_output: dict) -> str:
    """
    Takes the full pipeline_output dict (see main.py PIPELINE_OUTPUT_SHAPE)
    and returns a formatted, rule-based text report. No LLM calls — templated
    strings only, per clinical-honesty requirement.
    """
```

`app/streamlit_app.py` has no external contract — it's the GUI, owned solely
by P6, and it imports from every module above via `main.py`'s
`run_pipeline()` function.

---

## The end-to-end shape — `main.py`

```python
def run_pipeline(image: np.ndarray) -> dict:
    """
    Returns PIPELINE_OUTPUT_SHAPE:
    {
        'quality': {...},              # from quality.compute_quality_score
        'enhanced_image': np.ndarray,  # from enhancement.enhance_image
        'prediction': {...},           # from train.predict
        'uncertainty': {...},          # from uncertainty.mc_dropout_predict
        'gradcam': np.ndarray,         # from gradcam.generate_gradcam
        'lesions': {...},              # from lesion_overlay.detect_lesions
        'evidence_image': np.ndarray,  # from lesion_overlay.combine_evidence (Day 5)
        'report': str,                 # from report_generator.generate_report
        'recommendation': str          # 'routine' | 'review' | 'recapture'
    }
    """
```

Everyone's stub should return dummy values matching these shapes TODAY so
`main.py --dummy` runs end-to-end by end of Day 1.
