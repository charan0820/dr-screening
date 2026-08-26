# DR Screening — Hackathon Prototype

Quality-aware, explainable, uncertainty-calibrated Diabetic Retinopathy screening
prototype. Built on a subsampled APTOS 2019 dataset in a 6-day sprint.

## Team & Ownership

| Person | Module | File(s) |
|---|---|---|
| P1 (Admin) | Integration, repo, reviews | `main.py` |
| P2 | Data pipeline | `src/data_pipeline.py` |
| P3 | Quality + enhancement | `src/quality.py`, `src/enhancement.py` |
| P4 | Model + training | `src/model.py`, `src/train.py` |
| P5 | Explainability + uncertainty | `src/gradcam.py`, `src/uncertainty.py`, `src/lesion_overlay.py` |
| P6 | GUI + reporting | `app/streamlit_app.py`, `src/report_generator.py` |

**Rule: only edit files in your own row.** If you need something from another
module, call the agreed function signature (see `CONTRACTS.md`) — don't reach
into someone else's file directly. This is what keeps merge conflicts near zero.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Kaggle: **APTOS 2019 Blindness Detection** (use the resized/224px export, not
the raw 3000x3000 originals).

```bash
# via Kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle competitions download -c aptos2019-blindness-detection -p data/
```

Subsample to ~800-1200 images, stratified across the 5 classes, oversampling
grades 3/4 since they're rare. `src/data_pipeline.py` handles this.

## Day 1 Goal

Everyone can run `python main.py --dummy` and see a full (stubbed) pipeline
execute end-to-end: load → quality → enhance → classify → explain → report.
Nobody's function is *implemented* yet — just wired together and returning
placeholder values that match the agreed shapes in `CONTRACTS.md`.

## Branching

- Branch name: `day{N}-{initials}-{feature}` e.g. `day3-ps-enhancement`
- Pull + rebase on `main` every morning before starting new work
- PR merge order matters more than who finishes first — see day-by-day plan
  in the team doc. Roughly: data → quality → model → explainability → GUI → integration.
- P1 is the only merge gatekeeper for `main.py`; P6 is the only one who edits
  `app/streamlit_app.py` directly — everyone else exposes functions, they don't
  touch the integration files.

## Full 6-Day Plan

1. **Setup & contracts** — env, data download, stub pipeline runs end-to-end
2. **Preprocessing & skeletons** — augmentation, quality scoring, model skeleton, Grad-CAM scaffold, GUI shell
3. **Enhancement & training start** — CLAHE + recapture logic, kick off real training
4. **Classification + explainability** — finish training, real Grad-CAM + uncertainty, GUI wiring
5. **Integration + report + capacity sim** — full pipeline in `main.py` and GUI, ablation comparison, telemedicine capacity sim
6. **Testing & demo prep** — bug bash, freeze, rehearse pitch
