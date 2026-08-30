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

## Day 6 Goal

Bug bash, freeze, and demo rehearsal — no new features today. Run
`python scripts/health_check.py` and fix everything it flags before
touching the pitch deck.

## Known Limitations (state these honestly to judges — don't oversell)

- Trained on a small stratified subsample (~1,000 images), not the full
  APTOS set — absolute accuracy numbers will be modest; the QWK-based
  ablation comparison is the more meaningful result to lead with.
- Image-level train/val/test split, not patient-level (APTOS provides no
  patient ID field) — a known, disclosed limitation, not an oversight.
- Lesion overlay (`src/lesion_overlay.py`) is a classical heuristic for
  visual/demo purposes, not a validated microaneurysm/hemorrhage detector.
- The telemedicine capacity model (`scripts/telemedicine_simulation.py`)
  is a simplified deterministic calculation, not a full discrete-event
  queueing simulation with arrival-time variance.
- This is a screening/referral-support prototype, not an autonomous
  diagnostic device, and has not undergone clinical validation.

## Day 6 Checklist

- [ ] `python scripts/health_check.py` exits 0 (all checks pass)
- [ ] Real checkpoint exists at `models/best_checkpoint.keras`
- [ ] Real subsample exists at `data/train_subsample.csv` + `data/train_images/`
- [ ] `streamlit run app/streamlit_app.py` tested live, at least twice, by
      two different people (catches "works on my machine" issues)
- [ ] `scripts/batch_sanity_check.py` re-run with the final checkpoint —
      keep its output in `results/batch_check/` as an offline demo fallback
      in case live inference or venue wifi is unreliable
- [ ] Confusion matrix, ablation results, and telemedicine chart all
      regenerated with the FINAL checkpoint (not an early/stale one)
- [ ] Pitch rehearsed at least once end-to-end, timed

