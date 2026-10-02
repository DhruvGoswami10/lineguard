# LineGuard — ICT304 (Murdoch) group project

Camera-based production-line defect inspection. An image of a part comes in; the
system scores it, flags it OK or REJECT, and shows a heatmap of where the defect is.

This repo is the **programming side only**. Teammates write the report. Do not
write report prose; produce code, results and run instructions they can use.

## Current milestone: assignment prototype, due Sat 3 Oct 2026, midnight

Build **one AI sub-system: defect detection**, with two techniques compared.

How it is marked (prototype = 20%):
- Code provided
- Initial results
- Steps to run the demo
- **If the code or demo doesn't run, 50% of the whole mark is lost.** Reproducibility beats features.

Out of scope until the final (7 Nov): folder watcher / ingest, SQLite QC log,
dashboard, alerts, webcam. Don't build these now.

## Data: MVTec AD, `bottle` category only

- Licence CC BY-NC-SA 4.0. Cite in README.
- Expected layout under `data/mvtec/bottle/`:
  - `train/good/`: defect-free only
  - `test/{good,broken_large,broken_small,contamination}/`
  - `ground_truth/<defect>/*_mask.png`: pixel masks
- Print image counts per folder on load and fail loudly if the layout is wrong.
- `data/` is git-ignored. README explains how to download.
- Download: `python -m src.download_data` (stdlib only). Source is the Hugging Face
  mirror `foersben/mvtec-ad`, pinned to commit `c75b39616f84db43677bcc8228caaafaf5096d7f`,
  every file SHA-256 checked. Official source: https://www.mvtec.com/company/research/datasets/mvtec-ad

### The split (most important rule)

The train set has no defects, so the CNN can't learn from it alone.

1. Split `test/` **50/50, stratified by folder** (good + each defect type), fixed seed.
2. **Split A (calibration):** the CNN trains on it. The autoencoder uses it **only** to pick its threshold.
3. **Split B (held-out):** both models are evaluated here and only here.
4. Hold out 10% of `train/good` as validation for early stopping.
5. Save the split to `splits/bottle_split.json` and commit it. Every script reads this file; nothing re-splits.

Both models must be scored on identical images or the comparison is invalid.

## Technique 1: convolutional autoencoder (expected final choice)

- Trained on `train/good` only (minus validation).
- 128×128 RGB, scaled to [0,1].
- Encoder: 4 stride-2 conv blocks. Decoder mirrors with ConvTranspose. Sigmoid output.
- MSE loss, Adam 1e-3, batch 16, up to 100 epochs, early stopping on validation loss.
- Error map = per-pixel squared error averaged over channels, then Gaussian-smoothed.
- Anomaly score = max of the smoothed map.
- Heatmap = the smoothed error map, upsampled and overlaid.
- Threshold = value that maximises F1 on split A.

## Technique 2: supervised CNN classifier

- torchvision ResNet18, ImageNet weights, final layer replaced with 2 classes (good / defect).
- 224×224, ImageNet normalisation.
- Train on `train/good` (minus validation) + all of split A.
- Class-weighted cross-entropy, since defects are scarce.
- Augmentation: flips, ±15° rotation, mild colour jitter.
- Adam 1e-4, ~20 epochs.
- Score = softmax probability of "defect". Threshold = 0.5. No split-A threshold for the CNN:
  split A is its training data, so any threshold looks perfect there (decision 2026-10-02).
- Heatmap = Grad-CAM on `layer4`, implemented by hand with hooks (no extra library; it must be explainable in an interview).
- When loading saved weights, build the model with `weights=None` so the demo needs no internet.

## Checkpoints

`weights/ae.pt` and `weights/cnn.pt` each store the model weights **and** the threshold the
model uses, so the demo needs nothing but `weights/` + `samples/`.

## Evaluation (`src/evaluate.py`, split B only)

- Image-level AUROC
- Precision, recall, F1 at the chosen threshold
- 95% bootstrap confidence intervals (resampling split B) for AUROC, precision, recall, F1
- Confusion matrix (and specificity)
- Recall per defect type
- Mean CPU inference ms/image
- AE only: pixel-level AUROC using the masks

Outputs to `results/`:
- `metrics.csv`
- `roc.png` (both models on one plot)
- `score_hist_ae.png`, `score_hist_cnn.png` (good vs defect)
- `heatmaps/`: 6 examples per model, mix of correct and incorrect
- `SUMMARY.md`: results table, hyperparameters used, one-line takeaway per model. This is the handoff to the report writers.

Expected: CNN looks stronger on split B because it saw these defect types. AE wins on
needing zero defect examples. Report numbers as they are; don't tune on split B.

## Demo (`src/demo.py`)

```
python -m src.demo --model ae  --input samples/
python -m src.demo --model cnn --input samples/some_image.png
```

- Per image, prints filename, score, threshold and OK/REJECT.
- Saves an image | heatmap overlay | verdict figure to `results/demo/`.
- Must run on CPU with only `weights/` + `samples/`. No dataset download, no training, no internet.

## Repo layout

```
CLAUDE.md            this spec
README.md            setup, data download, train/eval/demo commands, dataset citation
requirements.txt     exact versions that were actually tested
src/common.py        seeding and device selection
src/download_data.py fetch MVTec AD bottle (stdlib only, hash-checked)
src/data.py          loading, transforms, split creation
src/models.py        ConvAutoencoder, build_resnet18()
src/gradcam.py
src/inference.py     one scoring path shared by training, evaluation and demo
src/visualize.py     image | heatmap | verdict figures
src/train_ae.py
src/train_cnn.py
src/evaluate.py
src/demo.py
tests/               unittest suite: python -m unittest discover -s tests -v
splits/              committed split JSON
weights/             ae.pt, cnn.pt (committed; check size)
samples/             ~10 good + ~10 defect images from split B, with attribution note
results/
docs/ai_prompts.md   prompt log (see below)
docs/report_inputs.md facts and tables for the report writers (no prose)
```

## Conventions

- Python 3.12 (demo also tested on 3.13). Deps: torch, torchvision, numpy, scikit-learn, scipy, matplotlib, pillow. Ask before adding more.
- Tests use the standard-library `unittest` (no pytest).
- Relative paths only, set via argparse defaults. No absolute paths, no notebooks as the deliverable.
- Fix seeds (python, numpy, torch) in every script.
- Device: use CUDA if available, else CPU. Everything must still work on CPU.
  Exceptions: `evaluate.py` and `demo.py` default to CPU (the brief asks for CPU timings, and CPU
  scores are the ones a marker reproduces); the AE threshold is calibrated on CPU.
- Short functions, docstrings, and comments that explain *why*. Every team member may be interviewed on this code.
- Small commits with clear messages. Version control is a marked, mandatory tool.

## AI-use logging (required by the brief)

The brief requires every prompt used with AI tools to appear in the report appendix.
At the start of each task, append the user's request verbatim to `docs/ai_prompts.md`
as `- YYYY-MM-DD — <prompt>`. Don't paraphrase it.

## Before calling the milestone done

1. Fresh venv → `pip install -r requirements.txt`.
2. `python -m src.demo --model ae --input samples/` and the same with `--model cnn` both run clean on CPU.
3. `results/SUMMARY.md` and the plots exist and match `metrics.csv`.
4. README steps work exactly as written.

## Decision log

- 2026-10-02 — Python 3.12 instead of 3.11: current numpy (2.5.3) and scipy (1.18.1) need Python ≥ 3.12,
  and torch 2.14.1 has no wheels for Python 3.15.
- 2026-10-02 — CNN uses threshold 0.5 only (split A is its training data).
- 2026-10-02 — Thresholds stored inside checkpoints; bootstrap CIs added; unittest suite added;
  stdlib download script added.
- 2026-10-02 (after independent review) — AE threshold placed halfway between neighbouring split-A
  scores and calibrated on CPU (was exactly on a GPU-computed score; split-B results unchanged).
  Precision/recall/specificity CIs are exact binomial (bootstrap gave [1.000–1.000] with zero errors).
  Demo defaults to CPU. All 28 installed packages pinned.
