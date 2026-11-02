# LineGuard — ICT304 (Murdoch) group project

Camera-based production-line defect inspection. An image of a part comes in; the
system scores it, flags it OK or REJECT, and shows a heatmap of where the defect is.

This repo is the **programming side only**. Teammates write the report. Do not
write report prose; produce code, results and run instructions they can use.

## Current milestone: final system, due Sat 7 Nov 2026

Assignment 1 (3 Oct, tag `v0.1-assignment`) delivered the AI sub-system with two techniques.
The final delivers the whole line: every non-AI sub-system from the Assignment 1 design,
a third AI technique (FoundAD-style, the new primary detector), the full test plan, the
user guide, results pack and inputs for the report, demo video and presentation.

How the final is marked (feedback sheet): requirements 20%, system design 35% (demo video
required: -20% without), evaluation 20% (-15% if any subsystem is untested), user guide +
code 15%, content/format 10%. **If the code or demo doesn't run, 50% of the mark is lost.**
Reproducibility still beats features.

### Final-system scope (all built)

- Acquisition: video file / webcam / folder through one interface; station trigger (dark blob fully
  inside the ROI, part-centred crop), N frame votes (median). `src/acquisition.py`
- Demo camera feed: `samples/line_demo.mp4`, rendered from the 20 real split-B sample photos moving on
  a simulated backlit belt (`src/make_line_video.py`). A real recording drops in with no code change.
- Decision engine: PASS / REVIEW / REJECT (t_low, t_high), severity bands relative to t_high. `src/decision.py`
- QC log: SQLite (inspections, settings, alerts) + CSV export. `src/qclog.py`
- Reject signal: JSONL event log + beep. `src/reject.py`
- Alerts: p-chart reject-rate spike, camera stall; once per incident. `src/alerts.py`
- Dashboard: Streamlit, reads the QC log, writes operator settings and review labels. `src/dashboard.py`
- Line runner: `python -m src.run_line --source ...` wires everything, writes `results/line/`. `src/run_line.py`
- Technique 3: frozen DINOv2 ViT-S/14 (`src/dino.py`, own minimal implementation loading the official
  weights, `src/download_backbone.py`) + FoundAD predictor trained on 4 good images with CutPaste
  (`src/foundad.py`, `src/train_foundad.py`). Thresholds: t_high best-F1 on split A, t_low 95th
  percentile of unseen good images. Checkpoint stores only the predictor.
- Evaluation extended to three models; `results/SUMMARY.md` has three columns.

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

## Technique 1: convolutional autoencoder (Assignment 1's preferred technique)

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

## Technique 3: FoundAD-style few-shot detector (final primary detector)

- Frozen DINOv2 ViT-S/14, features from the 3rd-from-last block, 448 px input (32x32 patch grid).
- Predictor: Linear -> 6 transformer blocks (width 384, 12 heads, input embedding added back after each
  block, no positional embedding) -> LayerNorm -> Linear. ~11 M parameters. Only this is trained.
- Training: 4 seeded good images from `train`; each step augments them (flip/rot90 p=0.3, colour/grey/blur
  p=0.3), makes a CutPaste copy (blob or scar pasted on the Otsu foreground) for half the contexts, and
  minimises MSE(predictor(context features), clean features). AdamW 1e-3, wd 1e-4, batch 4, 1000 steps.
- Score = mean of the top-10 patch errors; heatmap = the 32x32 error grid (display scaled from the
  median normal patch error "map_floor" to t_high).
- Rationale and reference: Zhai et al. 2025 (arXiv:2510.01934), github.com/ymxlzgy/FoundAD (MIT).
  Re-implemented, not vendored: their environment pins Python 3.8 / torch 2.2 and 20 extra packages.
- DINOv3 is not used: its weights are gated behind Meta's licence, so a marker could not reproduce.

## Checkpoints

`weights/ae.pt`, `weights/cnn.pt` and `weights/foundad.pt` each store the model weights **and** the
threshold(s) the model uses, so the demo needs nothing but `weights/` + `samples/`. The FoundAD
checkpoint holds the predictor only; the DINOv2 file (git-ignored, 88 MB) is fetched once by
`python -m src.download_backbone` and hash-checked.

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

## Demo (`src/demo.py`) and the line (`src/run_line.py`)

```
python -m src.demo --model ae  --input samples/
python -m src.demo --model cnn --input samples/some_image.png
python -m src.run_line --source samples/line_demo.mp4 --reset     # the end-to-end demo
streamlit run src/dashboard.py                                    # in a second terminal
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

- Python 3.12 (demo also tested on 3.13). Deps: torch, torchvision, numpy, scikit-learn, scipy, matplotlib,
  pillow, opencv-python (video/camera, final), streamlit (dashboard, final). Ask before adding more.
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

1. Fresh venv → `pip install -r requirements.txt` → `python -m src.download_backbone`.
2. `python -m src.demo --model <ae|cnn|foundad> --input samples/` all run clean on CPU.
3. `python -m src.run_line --source samples/line_demo.mp4 --reset` decides all 20 parts; the dashboard opens on the log.
4. `results/SUMMARY.md` and the plots exist and match `metrics.csv` (three models).
5. `python -m unittest discover -s tests -v` passes (including the end-to-end test).
6. README steps work exactly as written, on Windows too (the final-system packages were pinned on Linux).

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
- 2026-11-03 — Technique 3 added as primary detector (FoundAD-style on frozen DINOv2 ViT-S/14, re-implemented
  in-repo; DINOv2 over DINOv3 because DINOv3 weights are gated). Camera stage = recorded conveyor video
  rendered from real split-B photos (no live camera needed for the demo; webcam supported). Station trigger
  tracks decided parts by x so parts closer than the station width are separated. REVIEW band: t_low = 95th
  percentile of unseen good-image scores. Severity relative to t_high (5% / 15%). Alerts: p-chart 3-sigma,
  baseline 5%, window 20; stall 10 s. Dashboard settings override the checkpoint thresholds per part.
