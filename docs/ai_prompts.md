# AI prompt log

The ICT304 brief requires every prompt used with AI tools to appear in the report appendix.

- **Tool:** Claude Code (command-line coding assistant) running Claude Opus 5.5, by Anthropic.
- **Who typed them:** Dhruv Goswami (programming lead).
- **Format:** one entry per request, copied verbatim (typos included), newest last.
- **Not in this file:** the earlier planning chat on claude.ai (Claude web), where the LineGuard
  spec was drafted. Dhruv to export those prompts (claude.ai → Settings → Privacy → Export data)
  and add them to the appendix.

Suggested APA 7 reference: Anthropic. (2026). *Claude* (Opus 5.5) [Large language model]. https://claude.ai

---

- 2026-10-02 — (multi-line prompt, verbatim below)

```text
on the claude web about the ICT304 factory vision ML assignment and I want to do this now. so let's start but I need to only do the coding part of the project since htis is the group assingment I will do the programming and the ML and all and the others will do the documentation. so basically I will do eveyrhting except documentation. so here are the files:

C:\Users\ASUS\Downloads\ICT304_Assignment_Project_TS_2026.pdf
"C:\Users\ASUS\Downloads\Assignment Feedback sheet.docx"

checkj and also check the claude web memroy and all that for the assignment and then talk on how will we do this one
```

- 2026-10-02 — (multi-line prompt, verbatim below; the pasted spec was drafted earlier on claude.ai)

````text
here look at this:

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
- Score = softmax probability of "defect". Threshold = 0.5, also report best-F1 threshold from split A.
- Heatmap = Grad-CAM on `layer4`, implemented by hand with hooks (no extra library; it must be explainable in an interview).
- When loading saved weights, build the model with `weights=None` so the demo needs no internet.

## Evaluation (`src/evaluate.py`, split B only)

- Image-level AUROC
- Precision, recall, F1 at the chosen threshold
- Confusion matrix
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
README.md            setup, data download, train/eval/demo commands, dataset citation
requirements.txt     exact versions that were actually tested
src/data.py          loading, transforms, split creation
src/models.py        ConvAutoencoder, build_resnet18()
src/train_ae.py
src/train_cnn.py
src/evaluate.py
src/demo.py
src/gradcam.py
splits/              committed split JSON
weights/             ae.pt, cnn.pt (committed; check size)
samples/             ~10 good + ~10 defect images from split B, with attribution note
results/
docs/ai_prompts.md   prompt log (see below)
```

## Conventions

- Python 3.11. Deps: torch, torchvision, numpy, scikit-learn, scipy, matplotlib, pillow. Ask before adding more.
- Relative paths only, set via argparse defaults. No absolute paths, no notebooks as the deliverable.
- Fix seeds (python, numpy, torch) in every script.
- Device: use CUDA if available, else CPU. Everything must still work on CPU.
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
````

- 2026-10-02 — ok now tell me about the assignment and what we have to do for the assignment 1. I want to finish it all and give it to the team for the documentation. so talk short and give me full plan

- 2026-10-02 — let's do everything today and finish it and I then give full package to the team. tell me is this going to be like collab notebook or how is it going to be??
