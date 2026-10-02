# LineGuard: AI defect inspection for production lines

ICT304 group project (Murdoch University), **assignment prototype**: one AI sub-system,
defect detection, built with two techniques and compared on the same held-out images.

A photo of a part comes in. LineGuard scores it, decides **OK** or **REJECT**, and shows a
heatmap of where the defect is. The prototype inspects glass bottles (MVTec AD `bottle`).

| Technique | Learns from | Heatmap |
|---|---|---|
| 1. Convolutional autoencoder (unsupervised) | good bottles only | rebuild error |
| 2. ResNet-18 classifier (supervised, ImageNet-pretrained) | good + labelled defect bottles | Grad-CAM |

## Results (held-out split B: 42 images, 10 good, 32 defect)

| | Autoencoder | ResNet-18 |
|---|---|---|
| Defect images needed for training | **0** | 31 |
| Labelled images used to set the threshold | 41 (split A) | none (fixed 0.5) |
| Image AUROC [95% CI] | 0.928 [0.837–0.990] | 0.988 [0.956–1.000] |
| Defects caught (recall) | 28 / 32 | 31 / 32 |
| Good bottles wrongly rejected | 2 / 10 | 0 / 10 |
| CPU time per image | ~7 ms | ~21 ms |

Full numbers, hyperparameters and plots: [`results/SUMMARY.md`](results/SUMMARY.md).

---

## 1. Run the demo (CPU, no dataset needed, about 5 minutes)

The trained weights (`weights/`) and 20 sample images (`samples/`) are included,
so the demo needs no dataset, no training and no internet once installed.

**Step 1: install Python 3.12 or 3.13 (64-bit)** from https://www.python.org/downloads/.
Those are the tested versions. Do not use Python 3.15: PyTorch has no build for it yet.
Computer: Windows 10/11 or Linux (64-bit), or a Mac with Apple Silicon (M1 or newer) on macOS 14+;
PyTorch 2.14 has no build for Intel Macs.

**Step 2: open a terminal in this folder** (the one containing `README.md`). On Windows, keep the
folder path short (e.g. `C:\lineguard`): some PyTorch files have long names.

**Step 3: create a virtual environment and install the libraries.**

Windows (PowerShell or Command Prompt), with Python 3.12 (use `py -3.13` for 3.13):

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux:

```
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

(Linux only: to skip the large GPU build of PyTorch, run
`.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu`
before the line above.)

**Step 4: run the demo** with either model. Windows:

```
.venv\Scripts\python -m src.demo --model ae --input samples/
.venv\Scripts\python -m src.demo --model cnn --input samples/
```

macOS / Linux: the same with `.venv/bin/python` instead of `.venv\Scripts\python`.
One image works too: `--input samples/broken_small_010.png`. Any `.png`/`.jpg` image can be used.
The demo runs on the CPU by default, which reproduces the reported scores exactly
(`--device auto` uses a GPU if one is available).

**What you get**

- One line per image in the terminal: file name, score, threshold, verdict. Example:

  ```
  LineGuard demo | Convolutional autoencoder | REJECT if score >= 0.0045313 | cpu
  image                              score   threshold  verdict
  broken_small_010.png            0.032895   0.0045313  REJECT
  good_003.png                   0.0027083   0.0045313  OK
  ```
- One figure per image in `results/demo/`: input | heatmap | verdict.
  Autoencoder heatmap: red = rebuild error above the reject threshold.
  ResNet-18 heatmap: Grad-CAM, where the network saw evidence for "defect".
- Sample file names show the true answer (`good_...`, `broken_large_...`, ...).

## 2. Reproduce everything (download, train, evaluate)

Run these from the project folder with the same Python as above
(`.venv\Scripts\python` on Windows, `.venv/bin/python` on macOS/Linux; written as `python` below).

| Step | Command | What it does | Time |
|---|---|---|---|
| Download data | `python -m src.download_data` | MVTec AD `bottle`, 357 files, 157 MB into `data/mvtec/bottle/`; every image SHA-256 checked (the 2 text files by size). Safe to re-run. | ~1 min |
| Check data | `python -m src.data --check` | Prints image counts per folder; fails loudly if anything is missing | seconds |
| Train technique 1 | `python -m src.train_ae` | Autoencoder → `weights/ae.pt` (weights + threshold) | GPU ~20 s, CPU ~4 min |
| Train technique 2 | `python -m src.train_cnn` | ResNet-18 → `weights/cnn.pt` (downloads ImageNet weights once) | GPU ~2.5 min, CPU ~7 min |
| Evaluate | `python -m src.evaluate` | Scores split B on CPU → `results/` (metrics, plots, heatmaps, `SUMMARY.md`) | ~1 min |
| Tests | `python -m unittest discover -s tests -v` | 51 unit tests | ~1 min |

Notes:
- Training **overwrites** `weights/`. To keep the shipped weights, add `--out my_weights/ae.pt`
  (and `--results my_results`), then evaluate with `python -m src.evaluate --weights my_weights --out my_results`.
- Times were measured on the development laptop (RTX 4080 Laptop GPU; Ryzen 9 7940HS CPU, where the
  CPU figures are extrapolated from timed 1–3 epoch runs). Slower machines take longer.
- The split is fixed in `splits/bottle_split.json`. `python -m src.data --make-split` rebuilds it
  from the seed and refuses to overwrite a different file, so the committed split is verifiable.
- Every script fixes its random seeds (42). Evaluation uses the committed weights, so `python -m src.evaluate`
  reproduces the reported numbers exactly (timings vary by machine). Retraining both models from scratch on the
  development GPU reproduced every reported metric exactly; on other hardware retrained numbers may differ slightly.

### Optional: train on an NVIDIA GPU

```
.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

The scripts use the GPU automatically when one is available (`--device cpu` forces CPU).

## 3. How the data is used

MVTec AD gives defect-free images for training and a test set with defects. The CNN cannot
learn "defect" from defect-free images alone, so the test set is split in half:

| Part | Images | Used for |
|---|---|---|
| train | 188 good | autoencoder training, CNN training |
| val | 21 good | autoencoder early stopping (CNN: monitoring only) |
| split A | 41 (10 good, 31 defect) | CNN training; autoencoder threshold only |
| split B | 42 (10 good, 32 defect) | **evaluation of both models, nothing else** |

Split A/B is 50/50 within each folder (good, broken_large, broken_small, contamination), seed 42.
Both models are scored on exactly the same split-B images.

## 4. Project layout

```
README.md              this guide
CLAUDE.md              project spec and decision log
requirements.txt       exact library versions tested
src/download_data.py   fetch MVTec AD bottle (standard library only, SHA-256 checked)
src/data.py            layout check, fixed split, image transforms, dataset
src/models.py          ConvAutoencoder, build_resnet18()
src/gradcam.py         Grad-CAM written by hand with PyTorch hooks
src/inference.py       one scoring path used by calibration, evaluation and demo
src/visualize.py       training curves and image | heatmap | verdict figures
src/train_ae.py        train technique 1, pick its threshold on split A
src/train_cnn.py       train technique 2
src/evaluate.py        evaluate both on split B, write results/
src/demo.py            command-line demo
tests/                 unittest suite (standard library, no extra packages)
splits/                the committed split
weights/               ae.pt (2.4 MB), cnn.pt (45 MB): weights + threshold
samples/               20 split-B images + ATTRIBUTION.md
results/               metrics, plots, heatmaps, demo figures, logs, SUMMARY.md
docs/                  AI prompt log, facts for the report
```

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `No matching distribution found for torch==2.14.1` | Use 64-bit Python 3.12 or 3.13. On a Mac, PyTorch 2.14 needs Apple Silicon and macOS 14+ (no Intel Mac build). |
| pip fails with a long-path / "No such file or directory" error (Windows) | Move the project to a short path such as `C:\lineguard`, delete `.venv`, and repeat step 3. |
| `No module named src` | Run commands from the project folder (where `README.md` is). |
| `py` is not recognised (Windows) | Use the full path to `python.exe` from your Python 3.12 install instead of `py -3.12`. |
| `Weights not found` | The `weights/` folder is missing: re-extract the full zip, or retrain (section 2). |
| `Dataset folder not found` | Only training/evaluation need the dataset: run `python -m src.download_data`. |
| Download stops part-way | Re-run `python -m src.download_data`; finished files are kept. |

## 6. Dataset, licences and references

**Dataset.** MVTec AD, category `bottle`. Copyright 2019 MVTec Software GmbH, licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use).
Official source: https://www.mvtec.com/company/research/datasets/mvtec-ad. The download script uses the
Hugging Face mirror `foersben/mvtec-ad` pinned to commit `c75b39616f84db43677bcc8228caaafaf5096d7f`.
The dataset itself is not in this repository; the 20 sample images are (see `samples/ATTRIBUTION.md`).

- Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD: A comprehensive real-world dataset
  for unsupervised anomaly detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition (CVPR)* (pp. 9592–9600).
- Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). The MVTec anomaly detection dataset:
  A comprehensive real-world dataset for unsupervised anomaly detection. *International Journal of Computer Vision,
  129*(4), 1038–1059. https://doi.org/10.1007/s11263-020-01400-4

**Methods and libraries.**

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of
  the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778). (ResNet-18)
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image
  database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 248–255). (pretrained weights)
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations
  from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on
  Computer Vision (ICCV)* (pp. 618–626).
- PyTorch 2.14.1 and torchvision 0.29.1 (models, training); NumPy, SciPy (Gaussian smoothing),
  scikit-learn (AUROC, precision-recall), Matplotlib (figures), Pillow (images). Versions in `requirements.txt`.

**Own work vs. libraries.** The data pipeline, split, both model definitions, training loops, threshold
calibration, Grad-CAM, evaluation and demo are written for this project. Library code used as-is:
PyTorch layers/optimisers, torchvision's ResNet-18 definition and ImageNet weights, scikit-learn metrics,
SciPy's Gaussian filter.

**AI assistance.** The code was written with Claude Code (Anthropic) under the programming lead's direction.
The prompts from the coding sessions are logged verbatim in [`docs/ai_prompts.md`](docs/ai_prompts.md), as the
unit brief requires; the earlier planning prompts (claude.ai) are added to the report appendix by the team.
