# LineGuard code walkthrough (interview prep)

The brief says any team member may be interviewed on the code, and not understanding it can
mean zero. This page is the 15-minute version. Read it next to the code.

## The idea in five lines

1. A camera photo of a part goes in.
2. The AI model turns it into one number, the **score** (how defective it looks).
3. If score ≥ **threshold** → REJECT, otherwise OK.
4. A **heatmap** shows where on the part the model saw the problem.
5. We built it two ways and compared them on the same 42 images neither model ever saw.

## The two techniques

**Technique 1, convolutional autoencoder (unsupervised).** A network that squeezes a 128×128 image
down to a small 8×8×64 code and rebuilds the image from it. It is trained only on good bottles, so
it learns to rebuild good bottles well. Show it a cracked bottle and it rebuilds a *good-looking*
bottle, so the crack area comes out different from the input. That difference (the **rebuild error**)
is the heatmap, and its highest point is the score. It needs **no defect photos to learn**, and
factories have very few of those. A small labelled set (split A, 41 images) is used only to
pick its threshold.

**Technique 2, ResNet-18 classifier (supervised).** A standard image-classification network
already trained on ImageNet (1.2 million everyday photos), so it already knows edges, textures and
shapes (**transfer learning**). We replace its last layer with two outputs (good, defect) and
fine-tune it on our images, including 31 labelled defect photos. The score is its probability of
"defect". The heatmap comes from **Grad-CAM**.

## File by file

| File | What it does | Lines worth knowing |
|---|---|---|
| `src/download_data.py` | Downloads the 357 dataset files, checks every image against its SHA-256 hash (the 2 text files by size), retries dropped downloads | `fetch()` |
| `src/data.py` | Checks the folder layout, builds the fixed split, loads images, defines the resize/normalise steps | `make_split()`, `ae_transform()`, `cnn_transform()` |
| `src/models.py` | The autoencoder layers; ResNet-18 with a 2-class head | `ConvAutoencoder`, `build_resnet18()` |
| `src/gradcam.py` | Grad-CAM written by hand with PyTorch hooks | `GradCAM.__call__` |
| `src/inference.py` | The single scoring function used everywhere; saving/loading checkpoints | `ae_inspect()`, `cnn_inspect()`, `inspect()`, `best_f1_threshold()` |
| `src/train_ae.py` | Trains the autoencoder with early stopping, then picks its threshold on split A | `train()`, `calibrate_threshold()` |
| `src/train_cnn.py` | Fine-tunes ResNet-18 with class-weighted loss and augmentation | `class_weights()`, `run_epoch()` |
| `src/evaluate.py` | Scores split B on CPU, computes all metrics + confidence intervals, draws plots, writes SUMMARY.md | `threshold_metrics()`, `bootstrap_ci()` |
| `src/demo.py` | The command-line demo | `main()` |
| `src/visualize.py` | Training curves and the image / heatmap / verdict figure | `save_verdict_figure()` |
| `src/common.py` | Fixed random seeds; picks GPU or CPU | `set_seed()` |
| `tests/` | 51 automatic tests (standard-library `unittest`) | run `python -m unittest discover -s tests -v` |

## Key ideas in plain words

- **Split A / split B.** MVTec gives defect-free training images plus a mixed test set. The CNN must see
  some defects to learn them, so the test set is cut in half per folder. Split A: CNN training, and the
  autoencoder's threshold only. Split B: used *only* for the final evaluation of both models. Same images
  for both, so the comparison is fair. The split is saved in a file and never re-made.
- **Validation set.** 21 good training images held back. The autoencoder would stop training when the
  error on them stopped improving for 10 epochs (**early stopping**), and keeps the best epoch. Here it
  never triggered: the validation error kept falling until epoch 100, the maximum, so epoch 100 was kept.
- **MSE loss.** Mean squared difference between the input and the rebuilt image, so the autoencoder learns to rebuild.
- **Gaussian smoothing (σ = 4).** Blurs the error map so a single noisy pixel can't trigger a reject,
  while a real defect (a blob of error) stays a clear peak.
- **Score = maximum of the map.** One bad spot anywhere is enough to reject the part.
- **Threshold by best F1 on split A.** We try every possible threshold on split A and keep the one with
  the best balance of precision and recall (F1). The threshold is placed halfway between the two
  neighbouring scores, not on a score, so tiny GPU-vs-CPU rounding differences can't flip an image; it is
  computed on the CPU, like the demo. The CNN uses 0.5 instead, because split A is its own training data
  and any threshold would look perfect there.
- **Class-weighted cross-entropy.** 198 good vs 31 defect images: each defect counts about 6× more in the loss
  so the CNN can't win by calling everything good.
- **Augmentation.** Random flips, small rotations and colour changes during CNN training: more variety
  from the same 31 defect photos.
- **Grad-CAM.** ResNet-18's last block (`layer4`) outputs 512 feature maps on a 7×7 grid. A forward
  hook saves those maps, a gradient hook saves how much each one pushes the "defect" score up.
  Average gradient per map = its weight; weighted sum of the maps (negatives removed) = the heatmap.
  It is scaled to its own maximum, so it is relative: even a confidently OK image shows a red spot.
- **AUROC.** The chance that a random defect image scores higher than a random good image.
  1.0 = perfect ranking, 0.5 = guessing. It does not depend on the threshold.
- **Precision** = of the rejected parts, how many were really defective. **Recall** = of the defective parts,
  how many we caught. **Specificity** = of the good parts, how many we passed. **F1** = balance of precision and recall.
- **95% confidence intervals.** With only 42 test images every number is uncertain. For rates
  (precision, recall, specificity) we use the exact binomial (Clopper–Pearson) interval: e.g. 10 of 10 good
  bottles passed still only proves the true rate is above 0.69. For AUROC and F1 (no exact formula) we
  **bootstrap**: re-sample the 42 images with replacement 1,000 times, recompute, keep the middle 95%.
- **Pixel AUROC.** Same as AUROC but per pixel, against the hand-drawn defect masks: how well the
  autoencoder's heatmap points at the actual defect.
- **Seeds.** Every script fixes the random seed (42), so re-running gives the same split and training order.

## Likely questions

1. **Why two techniques?** The brief asks for at least two compared. One needs no defect data (AE), one does (CNN).
2. **Which is better?** On split B the CNN scores higher (AUROC 0.988 vs 0.928), but it needed 31 labelled
   defect photos to train and only knows those defect types. The AE trained on zero defect photos (it used
   a small labelled set only to set its threshold) and can flag any unusual part.
3. **Why is the AE the expected final choice?** Real factories rarely have many defect photos, and new
   defect types appear. The AE learns from good parts, which every line produces in volume.
4. **What does the AE get wrong?** Faint contamination (4 of 11 passed as OK) and 2 false alarms on good bottles.
5. **Why not evaluate on split A?** The CNN trained on it and the AE threshold was tuned on it. Results there would be inflated.
6. **Why 128×128 for the AE and 224×224 for the CNN?** 128 keeps the AE small and fast; 224 is the size ResNet-18's ImageNet weights were trained at.
7. **Why a sigmoid output?** Input pixels are scaled to [0, 1]; the sigmoid keeps the rebuilt pixels in the same range.
8. **Why ImageNet normalisation for the CNN?** The pretrained weights expect inputs normalised that way.
9. **Why does the demo need no internet?** The CNN is built with `weights=None` and then our own trained weights are loaded from `weights/cnn.pt`.
10. **How do you know the demo matches the reported results?** One function (`inspect`) does all scoring,
    the demo and the evaluation both run on the CPU, and a test checks the demo's score for every sample
    equals the score in `results/scores_split_b.csv`.
11. **Why CPU for evaluation?** The brief asks for CPU timings, and a marker without a GPU gets the same numbers.
12. **How fast is it?** About 7 ms (AE) and 21 ms (CNN) per image on a laptop CPU.
13. **What is our own work vs libraries?** Ours: data pipeline, split, model definitions, training loops,
    threshold calibration, Grad-CAM, evaluation, demo, tests. Libraries: PyTorch layers/optimisers, torchvision's
    ResNet-18 + ImageNet weights, scikit-learn metrics, SciPy's Gaussian filter.
14. **Where is the data from, and is it legal to use?** MVTec AD, CC BY-NC-SA 4.0: non-commercial use with
    attribution is allowed. Cited in the README and `samples/ATTRIBUTION.md`.
15. **How would you improve it?** More good images, augmentation for the AE, SSIM loss instead of MSE, or a
    stronger unsupervised method (e.g. PatchCore, which uses pretrained features); re-calibrate the threshold
    on real line data; test on more products.

## The final system

**One inspection, end to end** (`src/run_line.py`). A frame comes from the source
(`VideoSource` wraps OpenCV's `VideoCapture`, so a file path and a camera index are the same
code; `FolderSource` replays image files). `find_parts()` thresholds the station ROI at a
grey level (the belt is backlit, the part is dark), labels the connected dark blobs and
keeps those big enough to be a part, noting whether each touches the left edge (entering)
or the right edge (leaving). `PresenceTrigger.update()` picks the right-most undecided
blob, scores it on each frame it is fully inside, and after N votes (or when it starts to
leave) says "decide". Decided parts are followed by their x position until they leave, so
when the next part enters before the previous has left (parts closer than the station is
wide) it is still treated as new. `crop_part()` cuts a square around the blob with the
dataset's margin, so the model sees the same framing as in training.

`Line.finish_part()` takes the median of the frame scores (`decision.vote`), reads the
current rule (the dashboard's settings if the operator saved any, else the checkpoint's
thresholds), decides, saves the crop and the heatmap figure, writes the QC log row, fires
the reject signal on REJECT, and checks the alert rules. The frame drawn into the annotated
video shows the station box (grey empty, yellow collecting, green/orange/red decided), the
part box, the last verdict and score, the counters and the heatmap inset.

**Why two thresholds.** `t_low` is the prototype's calibrated threshold (best F1 on split A),
so the evaluation numbers still describe the line: a part at or above it does not pass. The
band above it, up to `t_high`, sends borderline parts to a person: on a real line scrapping a
good part or shipping a bad one both cost more than a look. `t_high` is placed above every
good image the model was calibrated on (halfway between the highest of them, 3.22, and the
lowest split-A defect above it, 5.34): with it, 3 of the 31 calibration good images would be
reviewed and none rejected. The first rule tried (REVIEW from the 95th percentile of good
scores, capped at the best-F1 cut) gave an empty band, because the best-F1 cut lies inside
the tail of the good scores. Severity is the excess over `t_high` as a percentage, so the
same bands work for the autoencoder's 0.004 scale and FoundAD's 4.3 scale.

**Why the QC log is SQLite.** One file, no server, in the standard library, safe for two
processes (the runner writes, the dashboard reads and writes settings). It also holds the
operator's confirmations of REVIEW parts, which is how labelled data accumulates for a
future retrain.

**Alerts** are the two rules from the design. The p-chart limit is
`p0 + 3·sqrt(p0(1-p0)/n)`; with the defaults (5 %, 20 parts) it is 19.6 %, so a window
with 4 or more rejects in 20 raises the alert. The monitor raises each incident once and
again only after the condition has cleared. A stall is `stall_seconds` without a part.

**The demo feed** (`src/make_line_video.py`) exists because the proof of concept must be
reproducible on any machine: the 21 held-out photos move across a rendered belt under a
fixed camera. Only the motion is simulated; the bottles are the dataset's images. The
manifest it writes is used by the end-to-end test to check one decision per part and that
good parts pass while defective ones do not.

**What the first end-to-end run taught us.** The first FoundAD model rejected all 21 parts
of the demo feed, although it was perfect on the PNGs of the same bottles. Experiments
isolated the cause: the video codec. A JPEG round trip at quality 75 alone lifted a good
bottle's score from 1.7 to 4.1, a lossless clip reproduced the PNG score exactly, and resize
and rotation changed nothing. The hot patches sat in the white belt around the bottle, where
the codec's faint blocks are "texture" the model had never seen. Two changes, both kept and
both the kind a real camera needs too: the score now only counts patches on the part
(`patch_foreground`: the belt is not inspected), and training pairs go through a JPEG round
trip half the time, so compressed-but-normal is normal. The lesson for the report: the
calibration data must come through the same acquisition path as production.

**Testing.** Every subsystem has hand-worked unit tests (`tests/test_decision.py`,
`test_qclog.py`, `test_alerts.py`, `test_acquisition.py`, `test_foundad.py`,
`test_dashboard.py` with Streamlit's headless harness), and `tests/test_line.py` renders
a four-part video and runs the real runner on it (T23).
