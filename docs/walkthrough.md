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
is the heatmap, and its highest point is the score. It needs **no defect photos at all**, and
factories have very few of those.

**Technique 2, ResNet-18 classifier (supervised).** A standard image-classification network
already trained on ImageNet (1.2 million everyday photos), so it already knows edges, textures and
shapes (**transfer learning**). We replace its last layer with two outputs (good, defect) and
fine-tune it on our images, including 31 labelled defect photos. The score is its probability of
"defect". The heatmap comes from **Grad-CAM**.

## File by file

| File | What it does | Lines worth knowing |
|---|---|---|
| `src/download_data.py` | Downloads the 357 dataset files, checks each against its SHA-256 hash, retries dropped downloads | `fetch()` |
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
| `tests/` | 46 automatic tests (standard-library `unittest`) | run `python -m unittest discover -s tests -v` |

## Key ideas in plain words

- **Split A / split B.** MVTec gives defect-free training images plus a mixed test set. The CNN must see
  some defects to learn them, so the test set is cut in half per folder. Split A: CNN training, and the
  autoencoder's threshold only. Split B: used *only* for the final evaluation of both models. Same images
  for both, so the comparison is fair. The split is saved in a file and never re-made.
- **Validation set.** 21 good training images held back. The autoencoder stops training when the error
  on them stops improving (**early stopping**, patience 10 epochs), so it doesn't just memorise.
- **MSE loss.** Mean squared difference between the input and the rebuilt image, so the autoencoder learns to rebuild.
- **Gaussian smoothing (σ = 4).** Blurs the error map so a single noisy pixel can't trigger a reject,
  while a real defect (a blob of error) stays a clear peak.
- **Score = maximum of the map.** One bad spot anywhere is enough to reject the part.
- **Threshold by best F1 on split A.** We try every possible threshold on split A and keep the one with
  the best balance of precision and recall (F1). The CNN uses 0.5 instead, because split A is its own
  training data and any threshold would look perfect there.
- **Class-weighted cross-entropy.** 198 good vs 31 defect images: each defect counts about 6× more in the loss
  so the CNN can't win by calling everything good.
- **Augmentation.** Random flips, small rotations and colour changes during CNN training: more variety
  from the same 31 defect photos.
- **Grad-CAM.** ResNet-18's last block (`layer4`) outputs 512 feature maps on a 7×7 grid. A forward
  hook saves those maps, a gradient hook saves how much each one pushes the "defect" score up.
  Average gradient per map = its weight; weighted sum of the maps (negatives removed) = the heatmap.
- **AUROC.** The chance that a random defect image scores higher than a random good image.
  1.0 = perfect ranking, 0.5 = guessing. It does not depend on the threshold.
- **Precision** = of the rejected parts, how many were really defective. **Recall** = of the defective parts,
  how many we caught. **Specificity** = of the good parts, how many we passed. **F1** = balance of precision and recall.
- **95% confidence interval (bootstrap).** Re-sample the 42 test images with replacement 1,000 times,
  recompute each metric, keep the middle 95%. Shows how much the result could move with a small test set.
- **Pixel AUROC.** Same as AUROC but per pixel, against the hand-drawn defect masks: how well the
  autoencoder's heatmap points at the actual defect.
- **Seeds.** Every script fixes the random seed (42), so re-running gives the same split and training order.

## Likely questions

1. **Why two techniques?** The brief asks for at least two compared. One needs no defect data (AE), one does (CNN).
2. **Which is better?** On split B the CNN scores higher (AUROC 0.988 vs 0.928), but it needed 31 labelled
   defect photos and only knows those defect types. The AE needed zero and can flag any unusual part.
3. **Why is the AE the expected final choice?** Real factories rarely have many defect photos, and new
   defect types appear. The AE only needs good parts, which every line produces in volume.
4. **What does the AE get wrong?** Faint contamination (4 of 11 passed as OK) and 2 false alarms on good bottles.
5. **Why not evaluate on split A?** The CNN trained on it and the AE threshold was tuned on it. Results there would be inflated.
6. **Why 128×128 for the AE and 224×224 for the CNN?** 128 keeps the AE small and fast; 224 is the size ResNet-18's ImageNet weights were trained at.
7. **Why a sigmoid output?** Input pixels are scaled to [0, 1]; the sigmoid keeps the rebuilt pixels in the same range.
8. **Why ImageNet normalisation for the CNN?** The pretrained weights expect inputs normalised that way.
9. **Why does the demo need no internet?** The CNN is built with `weights=None` and then our own trained weights are loaded from `weights/cnn.pt`.
10. **How do you know the demo matches the reported results?** One function (`inspect`) does all scoring,
    and a test checks the demo's score for every sample equals the score in `results/scores_split_b.csv`.
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
