# Sample images: source and licence

The 21 images in this folder are **unmodified copies** from the MVTec AD dataset,
category `bottle`, test set. Only the file names were changed to
`<folder>_<original number>.png` (e.g. `broken_small_010.png` = `test/broken_small/010.png`).

All 21 come from **split B**, the held-out half that no model was trained on, so the
demo shows the models on unseen bottles. Selection rule (no cherry-picking): all 10 good
images in split B, then the first 3 `broken_large`, first 3 `broken_small` and first 4
`contamination` images of split B by file name (the 20 images of Assignment 1). The final
system adds `broken_small_014.png`: the one split-B defect whose FoundAD score falls in
the REVIEW band (between the PASS and REJECT lines), so the demo exercises that path too.

## The demo camera feed (`line_demo.mp4`)

`line_demo.mp4` is rendered by `python -m src.make_line_video` from these same 21 images:
each photograph is resized, turned by a random angle and moved across a plain white
"belt" between two dark rails, so that the line runner can be demonstrated and tested
without a camera. The bottles are the dataset's photographs, unmodified apart from
scaling and rotation; nothing in the video is generated. `line_demo.manifest.json`
records which image is which part. The video is a derivative of the dataset and is
shared under the same CC BY-NC-SA 4.0 licence.

## Licence

Copyright 2019 MVTec Software GmbH. Licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International Licence (CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/

These copies are used for non-commercial university coursework and are shared
under the same licence.

## Citation

Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD: A comprehensive
real-world dataset for unsupervised anomaly detection. In *Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 9592–9600).

Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). The MVTec
anomaly detection dataset: A comprehensive real-world dataset for unsupervised anomaly
detection. *International Journal of Computer Vision, 129*(4), 1038–1059.
https://doi.org/10.1007/s11263-020-01400-4
