# Egg line: results (real conveyor footage)

<!-- written 2026-11-04 per Dhruv; numbers from results/egg_line/*.json|csv and results/logs/ -->

**Footage.** "Egg Counting with IP Camera on Conveyor Belt", CamContador, YouTube
[1EgUxp6BlBY](https://www.youtube.com/watch?v=1EgUxp6BlBY): fixed overhead IP camera over a farm
rod conveyor, 1920 x 1080, 20 fps, 179.8 s, standard YouTube licence (not committed). Only the band
above the camera vendor's counting overlay (rows 0-348) is used. Nothing in it is labelled.

**Model.** `weights/foundad_egg.pt`: technique 3 unchanged (frozen DINOv2 ViT-S/14, 10,943,232-parameter
predictor, CutPaste, JPEG round trip), foreground rule `disc`, trained 2000 steps on 100 normal egg
crops (CPU, 0.94 s/step, ~31 min). No defect example was used.

**Data.** 0-105 s: one crop per egg, 326 eggs. 21 set aside by eye as abnormal
(`splits/egg_line_excluded.csv`), 305 normal: 100 train, 205 calibrate. 105-179.8 s: the demo, never
used for training or calibration.

## Thresholds (label-free, from the 205 held-out normal crops)

| | value |
|---|---|
| mean / sd of their scores | 0.770 / 0.110 |
| REVIEW line (t_low) = mean + 2 sd | 0.991 |
| REJECT line (t_high) = mean + 3 sd | 1.101 |
| 99.5th percentile (the rule Roboflow's egg demo used, for comparison) | 1.110 |
| held-out normal crops at or above REVIEW / REJECT (single frame) | 9 / 3 of 205 (4.4 % / 1.5 %) |

## Check on the 21 crops set aside by eye (training segment; `calibration_check.json`)

Not an evaluation: they were picked from 128 px contact sheets before any score existed.

| Kind | Flagged | Detail |
|---|---|---|
| Marks, stains, flecks, specks, patches (13) | 8 (7 REJECT, 1 REVIEW) | dirt stain 11.37, purple mark 10.65, white flecks 10.10, pale patch 6.30, white specks 5.25, light flecks 2.70, pale shell with dark mark 1.93 (all REJECT); passed: dark speck 0.94, pale patch 0.85, white mark 0.85, white marks 0.84, green fleck 0.71 |
| White or pale shells (8) | 5 (1 REJECT, 4 REVIEW) | white egg 2.75 (REJECT); pale shells 1.10, 1.09, 1.05, 0.99 (REVIEW); passed: pale 0.86, white egg 0.85, pale 0.68 |

## Demo stretch (105-179.8 s; `inspections.csv`, `summary.json`)

234 eggs decided (215 on 5 frames, 8 on 4, 11 on 3): **PASS 210, REVIEW 13, REJECT 11**. 234 s of CPU
time for 74.8 s of footage on an Apple M5 Pro (scoring + rendering), so about 3x slower than real time.

Every decided egg was then looked at on contact sheets sorted by score (`python -m src.egg_demo review`),
at 128 px per egg; the sheets are in `media/review/` for the team to re-check:

| | REJECT | REVIEW | PASS |
|---|---|---|---|
| visible mark / stain / hole / speck | 6 (#69 green stain 12.67, #176 hole 7.50, #95 crack-like mark 5.39, #102 dark mark 5.24, #93 pale patch + dark mark 2.22, #138 dark speck 2.11) | 0 | none seen |
| white or pale shell | 0 | 3 (#37, #32, #119) | about 8, including the white egg #71 (0.77) |
| nothing visible (crowded crop: egg in a pile, centre between two eggs, blur) | 5 (#42 1.99, #46 1.85, #27 1.25, #114 1.14, #38 1.11) | 10 | the rest |

Takeaway: on this footage the detector catches marks and stains on the shell, strongly (the four
severe REJECTs score 4.8-11.5x the REJECT line; in the two checked, #69 and #176, the heatmap's hot
patch sits on the stain and on the hole). Shell colour is only weakly seen: some pale
eggs reach REVIEW, the white egg in the demo passes. False alarms come from crowded crops, mostly
REVIEW, all minor or moderate.

## How the numbers moved (three training runs, same recipe)

| Run | What changed | Normal crops (train / calibrate) | sd | REVIEW / REJECT | Demo: eggs, REVIEW, REJECT |
|---|---|---|---|---|---|
| 1 | first run | 356 (100 / 256) | 0.157 | 1.049 / 1.206 | 275, 1, 8 (one egg decided twice) |
| 2 | tracker: no take-over of the egg above; only eggs seen entering are inspected | 302 (100 / 202) | 0.337 | 1.481 / 1.818 | 224, 0, 5 (one marked egg never inspected) |
| 3 | entry rule loosened to 0.3 egg sizes (committed) | 305 (100 / 205) | 0.110 | 0.991 / 1.101 | 234, 13, 11 |

The mean + k sd rule moves with a few high held-out scores (crowded crops): the sd ranged 0.11-0.34
across runs. Thresholds were never chosen by looking at the demo stretch; the operator can move them
in the dashboard (`streamlit run src/dashboard.py -- --db results/egg_line/qc_log.sqlite`).

**Colour experiment** (two 300-step models on the run-1 crops, otherwise identical): the two white
eggs scored 1.5 and 1.5 sd above the held-out normal mean with FoundAD's colour / grey augmentation,
and 1.4 and 1.6 sd without it. The augmentation is not why colour is weakly seen; the frozen DINOv2
features are trained to be robust to colour changes. (The option was removed again: it did not help.)
