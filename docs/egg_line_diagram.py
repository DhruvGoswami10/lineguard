"""Draws docs/egg_line_pipeline.png: how the egg line (second product, real footage) reuses the
line's subsystems. Run from the project folder: python docs/egg_line_diagram.py
Same colours as docs/architecture.png (Graphviz); matplotlib so no extra tool is needed."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOX, EDGE, STORE, AI, AI_EDGE, NEW, TEXT = "#dbeafe", "#1e3a8a", "#e2e8f0", "#fde68a", "#92400e", "#dcfce7", "#0f172a"


def box(ax, x, y, w, h, text, fill=BOX, edge=EDGE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.0, color=TEXT, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, a, b, label="", side="right", style="-|>", dashed=False):
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    if side == "right":
        start, end = (ax0 + aw, ay0 + ah / 2), (bx0, by0 + bh / 2)
    else:  # down
        start, end = (ax0 + aw / 2, ay0), (bx0 + bw / 2, by0 + bh)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, color="#334155",
                                 linewidth=1.1, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, ha="center", fontsize=7.4,
                color="#334155")


def main() -> None:
    fig, ax = plt.subplots(figsize=(17, 6.2), dpi=200)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(8.5, 5.95, "LineGuard egg line: second product on real conveyor footage", ha="center", fontsize=12.5,
            color=TEXT)
    ax.text(8.5, 5.62, "green = new for the egg line; blue / yellow = the line's existing subsystems, unchanged "
            "(numbers as in architecture.png)", ha="center", fontsize=8.6, color="#475569")

    # online row
    y, h = 3.55, 1.45
    cam = box(ax, 0.2, y, 1.75, h, "1. Acquisition\nrecorded clip, fixed\noverhead IP camera\n1080p, 20 fps")
    band = box(ax, 2.2, y, 1.75, h, "Clean band\nonly the rows above\nthe camera vendor's\ncounting overlay", NEW)
    find = box(ax, 4.2, y, 2.05, h, "2. Egg finder\nred minus blue colour;\ntwo thresholds, so pale\neggs come out whole;\ntouching eggs split", NEW)
    track = box(ax, 6.5, y, 2.05, h, "2. Tracker (egg_line.py)\nfollows each egg down\nthe belt; only eggs seen\nentering are inspected", NEW)
    crop = box(ax, 8.8, y, 1.75, h, "3. Pre-processing\nsquare crop, 1.3 x\nexpected egg size\n(fisheye size model)", NEW)
    det = box(ax, 10.8, y, 2.05, h, "4. AI defect detection\nFoundAD: frozen DINOv2\n+ predictor; scores the\ncentred disc of the crop",
              AI, AI_EDGE, 2)
    vote = box(ax, 13.1, y, 1.55, h, "Votes\nmedian of the\nfirst 5 clean\nframes")
    dec = box(ax, 14.9, y, 1.9, h, "5. Decision engine\nPASS / REVIEW /\nREJECT, severity\ndecision.py")
    for a, b in ((cam, band), (band, find), (find, track), (track, crop), (crop, det), (det, vote), (vote, dec)):
        arrow(ax, a, b)

    # outputs
    y2, h2 = 1.75, 1.15
    log = box(ax, 13.1, y2, 1.75, h2, "8. QC log\nSQLite, one row\nper egg; CSV", STORE)
    dash = box(ax, 15.1, y2, 1.7, h2, "9. Operator\ndashboard\n(same page)")
    vid = box(ax, 8.6, y2, 2.0, h2, "Annotated video\noutline per egg, verdict\nchip, heatmap panel", NEW)
    arrow(ax, dec, log, side="down")
    ax.add_patch(FancyArrowPatch((log[0] + log[2], y2 + h2 / 2), (dash[0], y2 + h2 / 2), arrowstyle="-|>",
                                 mutation_scale=12, color="#334155", linewidth=1.1))
    ax.add_patch(FancyArrowPatch((det[0] + 0.25, y), (vid[0] + vid[2] / 2, y2 + h2), arrowstyle="-|>",
                                 mutation_scale=12, color="#334155", linewidth=1.1))

    # offline row
    y3, h3 = 0.25, 1.15
    seg = box(ax, 0.2, y3, 2.3, h3, "First 105 s of the clip\none crop per egg (326)", STORE)
    eye = box(ax, 2.8, y3, 2.3, h3, "Contact-sheet review\n21 abnormal set aside\n(file + reason listed)", NEW)
    trn = box(ax, 5.4, y3, 2.3, h3, "11. Training\n100 normal crops,\nCutPaste, 2000 steps")
    cal = box(ax, 8.0, y3, 2.3, h3, "Calibration, no labels\n205 other normal crops\nREVIEW: mean + 2 sd\nREJECT: mean + 3 sd")
    w = box(ax, 10.6, y3, 2.0, h3, "weights/\nfoundad_egg.pt\nmodel + thresholds", STORE)
    for a, b in ((seg, eye), (eye, trn), (trn, cal), (cal, w)):
        arrow(ax, a, b)
    ax.add_patch(FancyArrowPatch((w[0] + w[2] / 2, y3 + h3), (det[0] + det[2] * 0.6, y), arrowstyle="-|>",
                                 mutation_scale=12, color="#334155", linewidth=1.1, linestyle="--"))
    ax.text(0.2, 1.55, "Offline (0-105 s)", fontsize=8.6, color="#475569")
    ax.text(0.2, 5.15, "Online (105 s to the end: never used for training or calibration)", fontsize=8.6,
            color="#475569")
    fig.savefig("docs/egg_line_pipeline.png", bbox_inches="tight", facecolor="white")
    print("wrote docs/egg_line_pipeline.png")


if __name__ == "__main__":
    main()
