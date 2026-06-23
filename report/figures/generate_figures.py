"""Generate the publication-quality figures for the report.

Every figure here is produced from FIRST PRINCIPLES — illustrative schematics
drawn with matplotlib, or a *real* run of the project's own ``eldercare`` core
library on the synthetic stand->fall stream. No pretrained model weights are
used anywhere; the deep network is rendered from its architectural definition
(report/network/reference_model.py), not from a checkpoint.

Run:
    # from the repo root, with the project venv active
    python report/figures/generate_figures.py

Outputs PNGs (300 dpi) into report/figures/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# --- make the project's shared core/ importable ----------------------------
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "core"))

OUT = HERE.parent
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# House style — clean, serif, IEEE-ish.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

INK = "#1b2733"
ACCENT = "#2f6fed"   # blue   — system / data
GOOD = "#1f9d61"     # green  — privacy-safe / standing
ALERT = "#e2483b"    # red    — fall / alert
GREY = "#9aa6b2"
SOFT = "#eef3fb"

# COCO-17 joint names + skeleton edges (mirrors core/eldercare/schema.py).
COCO = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
IDX = {n: i for i, n in enumerate(COCO)}
EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def _box(ax, xy, w, h, text, fc=SOFT, ec=ACCENT, tc=INK, fs=9, lw=1.4, bold=False):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=tc, zorder=3,
        fontweight="bold" if bold else "normal", wrap=True,
    )
    return (x + w / 2, y + h / 2)


def _arrow(ax, p0, p1, color=INK, lw=1.6, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=12,
            lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )


# ===========================================================================
# FIGURE 1 — System architecture with the frame-egress=0 privacy boundary.
# ===========================================================================
def fig_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Edge node enclosure (the privacy boundary).
    edge = FancyBboxPatch(
        (0.4, 5.7), 9.2, 4.9,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=2.2, edgecolor=GOOD, facecolor="#f3fbf6", zorder=0,
        linestyle=(0, (6, 3)),
    )
    ax.add_patch(edge)
    ax.text(5.0, 10.32, "EDGE NODE  ·  Raspberry Pi 5 + Camera",
            ha="center", fontsize=11, fontweight="bold", color=GOOD)
    ax.text(5.0, 9.95, "raw frames live and die inside this boundary  —  frame-egress = 0",
            ha="center", fontsize=8.5, style="italic", color=GOOD)

    c = _box(ax, (0.9, 8.7), 1.9, 0.95, "Capture\n(OpenCV /\npicamera2)", fc="#ffffff", ec=GOOD)
    p = _box(ax, (3.2, 8.7), 1.9, 0.95, "Pose est.\nMoveNet /\nBlazePose", fc="#ffffff", ec=GOOD)
    w = _box(ax, (5.5, 8.7), 1.9, 0.95, "Temporal\nwindow\nT=32-64", fc="#ffffff", ec=GOOD)
    k = _box(ax, (7.8, 8.7), 1.5, 0.95, "Fall\nclassifier\nCTR-GCN", fc="#fff5f3", ec=ALERT)
    a = _box(ax, (5.5, 6.15), 1.9, 0.95, "Alarm logic\nEMA + k/m\nconfirm", fc="#ffffff", ec=GOOD)

    _arrow(ax, (2.8, 9.17), (3.2, 9.17), GOOD)
    _arrow(ax, (5.1, 9.17), (5.5, 9.17), GOOD)
    _arrow(ax, (7.4, 9.17), (7.8, 9.17), ALERT)
    _arrow(ax, (8.55, 8.7), (7.0, 7.1), ALERT, rad=-0.2)   # classifier -> alarm
    ax.text(2.7, 8.45, "frames", fontsize=7, color=GREY, ha="center")
    ax.text(5.0, 8.45, "17 keypoints (x,y,score)", fontsize=7, color=GREY, ha="center")

    # The wire — only telemetry crosses.
    _arrow(ax, (6.45, 6.15), (6.45, 4.95), ACCENT, lw=2.2)
    ax.text(6.65, 5.55, "WebSocket  ·  { keypoints[], fall_score, event, ts }",
            fontsize=8.5, color=ACCENT, fontweight="bold", va="center")
    ax.text(6.65, 5.25, "skeleton telemetry + alerts ONLY — never a pixel",
            fontsize=7.5, color=ACCENT, style="italic", va="center")

    # core/ shared library band.
    core = _box(ax, (0.4, 3.7), 9.2, 0.95,
                "core/  (single source of truth — Python, ported byte-for-byte to TypeScript)\n"
                "keypoint schema · EMA smoothing · geometric features · alarm confirm · metrics",
                fc=SOFT, ec=ACCENT, fs=8.5, bold=False)

    # Backend + frontend.
    be = _box(ax, (0.9, 1.6), 3.7, 1.5,
              "FastAPI BACKEND\nWS hub (fan-in nodes /\nfan-out dashboards)\nSQLModel · JWT · escalation",
              fc="#ffffff", ec=ACCENT, fs=8.5)
    fe = _box(ax, (5.4, 1.6), 3.7, 1.5,
              "REACT DASHBOARD\nlive skeleton overlay\nonnxruntime-web replay\nevent timeline · ack/dismiss",
              fc="#ffffff", ec=ACCENT, fs=8.5)

    _arrow(ax, (6.45, 4.95), (6.45, 4.65), ACCENT, lw=2.2)   # wire into core band
    _arrow(ax, (2.7, 3.7), (2.7, 3.1), ACCENT)
    _arrow(ax, (7.2, 3.7), (7.2, 3.1), ACCENT)
    ax.text(2.55, 3.4, "reused by", fontsize=7, color=GREY, ha="center")
    ax.text(7.05, 3.4, "ported to TS", fontsize=7, color=GREY, ha="center")
    _arrow(ax, (4.6, 2.35), (5.4, 2.35), ACCENT, style="<|-|>")
    ax.text(5.0, 2.5, "live frames", fontsize=7, color=GREY, ha="center")

    legend = [
        mpatches.Patch(facecolor="#f3fbf6", edgecolor=GOOD, label="privacy boundary (on-device)"),
        mpatches.Patch(facecolor="#fff5f3", edgecolor=ALERT, label="deep network (this report)"),
        mpatches.Patch(facecolor=SOFT, edgecolor=ACCENT, label="shared / off-device"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
              fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig. 1  System architecture — the skeleton is the privacy boundary",
                 fontsize=12, fontweight="bold", y=0.99)
    _save(fig, "fig1_system_architecture.png")


# ===========================================================================
# FIGURE 2 — CTR-GCN training network, block diagram.
# ===========================================================================
def fig_network_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    y = 3.6
    _box(ax, (0.2, y - 0.6), 1.7, 1.2,
         "Input\n(N, C=3, T, V=17)\nx, y, score", fc=SOFT, ec=GREY, fs=8)
    _arrow(ax, (1.9, y), (2.35, y))

    _box(ax, (2.35, y - 0.7), 1.85, 1.4,
         "Confidence\nGate\nmask + temporal\nimpute", fc="#fff5f3", ec=ALERT, fs=8, bold=True)
    _arrow(ax, (4.2, y), (4.65, y))

    _box(ax, (4.65, y - 0.6), 1.55, 1.2,
         "Data BN\n+ spatial\nnorm", fc=SOFT, ec=GREY, fs=8)
    _arrow(ax, (6.2, y), (6.6, y))

    # The stack of GCN blocks.
    block_x = 6.6
    chans = [64, 64, 64, 128, 128, 256, 256, 256, 256, 256]
    bw, gap = 0.42, 0.07
    for i, ch in enumerate(chans):
        x = block_x + i * (bw + gap)
        h = 0.7 + ch / 256 * 1.0
        fc = "#dfeafe" if ch == 64 else ("#bcd3fb" if ch == 128 else "#93b6f5")
        ax.add_patch(Rectangle((x, y - h / 2), bw, h, facecolor=fc,
                               edgecolor=ACCENT, lw=1.0, zorder=2))
    ax.annotate("", xy=(block_x + len(chans) * (bw + gap), y),
                xytext=(block_x - 0.05, y),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.text(block_x + len(chans) * (bw + gap) / 2, y + 1.55,
            "10 × CTR-GC blocks  (channels 64→128→256, stride-2 at blocks 4 & 6)",
            ha="center", fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.text(block_x + len(chans) * (bw + gap) / 2, y - 1.65,
            "each block = spatial CTR-GC  ⊕  temporal multi-scale conv  ⊕  residual",
            ha="center", fontsize=7.5, color=GREY, style="italic")

    xend = block_x + len(chans) * (bw + gap) + 0.15
    _arrow(ax, (xend, y), (xend + 0.4, y))
    _box(ax, (xend + 0.4, y - 0.55), 1.25, 1.1,
         "Global\nAvg Pool\n(T,V)", fc=SOFT, ec=GREY, fs=8)
    _arrow(ax, (xend + 1.65, y), (xend + 2.05, y))
    _box(ax, (xend + 2.05, y - 0.55), 1.3, 1.1,
         "FC → 2\nsoftmax\nfall / not", fc="#fff5f3", ec=ALERT, fs=8, bold=True)

    # Inset: the CTR-GC block internals.
    ix, iy, iw, ih = 0.6, 0.15, 12.8, 1.9
    ax.add_patch(FancyBboxPatch((ix, iy), iw, ih,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 linewidth=1.2, edgecolor=ACCENT, facecolor="#fbfdff", zorder=0))
    ax.text(ix + 0.15, iy + ih - 0.18, "CTR-GC block internals",
            fontsize=8.5, fontweight="bold", color=ACCENT, va="top")
    yb = iy + 0.75
    _box(ax, (ix + 1.0, yb - 0.32), 1.5, 0.7, "shared topology\nA  (static graph)", fc="#ffffff", ec=GREY, fs=7)
    _box(ax, (ix + 2.9, yb - 0.32), 1.9, 0.7, "channel-wise refine\nΔA = f(x_i − x_j)", fc="#fff5f3", ec=ALERT, fs=7, bold=True)
    _box(ax, (ix + 5.2, yb - 0.32), 1.7, 0.7, "A + αΔA per\nchannel group", fc="#ffffff", ec=GREY, fs=7)
    _box(ax, (ix + 7.3, yb - 0.32), 1.6, 0.7, "graph conv\nΣ (A·x)·W", fc="#ffffff", ec=GREY, fs=7)
    _box(ax, (ix + 9.2, yb - 0.32), 1.9, 0.7, "temporal conv\nmulti-scale {1,2}", fc="#ffffff", ec=GREY, fs=7)
    _box(ax, (ix + 11.3, yb - 0.32), 1.3, 0.7, "+ residual\nBN·ReLU", fc="#ffffff", ec=GREY, fs=7)
    for x0 in [ix + 2.5, ix + 4.8, ix + 6.9, ix + 8.9, ix + 11.1]:
        _arrow(ax, (x0, yb), (x0 + 0.18, yb), GREY, lw=1.1)

    fig.suptitle("Fig. 2  Proposed network — confidence-gated CTR-GCN fall classifier",
                 fontsize=12, fontweight="bold", y=0.99)
    _save(fig, "fig2_network_architecture.png")


# ===========================================================================
# FIGURE 3 — Spatial-temporal skeleton graph (the structure the GCN convolves).
# ===========================================================================
def _skeleton_xy(progress: float = 0.0):
    """A simple standing (progress=0) -> fallen (progress=1) skeleton layout."""
    base = {
        "nose": (0.50, 0.92), "left_eye": (0.47, 0.95), "right_eye": (0.53, 0.95),
        "left_ear": (0.44, 0.93), "right_ear": (0.56, 0.93),
        "left_shoulder": (0.42, 0.78), "right_shoulder": (0.58, 0.78),
        "left_elbow": (0.38, 0.62), "right_elbow": (0.62, 0.62),
        "left_wrist": (0.36, 0.47), "right_wrist": (0.64, 0.47),
        "left_hip": (0.45, 0.50), "right_hip": (0.55, 0.50),
        "left_knee": (0.45, 0.28), "right_knee": (0.55, 0.28),
        "left_ankle": (0.45, 0.08), "right_ankle": (0.55, 0.08),
    }
    pts = np.array([base[n] for n in COCO], dtype=float)
    if progress > 0:  # rotate ~80deg about the hips toward horizontal
        piv = (base["left_hip"][0], base["left_hip"][1])
        th = np.radians(80 * progress)
        c, s = np.cos(th), np.sin(th)
        out = []
        for x, yv in pts:
            dx, dy = x - piv[0], yv - piv[1]
            out.append((piv[0] + dx * c - dy * s, piv[1] + dx * s + dy * c - 0.25 * progress))
        pts = np.array(out)
    return pts


def fig_spatiotemporal_graph() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.45]})

    # (a) the COCO-17 spatial graph
    ax = axes[0]
    pts = _skeleton_xy(0.0)
    for a, b in EDGES:
        xa, ya = pts[IDX[a]]
        xb, yb = pts[IDX[b]]
        ax.plot([xa, xb], [ya, yb], color=ACCENT, lw=2, zorder=1)
    # color joints by partition (root / centripetal / centrifugal) — ST-GCN style
    root = pts.mean(0)
    for i, (x, yv) in enumerate(pts):
        d_self = np.hypot(x - root[0], yv - root[1])
        col = GOOD if d_self < 0.18 else (ACCENT if yv > root[1] else ALERT)
        ax.scatter([x], [yv], s=120, color=col, edgecolor="white", lw=1.2, zorder=3)
        ax.text(x, yv, str(i), ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", zorder=4)
    ax.set_title("(a) COCO-17 spatial graph A\npartitioned: root / centripetal / centrifugal", fontsize=9.5)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # (b) the spatial-temporal volume (graph replicated over T frames)
    ax = axes[1]
    progs = [0.0, 0.0, 0.5, 1.0]
    dx = 1.55
    for f, pr in enumerate(progs):
        pts = _skeleton_xy(pr)
        ox = f * dx
        for a, b in EDGES:
            xa, ya = pts[IDX[a]]
            xb, yb = pts[IDX[b]]
            ax.plot([xa + ox, xb + ox], [ya, yb], color=ACCENT, lw=1.3, zorder=1, alpha=0.9)
        ax.scatter(pts[:, 0] + ox, pts[:, 1], s=22, color=INK, zorder=2)
        # temporal edges to previous frame (same joint across time)
        if f > 0:
            for i in range(len(COCO)):
                ax.plot([prev[i, 0] + (f - 1) * dx, pts[i, 0] + ox],
                        [prev[i, 1], pts[i, 1]], color=GREY, lw=0.5, ls=":", zorder=0)
        ax.text(ox + 0.5, -0.12, f"t{f}", ha="center", fontsize=8, color=GREY)
        prev = pts
    ax.annotate("fall", xy=(2 * dx + 0.5, 0.55), xytext=(2 * dx + 0.5, 1.12),
                ha="center", fontsize=8, color=ALERT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ALERT))
    ax.set_title("(b) spatial-temporal graph — A replicated over T frames\n"
                 "blue = bones (spatial) · dotted grey = same joint across time (temporal)",
                 fontsize=9.5)
    ax.set_xlim(-0.2, 4 * dx + 0.2)
    ax.set_ylim(-0.2, 1.25)
    ax.axis("off")

    fig.suptitle("Fig. 3  The structure the network convolves over", fontsize=12,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "fig3_spatiotemporal_graph.png")


# ===========================================================================
# FIGURE 4 — Confidence gating: mask low-score joints, temporally impute.
# ===========================================================================
def fig_confidence_gating() -> None:
    rng = np.random.default_rng(7)
    T, J = 32, 17
    # a clean ground-truth y-trajectory per joint (just for illustration)
    base = np.linspace(0.1, 0.9, J)[None, :] + 0.05 * np.sin(
        np.linspace(0, 3, T))[:, None]
    conf = np.full((T, J), 0.9)
    # punch an occlusion hole: legs (joints 13-16) drop out frames 10-20
    conf[10:21, 13:17] = 0.05
    base_obs = base.copy()
    base_obs[10:21, 13:17] = np.nan  # masked observation

    # temporal impute = forward fill then back fill along time per joint
    imp = base_obs.copy()
    for j in range(J):
        col = imp[:, j]
        # forward fill
        last = np.nan
        for t in range(T):
            if np.isnan(col[t]):
                col[t] = last
            else:
                last = col[t]
        # back fill leading gap
        nxt = np.nan
        for t in range(T - 1, -1, -1):
            if np.isnan(col[t]):
                col[t] = nxt
            else:
                nxt = col[t]
        imp[:, j] = col

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, M, title, cmap in [
        (axes[0], conf, "(a) per-joint confidence\nlegs occluded, frames 10–20",
         "viridis"),
        (axes[1], np.where(np.isnan(base_obs), np.nan, base_obs),
         "(b) masked input\nlow-score joints removed", "magma"),
        (axes[2], imp, "(c) temporally imputed\nforward/back-fill from context", "magma"),
    ]:
        im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=cmap,
                       interpolation="nearest")
        ax.set_xlabel("frame  t")
        ax.set_title(title, fontsize=9)
        ax.set_yticks([0, 5, 11, 13, 16])
        ax.set_yticklabels(["nose", "shldr", "hip", "knee", "ankle"], fontsize=7)
        if M is conf:
            ax.set_ylabel("joint v")
        # highlight occlusion band
        ax.add_patch(Rectangle((9.5, 12.5), 11, 4, fill=False,
                               edgecolor=ALERT, lw=1.5, ls="--"))
    fig.suptitle("Fig. 4  Confidence-gated input layer — a brief occlusion never punches a hole in the tensor",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, "fig4_confidence_gating.png")


# ===========================================================================
# FIGURE 5 — REAL run of the project's heuristic pipeline on the synthetic
# stand->fall stream. This is a genuine, reproducible result, not a mock-up.
# ===========================================================================
def fig_real_signal_trace() -> None:
    from eldercare import features
    from eldercare.alarm import AlarmConfig, AlarmState
    from eldercare.heuristic import HeuristicFallDetector
    from eldercare.schema import EventType
    from eldercare.synthetic import synthetic_stream

    fps = 30
    dt = 1.0 / fps
    det = HeuristicFallDetector()
    alarm = AlarmState(AlarmConfig())

    ts, ar, tilt, vel, raw, smoothed = [], [], [], [], [], []
    prev = None
    alert_t = None
    impact_t = 3.0  # the synthetic generator starts the fall at stand_s=3.0 s
    for t, kps in synthetic_stream(fps=fps):
        a = features.bounding_box_aspect_ratio(kps)
        ti = features.head_to_hip_angle(kps)
        v = 0.0 if prev is None else features.centroid_vertical_velocity(prev, kps, dt)
        prob = det.score(kps, dt)
        ev = alarm.update(prob)
        ts.append(t); ar.append(a); tilt.append(ti); vel.append(v)
        raw.append(prob); smoothed.append(alarm.smoothed)
        if ev is EventType.FALL and alert_t is None:
            alert_t = t
        prev = kps

    ts = np.array(ts)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    ax.plot(ts, ar, color=ACCENT, lw=1.8, label="bbox aspect ratio  (w/h)")
    ax.plot(ts, np.array(tilt) / 90.0, color="#8a5cf6", lw=1.6,
            label="torso tilt  (deg / 90)")
    ax.plot(ts, np.clip(vel, 0, None), color="#e08a1e", lw=1.4,
            label="downward centroid vel.")
    ax.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax.set_ylabel("geometric features")
    ax.set_title("(a) interpretable cues from the real synthetic stand→fall episode", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=1)

    ax = axes[1]
    ax.plot(ts, raw, color=GREY, lw=1.0, label="raw fall score")
    ax.plot(ts, smoothed, color=ALERT, lw=2.0, label="EMA-smoothed score")
    ax.axhline(0.6, color=INK, lw=0.9, ls="--", label=r"threshold $\tau=0.6$")
    ax.axvspan(0, impact_t, color=GOOD, alpha=0.06)
    ax.axvline(impact_t, color=GOOD, lw=1.2, ls="-", label="impact (fall start)")
    if alert_t is not None:
        ax.axvline(alert_t, color=ALERT, lw=1.4, ls="-")
        ax.annotate(f"FALL alert\n+{alert_t - impact_t:.2f} s",
                    xy=(alert_t, 0.6), xytext=(alert_t + 0.35, 0.32),
                    fontsize=8, color=ALERT, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ALERT))
    ax.set_ylabel("fall probability")
    ax.set_xlabel("time  (s)")
    ax.set_title("(b) alarm pipeline: EMA → threshold → k-of-m confirm  (one debounced alert)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=2)
    ax.set_ylim(-0.03, 1.05)

    fig.suptitle("Fig. 5  Reproducible baseline result — real eldercare/ pipeline on the synthetic episode",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout()
    _save(fig, "fig5_real_signal_trace.png")
    return alert_t, impact_t


# ===========================================================================
# FIGURE 6 — Sensitivity vs false-alarm-rate operating curve (illustrative).
# ===========================================================================
def fig_operating_curve() -> None:
    far = np.linspace(0.02, 6, 300)  # alarms / hour
    # Illustrative monotone-saturating sensitivity curves vs nuisance rate.
    def curve(scale, ceil):
        return ceil * (1 - np.exp(-far / scale))
    heur = curve(1.6, 0.90)
    stgcn = curve(1.0, 0.955)
    ctr = curve(0.7, 0.985)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(far, heur, color=GREY, lw=2, label="Baseline A — geometric heuristic")
    ax.plot(far, stgcn, color=ACCENT, lw=2, label="ST-GCN (deep baseline)")
    ax.plot(far, ctr, color=ALERT, lw=2.4, label="CTR-GCN + confidence gating (proposed)")

    # chosen operating point at FAR = 1/hr
    op = 0.7
    s_op = 0.985 * (1 - np.exp(-op / 0.7))
    ax.axvline(1.0, color=INK, lw=1.0, ls="--")
    ax.scatter([1.0], [0.985 * (1 - np.exp(-1.0 / 0.7))], s=80, color=ALERT,
               zorder=5, edgecolor="white")
    ax.annotate("chosen operating point\n≤ 1 false alert / hour",
                xy=(1.0, 0.985 * (1 - np.exp(-1.0 / 0.7))),
                xytext=(2.0, 0.62), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("false-alarm rate  (alerts / hour / camera)")
    ax.set_ylabel("sensitivity  (recall on falls)")
    ax.set_title("Fig. 6  Operating curve — deployment cost on the x-axis, not balanced accuracy",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 6)
    ax.set_ylim(0.3, 1.01)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.text(0.02, 0.32, "illustrative target shapes — to be replaced by measured curves (Phase 5)",
            fontsize=7, style="italic", color=GREY)
    _save(fig, "fig6_operating_curve.png")


# ===========================================================================
# FIGURE 7 — Time-to-alert distribution (illustrative target).
# ===========================================================================
def fig_time_to_alert() -> None:
    rng = np.random.default_rng(3)
    # log-normal-ish latency, p50~0.8s p95~1.8s as the §7 budget targets
    samples = np.clip(rng.lognormal(mean=np.log(0.8), sigma=0.42, size=4000), 0.2, 4)
    p50 = np.percentile(samples, 50)
    p95 = np.percentile(samples, 95)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.hist(samples, bins=50, color=SOFT, edgecolor=ACCENT, lw=0.6)
    ax.axvline(p50, color=GOOD, lw=2, label=f"p50 = {p50:.2f} s")
    ax.axvline(p95, color=ALERT, lw=2, label=f"p95 = {p95:.2f} s")
    ax.axvspan(0, 0.7, color=GREY, alpha=0.08)
    ax.text(0.35, ax.get_ylim()[1] * 0.9, "confirmation\nwindow ~0.7 s",
            fontsize=7.5, ha="center", color=GREY)
    ax.set_xlabel("time-to-alert  (impact → alert event, s)")
    ax.set_ylabel("count")
    ax.set_title("Fig. 7  Time-to-alert distribution (illustrative target)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "fig7_time_to_alert.png")


# ===========================================================================
# FIGURE 8 — Cross-dataset generalization gap (the headline claim).
# ===========================================================================
def fig_crossdataset_gap() -> None:
    methods = ["Heuristic", "ST-GCN", "CTR-GCN\n(proposed)"]
    in_ds = [0.86, 0.97, 0.985]
    cross = [0.78, 0.70, 0.90]  # proposed loses little; naive deep loses a lot

    x = np.arange(len(methods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    b1 = ax.bar(x - w / 2, in_ds, w, color=ACCENT, label="in-dataset F1 (UP-Fall)")
    b2 = ax.bar(x + w / 2, cross, w, color=ALERT, label="cross-dataset F1 (URFD/Le2i, zero-shot)")
    for i in range(len(methods)):
        gap = in_ds[i] - cross[i]
        ax.annotate(f"−{gap*100:.0f} pts", xy=(x[i], cross[i]),
                    xytext=(x[i], cross[i] - 0.09), ha="center", fontsize=8,
                    color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Fig. 8  The generalization gap — our primary metric\n"
                 "naive deep models overfit staged data; gating + topology refine close the gap",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.12))
    ax.text(0.0, 0.04, "illustrative target values — measured numbers land in report/tables (Phase 5)",
            fontsize=7, style="italic", color=GREY, transform=ax.transData)
    _save(fig, "fig8_crossdataset_gap.png")


# ===========================================================================
# FIGURE 9 — Occlusion-robustness ablation (illustrative target).
# ===========================================================================
def fig_occlusion_ablation() -> None:
    dropout = np.array([0, 30, 50])
    no_gate = np.array([0.95, 0.84, 0.71])
    gate = np.array([0.96, 0.91, 0.83])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dropout, no_gate, "o-", color=GREY, lw=2, ms=8, label="unmasked baseline")
    ax.plot(dropout, gate, "s-", color=ALERT, lw=2.4, ms=8, label="confidence-gated (proposed)")
    for d, a, b in zip(dropout, no_gate, gate):
        ax.annotate(f"+{(b-a)*100:.0f}", xy=(d, (a + b) / 2), xytext=(d + 1.5, (a + b) / 2),
                    fontsize=8, color=ALERT, fontweight="bold")
    ax.set_xlabel("simulated joint dropout  (%)")
    ax.set_ylabel("sensitivity")
    ax.set_xticks(dropout)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Fig. 9  Occlusion-robust temporal fusion (illustrative target)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    _save(fig, "fig9_occlusion_ablation.png")


# ===========================================================================
# FIGURE 10 — Alarm state machine + k-of-m confirmation timing.
# ===========================================================================
def fig_alarm_state_machine() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.4),
                             gridspec_kw={"height_ratios": [1, 1.2]})

    # (a) state machine
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    s_arm = _box(ax, (0.6, 0.7), 2.2, 1.0, "ARMED\nema < τ", fc="#f3fbf6", ec=GOOD, bold=True)
    s_cnt = _box(ax, (3.9, 0.7), 2.2, 1.0, "COUNTING\nema ≥ τ\n< k of m", fc=SOFT, ec=ACCENT, bold=True)
    s_lat = _box(ax, (7.2, 0.7), 2.2, 1.0, "LATCHED\nFALL fired", fc="#fff5f3", ec=ALERT, bold=True)
    _arrow(ax, (2.8, 1.2), (3.9, 1.2), ACCENT)
    ax.text(3.35, 1.42, "ema ≥ τ", fontsize=7.5, ha="center", color=ACCENT)
    _arrow(ax, (6.1, 1.2), (7.2, 1.2), ALERT)
    ax.text(6.65, 1.42, "k of last m", fontsize=7.5, ha="center", color=ALERT)
    _arrow(ax, (8.3, 0.7), (1.7, 0.5), GOOD, rad=0.35)
    ax.text(5.0, 0.15, "ema < τ  →  re-arm (will not re-fire while still down)",
            fontsize=7.5, ha="center", color=GOOD)
    _arrow(ax, (4.6, 0.7), (2.5, 0.55), GREY, rad=0.25)
    ax.set_title("(a) alarm state machine — debounce + latch", fontsize=9.5)

    # (b) k-of-m timing trace
    ax = axes[1]
    rng = np.random.default_rng(1)
    n = 40
    t = np.arange(n)
    prob = np.concatenate([
        rng.uniform(0.0, 0.3, 14),
        np.clip(rng.normal(0.85, 0.08, 16), 0, 1),
        rng.uniform(0.0, 0.25, 10),
    ])
    # EMA
    ema = np.zeros(n)
    for i in range(1, n):
        ema[i] = 0.3 * prob[i] + 0.7 * ema[i - 1]
    tau, k, m = 0.6, 5, 8
    above = ema >= tau
    fire = None
    win = []
    latched = False
    for i in range(n):
        win.append(above[i])
        if len(win) > m:
            win.pop(0)
        if not above[i]:
            latched = False
        elif sum(win) >= k and not latched:
            latched = True
            if fire is None:
                fire = i

    ax.bar(t, prob, color=GREY, alpha=0.35, width=0.8, label="raw fall score")
    ax.plot(t, ema, color=ALERT, lw=2, label="EMA")
    ax.axhline(tau, color=INK, lw=0.9, ls="--", label=r"$\tau=0.6$")
    ax.fill_between(t, 0, 1, where=above, color=ACCENT, alpha=0.08, step="mid")
    if fire is not None:
        ax.axvline(fire, color=ALERT, lw=1.6)
        ax.annotate("FALL\n(k=5 of m=8)", xy=(fire, tau),
                    xytext=(fire + 1.5, 0.78), fontsize=8, color=ALERT,
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color=ALERT))
    ax.set_xlabel("frame")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) k-of-m confirmation suppresses single-frame spikes", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)

    fig.suptitle("Fig. 10  From a noisy per-frame score to one debounced alert",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout()
    _save(fig, "fig10_alarm_state_machine.png")


def main() -> None:
    print("Generating report figures (no pretrained weights used) ...")
    fig_system_architecture()
    fig_network_architecture()
    fig_spatiotemporal_graph()
    fig_confidence_gating()
    alert_t, impact_t = fig_real_signal_trace()
    fig_operating_curve()
    fig_time_to_alert()
    fig_crossdataset_gap()
    fig_occlusion_ablation()
    fig_alarm_state_machine()
    print(f"\nDone. Real-pipeline result: time-to-alert = {alert_t - impact_t:.2f} s "
          f"(impact {impact_t:.1f}s -> alert {alert_t:.2f}s).")
    print(f"Figures in {OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
