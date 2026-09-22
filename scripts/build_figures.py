#!/usr/bin/env python3
"""Figures for the README, drawn from the run's own logs.

    python3 scripts/build_figures.py        -> docs/figures/*.png

Every series here is read from logs/dume-cycle1.log (the final run) and
logs/archive/20260921_2145/ (the run before the thermometer). Nothing is
restated by hand.
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUT, ACC, BAD = "#1a1a1a", "#7a7a7a", "#1f6b3a", "#9c2b2b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUT, "ytick.color": MUT, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

def health(path):
    out = []
    for l in open(path, errors="replace"):
        m = re.match(r"\[health b(\d+)\]\s*(.*)", l.strip())
        if m:
            d = {k: float(v) for k, _, v in (kv.partition("=") for kv in m.group(2).split()) if _}
            d["b"] = int(m.group(1))
            out.append(d)
    return out

def batches(path, field):
    xs, ys = [], []
    for l in open(path, errors="replace"):
        m = re.match(r"\[b(\d+)\]", l)
        if not m:
            continue
        v = re.search(rf"{field}=(-?[\d.]+)", l)
        if v:
            xs.append(int(m.group(1))); ys.append(float(v.group(1)))
    return np.array(xs), np.array(ys)

def smooth(y, w):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w) / w, mode="valid")

H = health("logs/dume-cycle1.log")
OLD = []
for f in sorted(os.listdir("logs/archive/20260921_2145")):
    if f.endswith(".log"):
        OLD += health(os.path.join("logs/archive/20260921_2145", f))

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"docs/figures/{name}.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("docs/figures/" + name + ".png")

# 1 — the sensor. The whole argument in one picture.
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.4), sharex=False,
                             gridspec_kw={"height_ratios": [1, 1]})
# both panels on fraction-of-run, so two runs of different length compare
b = np.array([h["b"] for h in H]) / max(h["b"] for h in H)
a1.plot(b, [h["thermal"] for h in H], lw=.8, color=ACC, label="die temperature, mean of 24 sensors")
a1.plot(b, [h["thermal_mean"] for h in H], lw=1.7, color=INK, label="the normal it re-learns")
a1.set_ylabel("\u00b0C"); a1.set_ylim(52, 63)
a1.set_title("What the machine was actually doing  (final run, 3767 batches)")
a1.legend(loc="upper right", ncol=2, fontsize=8)
ob = np.array([h["b"] for h in OLD]) / max(h["b"] for h in OLD)
a2.plot(ob, [h["thermal"] for h in OLD], lw=2.0, color=BAD)
a2.set_ylim(-0.3, 3.3); a2.set_yticks([0, 1, 2, 3])
a2.set_yticklabels(["nominal", "fair", "serious", "critical"])
a2.set_xlabel("fraction of run")
a2.set_title("What NSProcessInfo reported, same machine, same workload  (earlier run)")
a2.text(.5, .62, "one value, every batch, every run", transform=a2.transAxes,
        ha="center", va="center", color=BAD, fontsize=9.5)
for ax in (a1, a2):
    ax.set_xlim(0, 1)
save(fig, "01-sensor")

# 2 — the device votes
fig, ax = plt.subplots(figsize=(7.6, 2.6))
ax.plot(b, [h["k_thermal"] for h in H], lw=.9, color=INK)
ax.axhline(4.0, lw=.8, ls=":", color=MUT)
ax.text(b[-1], 4.005, "k_max = 4, the physical bound", ha="right", va="bottom", color=MUT, fontsize=8)
ax.set_ylabel("k_thermal"); ax.set_xlabel("batch")
ax.set_title("The device's vote on k — 79 distinct values, floor 3.422 (previously: one value, forever)")
save(fig, "02-k-thermal")

# 3 — losses, off the batch lines (the health record carried stale values)
fig, ax = plt.subplots(figsize=(7.6, 2.9))
for f, c, lw in [("central_loss", INK, 1.4), ("gate_loss", ACC, 1.2)]:
    x, y = batches("logs/dume-cycle1.log", f)
    w = 60
    ax.plot(x[w-1:], smooth(y, w), lw=lw, color=c, label=f"{f}  ({len(y)} points)")
ax.set_xlabel("batch"); ax.set_ylabel("loss (60-point mean)")
ax.set_title("Losses, read off the per-batch lines — not off the health record")
ax.legend(loc="upper right")
save(fig, "03-losses")

# 4 — the run-fraction scaling, isolated. Computed from the actual (pressure,
# left) pairs, NOT from band midpoints: pressure is logged BEFORE scaling, so
# holding it fixed is what separates the scaling from the regulator settling.
hot = [h for h in H if h["thermal_p"] > 0]
bands = [(.75, 1.01, "1.00\u20130.75"), (.5, .75, "0.75\u20130.50"),
         (.25, .5, "0.50\u20130.25"), (0, .25, "0.25\u20130.00")]
lab, k1, ka, ns = [], [], [], []
for lo, hi, name in bands:
    r = [h for h in hot if lo <= h["run_left"] < hi]
    if not r: continue
    lab.append(name); ns.append(len(r))
    k1.append(np.mean([4.0 ** (1 / (1 + h["thermal_p"])) for h in r]))
    ka.append(np.mean([4.0 ** (1 / (1 + h["thermal_p"] * h["run_left"])) for h in r]))
fig, ax = plt.subplots(figsize=(7.6, 2.8))
xs = np.arange(len(lab))
ax.bar(xs - .18, k1, .34, color="#c9c9c4", label="k if the run had no end")
ax.bar(xs + .18, ka, .34, color=ACC, label="k actual")
for i, (a_, b_, n_) in enumerate(zip(k1, ka, ns)):
    ax.text(i + .18, b_ + .004, f"+{b_-a_:.3f}", ha="center", fontsize=8, color=ACC)
    ax.text(i, 3.712, f"n={n_}", ha="center", fontsize=7.5, color=MUT)
ax.set_xticks(xs); ax.set_xticklabels(lab); ax.set_ylim(3.70, 4.02)
ax.set_xlabel("fraction of the run still to do"); ax.set_ylabel("experts")
ax.set_title("Heat is discounted by how much run is left \u2014 pressure held fixed")
ax.legend(loc="upper left", fontsize=8)
save(fig, "04-scaling")
