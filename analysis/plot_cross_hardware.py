#!/usr/bin/env python3
"""Cross-hardware MTP / spec-decode comparison plot — UPDATED 2026-04-26 for v3.

Shows the 4 datapoints for Qwen3.6-35B-A3B single-stream batch=1 spec-decode
delta, including the v3 clean A/B retest that REVERSES the 3090 vLLM sign:

  1. 1× 3090 (llama.cpp draft)         : -38.6 %  (v2 srogmann config, N=3)
  2. 2× 3090 PCIe (vLLM MTP, v1)       : -12.0 %  (FLAG-CONFOUNDED)
  3. 2× A100 NVLink (vLLM MTP, v2)     : -11.4 %  (PREFIX-CACHE-ON regime)
  4. 2× 3090 PCIe (vLLM MTP, v3 clean) : +27.5 %  ← clean A/B, cache OFF

Run:
    python analysis/plot_cross_hardware.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# Datapoints (delta % vs each hardware's own baseline)
hardware_labels = [
    "1× 3090\nllama.cpp draft\nsrogmann config",
    "2× 3090 PCIe TP=2\nvLLM MTP k=1\n[v1]",
    "2× A100-80GB\nSXM4 NVLink TP=2\nvLLM MTP k=1 [v2]",
    "2× 3090 PCIe TP=2\nvLLM MTP k=1\n[v3 — same HW as bar 2]",
]
deltas = [-38.6, -12.0, -11.4, +27.5]
delta_metrics = [
    "decode tok/s",
    "throughput (confounded)",
    "decode-only @ p4",
    "decode TPOT",
]
status_tags = [
    "STILL VALID",
    "CONFOUNDED",
    "CACHE-ON regime",
    "CLEAN A/B",
]
status_colors = {
    "STILL VALID": "#7F8C8D",
    "CONFOUNDED": "#C0392B",
    "CACHE-ON regime": "#E67E22",
    "CLEAN A/B": "#27AE60",
}
notes = [
    "v2.3 srogmann (N=3) + v3 DFlash (2026-05-07)\nllama.cpp + Q4 target\nboth NET LOSS",
    "0.80/2 vs 0.90/8 flag mismatch\n+ prefix-caching ON\n→ SUPERSEDED by v3",
    "prefix-caching ON (vllm #38182:\nMTP cache hit rate 92→71%)\nA100 cache-OFF retest pending",
    "matched 0.90/8/hermes flags\n--no-enable-prefix-caching · N=5\n−21.6% TPOT · v4 k=2/k=3 confirm",
]

n = len(hardware_labels)
fig, ax = plt.subplots(figsize=(14, 8.5), constrained_layout=False)
fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.34)

x = np.arange(n)
width = 0.6

bar_colors = ["#7F8C8D", "#C0392B", "#E67E22", "#27AE60"]
bars = ax.bar(
    x,
    deltas,
    width,
    color=bar_colors,
    edgecolor="#2C3E50",
    linewidth=1.2,
)

# Zero line
ax.axhline(0, color="#2C3E50", linewidth=1.4)

# Bar value labels (delta % above/below bar)
for bar, d, m in zip(bars, deltas, delta_metrics):
    h = bar.get_height()
    label_y = h + (3.0 if h >= 0 else -3.0)
    va = "bottom" if h >= 0 else "top"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        f"{d:+.1f}%",
        ha="center",
        va=va,
        fontsize=15,
        fontweight="bold",
        color="#2C3E50",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y + (5.5 if h >= 0 else -5.5),
        f"({m})",
        ha="center",
        va=va,
        fontsize=9,
        style="italic",
        color="#34495E",
    )

# Status tags inside bar middle
for bar, tag in zip(bars, status_tags):
    h = bar.get_height()
    tag_y = h / 2  # mid of bar
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        tag_y,
        tag,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=status_colors[tag],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.95,
        ),
    )

# Notes below x-axis labels (figure-fraction coords; sit between xtick labels and footers)
for i, note in enumerate(notes):
    fig.text(
        ax.get_position().x0 + (i + 0.5) / n * (ax.get_position().x1 - ax.get_position().x0),
        0.18,
        note,
        ha="center",
        va="top",
        fontsize=8.0,
        style="italic",
        color="#34495E",
    )

# Axes
ax.set_xticks(x)
ax.set_xticklabels(hardware_labels, fontsize=10.5)
ax.tick_params(axis="x", pad=8)
ax.set_ylabel("Δ vs each hardware's baseline (%)", fontsize=12)
ax.set_ylim(-58, 42)
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Title — single clear sentence
fig.suptitle(
    "Qwen3.6-35B-A3B spec-decode delta · v3 retest (2026-04-26) reverses the 3090 vLLM sign",
    fontsize=14,
    fontweight="bold",
    y=0.965,
)

# Subtitle right under suptitle, brief
fig.text(
    0.5,
    0.91,
    "Bar 4 (v3) = same 3090 hardware as bar 2, with both confounders removed (matched flags + prefix-cache OFF)",
    ha="center",
    va="top",
    fontsize=10.5,
    color="#2C3E50",
    style="italic",
)

# Footer — single line, tight
fig.text(
    0.5,
    0.06,
    "v3 methodology: matched 0.90/8/hermes serve flags · --no-enable-prefix-caching · streaming TTFT separation · N=5 trials × 5 prompts (Exp 1) + 20 reqs at C∈{1,4,8} (Exp 3)",
    ha="center",
    va="top",
    fontsize=8.5,
    color="#34495E",
    style="italic",
)
fig.text(
    0.5,
    0.035,
    "References: vllm #38182 (MTP × prefix-cache hit rate 92→71%), vllm #40756 (cache+MTP+chunked-prefill crash), MoESD arXiv 2505.19645, Utility-Driven SD arXiv 2506.20675"
    "  ·  v4 update (2026-05-07): k∈{1,2,3} sweep + vLLM 0.20.1 cross-version + 60-min stability — see v4_2026_05_07/",
    ha="center",
    va="top",
    fontsize=8,
    color="#34495E",
    style="italic",
)

# Save
out_path = pathlib.Path(__file__).parent / "plot_cross_hardware.png"
plt.savefig(out_path, dpi=150, bbox_inches=None, facecolor="white")
print(f"Saved: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")
