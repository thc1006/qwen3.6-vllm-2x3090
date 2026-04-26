#!/usr/bin/env python3
"""Cross-hardware MTP / spec-decode comparison plot — UPDATED 2026-04-26 for v3.

Shows the 4 datapoints currently available for Qwen3.6-35B-A3B
single-stream batch=1 spec-decode delta, including the v3 clean A/B
retest that REVERSES the 3090 vLLM sign:

  1. 1× 3090 (llama.cpp draft)         : -38.6 %  (v2 srogmann config, N=3)
  2. 2× 3090 PCIe (vLLM MTP, v1)       : -12.0 %  (FLAG-CONFOUNDED)
  3. 2× A100 NVLink (vLLM MTP, v2)     : -11.4 %  (PREFIX-CACHE-ON regime)
  4. 2× 3090 PCIe (vLLM MTP, v3 clean) : +27.5 %  ← clean A/B, cache OFF

Key visual: bars 2 + 3 are now framed as confounded / cache-ON datapoints,
and bar 4 (v3) is the same hardware as bar 2 with the confounders removed,
showing the actual sign once the confound is controlled.

Run:
    python analysis/plot_cross_hardware.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# Datapoints (delta % vs each hardware's own baseline)
hardware_labels = [
    "1× 3090\nllama.cpp draft\n(srogmann config)",
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
    "v2.3 N=3 srogmann config\nllama.cpp+Q4+draft,\nmechanism still applies",
    "0.80/2 vs 0.90/8 flag mismatch\n+ prefix-caching ON\n→ SUPERSEDED by v3",
    "prefix-caching ON\n(vllm #38182:\nMTP cache hit rate 92→71%)\ncache-OFF retest pending",
    "matched 0.90/8/hermes,\n--no-enable-prefix-caching,\nstreaming N=5 trials,\n−21.6% decode TPOT",
]

n = len(hardware_labels)
fig, ax = plt.subplots(figsize=(13, 7.2), constrained_layout=True)
x = np.arange(n)
width = 0.55

bar_colors = ["#7F8C8D", "#C0392B", "#E67E22", "#27AE60"]
bars = ax.bar(
    x,
    deltas,
    width,
    color=bar_colors,
    edgecolor="#2C3E50",
    linewidth=1.0,
)

# Zero line
ax.axhline(0, color="#2C3E50", linewidth=1.2)

# Bar labels (delta % + metric)
for bar, d, m in zip(bars, deltas, delta_metrics):
    h = bar.get_height()
    label_y = h + (2.5 if h >= 0 else -2.5)
    va = "bottom" if h >= 0 else "top"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        f"{d:+.1f}%",
        ha="center",
        va=va,
        fontsize=14,
        fontweight="bold",
        color="#2C3E50",
    )
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y + (4.5 if h >= 0 else -4.5),
        f"({m})",
        ha="center",
        va=va,
        fontsize=8.5,
        style="italic",
        color="#34495E",
    )

# Status tags (above bar, between bar top and label)
for bar, tag in zip(bars, status_tags):
    h = bar.get_height()
    tag_y = h / 2  # mid of bar
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        tag_y,
        tag,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=status_colors[tag],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.92,
        ),
    )

# Notes below x-axis labels
for i, note in enumerate(notes):
    ax.text(
        x[i],
        -50,
        note,
        ha="center",
        va="top",
        fontsize=7.8,
        style="italic",
        color="#34495E",
    )

# Axes
ax.set_xticks(x)
ax.set_xticklabels(hardware_labels, fontsize=10)
ax.set_ylabel("Δ vs each hardware's baseline (%)", fontsize=11)
ax.set_ylim(-55, 40)
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Title + subtitle
fig.suptitle(
    "Qwen3.6-35B-A3B spec-decode delta — v3 retest (2026-04-26) "
    "REVERSES the 3090 vLLM sign",
    fontsize=14,
    fontweight="bold",
    y=1.04,
)
ax.set_title(
    "Bars 2+3 were confounded / cache-ON; bar 4 (v3) is the same 3090 hardware "
    "as bar 2 with both confounders removed.\n"
    "Bar 1 (llama.cpp draft) still valid — engine + spec-method specific, not engine-independent.",
    fontsize=10,
    color="#2C3E50",
    pad=8,
)

# Connect bar 2 → bar 4 with a dashed line + arrow showing the sign flip
ax.annotate(
    "",
    xy=(x[3], deltas[3] - 1),
    xytext=(x[1], deltas[1] + 1),
    arrowprops=dict(
        arrowstyle="->",
        color="#16A085",
        lw=2.5,
        linestyle=(0, (4, 2)),
        connectionstyle="arc3,rad=-0.18",
    ),
)
ax.text(
    (x[1] + x[3]) / 2,
    -2,
    "fix flag confound (+30 pp)\n+ prefix-cache OFF (+10 pp)",
    ha="center",
    va="top",
    fontsize=9,
    fontweight="bold",
    color="#16A085",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F8F5", edgecolor="#16A085"),
)

# Footer
footer_lines = [
    "v3 methodology: matched flags 0.90/8/hermes, --no-enable-prefix-caching, streaming TTFT separation, N=5 trials × 5 prompts (Exp 1) + N=5 × 20 reqs at C∈{1,4,8} (Exp 3).",
    "Open follow-up: A100 v3-equivalent retest with cache OFF — does the A100 also flip positive? (Modal credits permitting.)",
    "References: vllm #38182 (MTP × prefix-cache hit rate degradation), vllm #40756 (MTP + cache + chunked prefill crash). Theory: MoE-Spec arXiv 2602.16052 + Utility-Driven SD arXiv 2506.20675.",
]
for i, line in enumerate(footer_lines):
    fig.text(
        0.5,
        -0.03 - i * 0.025,
        line,
        ha="center",
        va="top",
        fontsize=7.8,
        color="#34495E",
        style="italic",
    )

# Save
out_path = pathlib.Path(__file__).parent / "plot_cross_hardware.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")
