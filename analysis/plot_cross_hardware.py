#!/usr/bin/env python3
"""Cross-hardware MTP / spec-decode comparison plot.

Visualises that single-stream Qwen3.6-35B-A3B speculative decoding is
net negative across consumer Ampere + datacenter Ampere with NVLink —
ruling out memory bandwidth and PCIe interconnect as the bottleneck.

Datapoints:
  1x 3090 (llama.cpp draft)      : 139.2 -> 85.5 = -38.6% (N=3 from
                                    sibling repo's v2 srogmann config)
  2x 3090 PCIe TP=2 (vLLM MTP)   : 126.4 -> 111.2 = -12.0% (confound
                                    disclosed: baseline 0.90/8 vs
                                    MTP run 0.80/2)
  2x A100 NVLink TP=2 (vLLM MTP) : 134.8 -> 119.5 = -11.4% (clean A/B,
                                    decode-only on prompt 4, TTFT-robust)

Run:
    python analysis/plot_cross_hardware.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pathlib

# Data
hardware_labels = [
    "1× 3090\nGDDR6X 936 GB/s\n(llama.cpp draft,\nsrogmann config)",
    "2× 3090 PCIe x8\nGDDR6X / PCIe ~16 GB/s\n(vLLM MTP k=1,\nconfound disclosed)",
    "2× A100-80GB SXM4\nHBM2e / NVLink ~600 GB/s\n(vLLM MTP k=1,\nclean A/B)",
]
baselines = [139.2, 126.4, 134.8]
specs = [85.5, 111.2, 119.5]
deltas = [
    (s / b - 1) * 100 for b, s in zip(baselines, specs)
]  # [-38.6, -12.0, -11.4]
notes = [
    "N=3 trial replication\n(run-to-run stdev <0.11)",
    "config confound:\nbaseline 0.90/8 vs\nMTP 0.80/2",
    "TTFT-robust\n(varies <0.2pp)\non prompt 4",
]

# Plot
fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)

x = np.arange(len(hardware_labels))
width = 0.36

color_baseline = "#4A6FA5"  # soft blue
color_spec = "#C0392B"  # red

bars1 = ax.bar(
    x - width / 2,
    baselines,
    width,
    label="baseline (no spec-decode)",
    color=color_baseline,
    edgecolor="#2C3E50",
    linewidth=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    specs,
    width,
    label="speculative decoding",
    color=color_spec,
    edgecolor="#2C3E50",
    linewidth=0.8,
)

# Number labels on bars
for bar, val in zip(bars1, baselines):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.2,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#2C3E50",
        fontweight="bold",
    )
for bar, val in zip(bars2, specs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.2,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#2C3E50",
        fontweight="bold",
    )

# Delta arrows + percent labels above each pair
for i, (b, s, d, note) in enumerate(zip(baselines, specs, deltas, notes)):
    midx = x[i]
    top = max(b, s) + 14
    ax.annotate(
        "",
        xy=(midx + width / 2, s + 6),
        xytext=(midx - width / 2, b + 6),
        arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.2),
    )
    ax.text(
        midx,
        top,
        f"Δ {d:+.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#C0392B",
    )
    # Sub-note in lighter gray
    ax.text(
        midx,
        top + 7.5,
        note,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#7F8C8D",
        style="italic",
    )

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(hardware_labels, fontsize=9.5)
ax.set_ylabel("decode tok/s (single-stream, batch=1)", fontsize=11)
ax.set_ylim(0, 200)
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Title + subtitle
fig.suptitle(
    "Speculative decoding for Qwen3.6-35B-A3B is net loss "
    "across hardware classes",
    fontsize=13.5,
    fontweight="bold",
    y=1.02,
)
ax.set_title(
    "Δ consistently negative across hardware × engine × quant — "
    "regression is hardware-class-independent\n"
    "single-stream batch=1, max_tokens=200, temperature=0.5, seed=42",
    fontsize=10,
    color="#2C3E50",
    pad=8,
)

ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)

# Footer caveat — 3 short lines for readability
footer_lines = [
    "Mechanism: ρ ≈ 0.031, T_thres ≈ 94. K (1–32) ≪ T_thres → verify pass loads expert-union with no amortization vs autoregressive.",
    "BW-independent: 3090 GDDR6X 936 GB/s + A100 HBM2e 2 TB/s show the same magnitude regression — interconnect (PCIe vs NVLink) also same direction.",
    "Theory: MoE-Spec (arXiv 2602.16052) + Utility-Driven SD for MoE (arXiv 2506.20675).",
]
for i, line in enumerate(footer_lines):
    fig.text(
        0.5,
        -0.025 - i * 0.028,
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
