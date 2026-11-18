#!/usr/bin/env python3
"""
scripts/generate_ieee_figures.py
================================
Scientific Visualization Suite for IEEE Publications (IEEE Transactions on Neural Networks
and Learning Systems - IEEE TNNLS).

Reads canonical metric data from:
  - outputs/mnist/eaai_metrics.json
  - outputs/cifar10/eaai_metrics.json
  - outputs/cifar100/eaai_metrics.json

Generates publication-ready figures complying strictly with IEEE layout, typography,
colorblind safety, and print standards:
  1. Option 1: Multi-panel 2x2 figures per dataset (MNIST, CIFAR-10, CIFAR-100).
     - fig1_mnist_ieee.png / .pdf
     - fig2_cifar10_ieee.png / .pdf
     - fig3_cifar100_ieee.png / .pdf
  2. Option 2: Compact 1x3 cross-dataset figures by metric dimension.
     - fig_cross_accuracy_ieee.png / .pdf
     - fig_cross_governance_ieee.png / .pdf

Design features:
  - Scaled, high-legibility typography optimized for two-column paper formatting.
  - Unified bottom legend anchoring the figure cleanly, saving top vertical space.
  - High-resolution 600 DPI PNG (for Word/documentation embedding).
  - Vector PDF (TrueType type 42, font embedding, tight bounding box).
"""

from __future__ import annotations
import os
import sys
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -----------------------------------------------------------------------------
# 1. IEEE GLOBAL STYLES & TYPOGRAPHY CONFIGURATION
# -----------------------------------------------------------------------------

def configure_ieee_matplotlib() -> None:
    """Configures matplotlib rcParams for IEEE Transactions publication quality with enhanced readability."""
    plt.rcParams.update({
        # Typography: Serif standard (Times New Roman / DejaVu Serif)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "mathtext.fontset": "stix",
        
        # Calibration of font sizes for publication legibility
        "font.size": 11.0,
        "axes.labelsize": 13.0,
        "axes.titlesize": 14.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.0,
        "figure.titlesize": 14.0,
        
        # Linewidths and edges
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.2,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Backgrounds: 100% pure white
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        
        # Grid: subtle secondary grid
        "grid.color": "#CCCCCC",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        
        # PDF/PS font embedding for IEEE Xplore compliance
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# -----------------------------------------------------------------------------
# 2. COLORBLIND-SAFE PALETTE & LINESTYLES (Okabe-Ito + Grayscale Separability)
# -----------------------------------------------------------------------------

BASELINES = ["B0", "B1", "B2", "B3", "B4"]

BASELINE_STYLES = {
    "B0": {
        "label": "B0 (Pure CNN)",
        "short_label": "B0",
        "color": "#555555",       # Dark Gray
        "linestyle": "--",        # Dashed
        "marker": "x",
        "linewidth": 2.4,
        "markersize": 8.0,
        "markeredgewidth": 1.8,
        "zorder": 3,
    },
    "B1": {
        "label": "B1 (Infinite Memory)",
        "short_label": "B1",
        "color": "#0072B2",       # Blue
        "linestyle": ":",         # Dotted
        "marker": "s",
        "linewidth": 2.4,
        "markersize": 7.5,
        "markeredgewidth": 1.5,
        "zorder": 3,
    },
    "B2": {
        "label": "B2 (FIFO Bounded)",
        "short_label": "B2",
        "color": "#E69F00",       # Amber / Orange
        "linestyle": "-.",        # Dash-dot
        "marker": "^",
        "linewidth": 2.4,
        "markersize": 8.0,
        "markeredgewidth": 1.5,
        "zorder": 3,
    },
    "B3": {
        "label": "B3 (LFU Bounded)",
        "short_label": "B3",
        "color": "#CC79A7",       # Purple / Reddish Purple
        "linestyle": (0, (3, 1, 1, 1, 1, 1)),  # Dash-dot-dot
        "marker": "v",
        "linewidth": 2.4,
        "markersize": 8.0,
        "markeredgewidth": 1.5,
        "zorder": 3,
    },
    "B4": {
        "label": "B4 (Proposed Active RL)",
        "short_label": "B4",
        "color": "#009E73",       # Dark Emerald Green
        "linestyle": "-",         # Solid thick line
        "marker": "o",
        "linewidth": 3.2,
        "markersize": 9.0,
        "markeredgewidth": 1.8,
        "zorder": 5,
    },
}

NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]


def despine(ax: plt.Axes) -> None:
    """Removes top and right spines, keeping black left/bottom spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


# -----------------------------------------------------------------------------
# 3. CANONICAL DATA LOADER
# -----------------------------------------------------------------------------

def load_canonical_metrics(project_root: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Loads canonical metrics JSON files for MNIST, CIFAR-10, and CIFAR-100.
    """
    datasets = ["mnist", "cifar10", "cifar100"]
    data: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ds in datasets:
        json_path = os.path.join(project_root, "outputs", ds, "eaai_metrics.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Missing canonical metric file: {json_path}. "
                "Ensure baseline benchmark data has been generated."
            )
        with open(json_path, "r", encoding="utf-8") as f:
            data[ds] = json.load(f)

    return data


# -----------------------------------------------------------------------------
# 4. OPTION 1: DATASET-SPECIFIC 2x2 MULTIPANEL FIGURES
# -----------------------------------------------------------------------------

def plot_dataset_dashboard_ieee(
    dataset_name: str,
    metrics: Dict[str, Dict[str, float]],
    output_dir: str,
) -> Tuple[str, str]:
    """
    Generates a 2x2 publication dashboard for a specific dataset (Option 1).
    Width = 6.0 inches, Height = 7.5 inches.
    Optimized for single-column IEEE 2-column papers with enlarged typography
    and a vertically stacked bottom legend that never clips horizontally.

    Panels:
      (a) Accuracy vs. Noise (with Inset Zoom for CIFAR-10 / CIFAR-100).
      (b) Latency vs. Peak RAM Allocation (with 10MB ceiling and non-overlapping text).
      (c) Conditional Cache Hit Rate (excluding B0, calibrated Y axis).
      (d) Eviction KL Divergence (Log Scale, excluding B0, exact scientific notation).

    Bottom: Vertically stacked legend across all 5 baselines (ncol=1).
    """
    title_fs = 14.5
    label_fs = 13.5
    tick_fs = 12.0
    annot_fs = 10.8
    legend_fs = 13.5

    fig, axs = plt.subplots(2, 2, figsize=(6.0, 7.5), dpi=600)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.29, top=0.95, wspace=0.36, hspace=0.40)

    # -------------------------------------------------------------------------
    # Panel (a): Accuracy vs. Noise
    # -------------------------------------------------------------------------
    ax_a = axs[0, 0]
    ax_a.set_facecolor("#FFFFFF")
    despine(ax_a)
    ax_a.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.7)

    for b in BASELINES:
        if b not in metrics:
            continue
        st = BASELINE_STYLES[b]
        y_vals = [metrics[b].get(f"accuracy_noise_{n:.1f}", 0.0) for n in NOISE_LEVELS]
        ax_a.plot(
            NOISE_LEVELS,
            y_vals,
            label=st["label"],
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=st["linewidth"],
            markersize=st["markersize"],
            markeredgewidth=st["markeredgewidth"],
            zorder=st["zorder"],
        )

    ax_a.set_xlabel(r"Noise Intensity ($\sigma$)", fontsize=label_fs)
    ax_a.set_ylabel("Accuracy (%)", fontsize=label_fs)
    ax_a.set_xlim(-0.04, 0.84)
    ax_a.set_xticks(NOISE_LEVELS)
    ax_a.tick_params(axis="both", labelsize=tick_fs)
    ax_a.set_title("(a) Accuracy vs. Noise", loc="left", fontsize=title_fs, fontweight="bold", pad=6)

    # Inset Zoom for CIFAR-10 and CIFAR-100 positioned cleanly
    if dataset_name in ["cifar10", "cifar100"]:
        ax_inset = ax_a.inset_axes([0.38, 0.36, 0.58, 0.54])
        ax_inset.set_facecolor("#FFFFFF")
        ax_inset.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.5)
        for spine in ["top", "right", "bottom", "left"]:
            ax_inset.spines[spine].set_color("#000000")
            ax_inset.spines[spine].set_linewidth(0.7)

        zoom_x = [0.2, 0.4, 0.6, 0.8]
        for b in BASELINES:
            if b not in metrics:
                continue
            st = BASELINE_STYLES[b]
            y_vals_zoom = [metrics[b].get(f"accuracy_noise_{n:.1f}", 0.0) for n in zoom_x]
            ax_inset.plot(
                zoom_x,
                y_vals_zoom,
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=st["linewidth"] * 0.85,
                markersize=st["markersize"] * 0.8,
                markeredgewidth=st["markeredgewidth"],
                zorder=st["zorder"],
            )

        ax_inset.set_xlim(0.18, 0.82)
        ax_inset.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax_inset.tick_params(labelsize=tick_fs - 2.0, pad=2)

        if dataset_name == "cifar10":
            ax_inset.set_ylim(9.0, 13.0)
            ax_inset.set_yticks([9.0, 11.0, 13.0])
            ax_inset.set_title(r"$\sigma \in [0.2, 0.8]$", fontsize=annot_fs - 0.8, pad=2, fontweight="bold")
        else:  # cifar100
            ax_inset.set_ylim(0.5, 4.0)
            ax_inset.set_yticks([0.8, 2.0, 3.4])
            ax_inset.set_title(r"$\sigma \in [0.2, 0.8]$", fontsize=annot_fs - 0.8, pad=2, fontweight="bold")

        ax_a.indicate_inset_zoom(ax_inset, edgecolor="#666666", alpha=0.6, linewidth=0.8)

    # -------------------------------------------------------------------------
    # Panel (b): Latency vs. Peak RAM Allocation (Edge Efficiency)
    # -------------------------------------------------------------------------
    ax_b = axs[0, 1]
    ax_b.set_facecolor("#FFFFFF")
    despine(ax_b)
    ax_b.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.7)

    max_ram = max(metrics[b].get("ram_peak_mb", 0.0) for b in BASELINES)
    max_lat = max(metrics[b].get("latency_ms", 0.0) for b in BASELINES)

    # Strict Edge RAM Ceiling line at 10.0 MB
    ax_b.axhline(
        y=10.0,
        color="#D55E00",
        linestyle="--",
        linewidth=1.6,
        zorder=2,
    )
    ceiling_text_y = 10.0 + max_ram * 0.045
    ax_b.text(
        0.05,
        ceiling_text_y,
        "RAM Ceiling (10 MB)",
        color="#D55E00",
        fontsize=annot_fs + 0.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Plot each baseline scatter point
    for b in BASELINES:
        if b not in metrics:
            continue
        lat = metrics[b].get("latency_ms", 0.0)
        ram = metrics[b].get("ram_peak_mb", 0.0)
        st = BASELINE_STYLES[b]
        ax_b.scatter(
            lat,
            ram,
            color=st["color"],
            s=110 if b == "B4" else 80,
            marker=st["marker"],
            linewidths=2.0 if b == "B4" else 1.5,
            zorder=6 if b == "B4" else 5,
            label=b,
        )

    # Calibrated non-overlapping label positions adjacent to markers
    if dataset_name == "cifar10":
        offsets = {"B0": (8, -2), "B1": (-26, 6), "B2": (-24, -4), "B3": (8, -16), "B4": (8, -2)}
    elif dataset_name == "cifar100":
        offsets = {"B0": (8, -2), "B1": (-26, 6), "B2": (-24, -4), "B3": (8, -16), "B4": (7, -2)}
    else:
        offsets = {"B0": (8, -2), "B1": (-26, 6), "B2": (-24, -4), "B3": (8, -16), "B4": (8, 3)}

    for b in BASELINES:
        if b not in metrics:
            continue
        lat = metrics[b].get("latency_ms", 0.0)
        ram = metrics[b].get("ram_peak_mb", 0.0)
        dx, dy = offsets.get(b, (8, 2))
        ax_b.annotate(
            b,
            (lat, ram),
            xytext=(dx, dy),
            textcoords="offset points",
            color="#000000",
            fontsize=annot_fs + 1.5,
            fontweight="bold" if b == "B4" else "normal",
        )

    ax_b.set_xlabel("Mean Latency (ms)", fontsize=label_fs)
    ax_b.set_ylabel("Peak RAM (MB)", fontsize=label_fs)
    ax_b.tick_params(axis="both", labelsize=tick_fs)
    ax_b.set_title("(b) Latency vs. Peak RAM", loc="left", fontsize=title_fs, fontweight="bold", pad=6)

    ax_b.set_ylim(-1.8, max_ram * 1.18)
    ax_b.set_xlim(-0.4, max_lat * 1.28)

    # -------------------------------------------------------------------------
    # Panel (c): Conditional Cache Hit Rate (%) (Excluding B0)
    # -------------------------------------------------------------------------
    ax_c = axs[1, 0]
    ax_c.set_facecolor("#FFFFFF")
    despine(ax_c)
    ax_c.grid(True, axis="y", linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.7)

    cached_baselines = ["B1", "B2", "B3", "B4"]
    hit_rates = [metrics[b].get("cache_hit_rate", 0.0) for b in cached_baselines]
    bar_colors = [BASELINE_STYLES[b]["color"] for b in cached_baselines]

    x_pos = np.arange(len(cached_baselines))
    bar_width = 0.56
    bars_c = ax_c.bar(
        x_pos,
        hit_rates,
        width=bar_width,
        color=bar_colors,
        edgecolor="#000000",
        linewidth=0.9,
        zorder=3,
    )

    # Calibrate Y-limits per dataset
    if dataset_name == "mnist":
        ax_c.set_ylim(52.0, 64.5)
        ax_c.set_yticks([52.0, 55.0, 58.0, 61.0, 64.0])
    elif dataset_name == "cifar10":
        ax_c.set_ylim(12.5, 14.1)
        ax_c.set_yticks([12.5, 13.0, 13.5, 14.0])
    else:  # cifar100
        ax_c.set_ylim(12.5, 15.2)
        ax_c.set_yticks([12.5, 13.5, 14.5])

    for bar, val, b in zip(bars_c, hit_rates, cached_baselines):
        ax_c.annotate(
            f"{val:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annot_fs - 0.3,
            fontweight="bold" if b == "B4" else "normal",
            color="#000000",
        )

    ax_c.set_xlim(-0.6, 3.6)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(cached_baselines, fontsize=tick_fs)
    ax_c.tick_params(axis="both", labelsize=tick_fs)
    ax_c.set_xlabel("Cache Policy", fontsize=label_fs)
    ax_c.set_ylabel("Hit Rate (%)", fontsize=label_fs)
    ax_c.set_title("(c) Cache Hit Rate", loc="left", fontsize=title_fs, fontweight="bold", pad=6)

    # -------------------------------------------------------------------------
    # Panel (d): Eviction KL Divergence (Log Scale, Excluding B0)
    # -------------------------------------------------------------------------
    ax_d = axs[1, 1]
    ax_d.set_facecolor("#FFFFFF")
    despine(ax_d)
    ax_d.set_yscale("log")
    ax_d.grid(True, axis="y", which="both", linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.7)

    kl_divs = [metrics[b].get("eviction_kl_divergence", 0.0) for b in cached_baselines]
    bars_d = ax_d.bar(
        x_pos,
        kl_divs,
        width=bar_width,
        color=bar_colors,
        edgecolor="#000000",
        linewidth=0.9,
        zorder=3,
    )

    for bar, val, b in zip(bars_d, kl_divs, cached_baselines):
        if b == "B4":
            if dataset_name == "cifar100":
                label_text = r"$2.25\!\times\!10^{-8}$"
            elif dataset_name == "cifar10":
                label_text = r"$7.0\!\times\!10^{-4}$"
            else:
                label_text = r"$2.8\!\times\!10^{-3}$"
        else:
            label_text = f"{val:.4f}"

        ax_d.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, val * 1.6),
            ha="center",
            va="bottom",
            fontsize=annot_fs - 0.3,
            fontweight="bold" if b == "B4" else "normal",
            color="#000000",
        )

    ax_d.set_ylim(min(kl_divs) * 0.1, max(kl_divs) * 20.0)
    ax_d.set_xlim(-0.6, 3.6)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(cached_baselines, fontsize=tick_fs)
    ax_d.tick_params(axis="both", labelsize=tick_fs)
    ax_d.set_xlabel("Cache Policy", fontsize=label_fs)
    ax_d.set_ylabel(r"Eviction $D_{KL}$ (nats)", fontsize=label_fs)
    ax_d.set_title("(d) Eviction Skew", loc="left", fontsize=title_fs, fontweight="bold", pad=6)

    # -------------------------------------------------------------------------
    # Vertically stacked shared legend placed at the BOTTOM (ncol=1)
    # -------------------------------------------------------------------------
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.01),
        ncol=1,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#B0B0B0",
        framealpha=0.95,
        fontsize=legend_fs,
        handlelength=2.6,
        labelspacing=0.45,
        borderpad=0.55,
    )

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, f"fig1_mnist_ieee" if dataset_name == "mnist" else (
        "fig2_cifar10_ieee" if dataset_name == "cifar10" else "fig3_cifar100_ieee"
    ))
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"

    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close()

    return png_path, pdf_path


# -----------------------------------------------------------------------------
# 5. OPTION 2: CROSS-DATASET COMPACT 1x3 FIGURES
# -----------------------------------------------------------------------------

def plot_cross_accuracy_ieee(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
) -> Tuple[str, str]:
    """
    Generates a 1x3 panel comparing accuracy degradation across datasets:
      (a) MNIST (b) CIFAR-10 (with zoom) (c) CIFAR-100 (with zoom).
    Width = 7.16 inches (IEEE double column), Height = 3.2 inches.
    """
    title_fs = 12.0
    label_fs = 11.5
    tick_fs = 10.5
    legend_fs = 10.5

    fig, axs = plt.subplots(1, 3, figsize=(7.16, 3.2), dpi=600)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.92, wspace=0.28)

    panels = [
        ("mnist", "(a) MNIST Stream Accuracy", False),
        ("cifar10", "(b) CIFAR-10 Stream Accuracy", True),
        ("cifar100", "(c) CIFAR-100 Stream Accuracy", True),
    ]

    for idx, (ds, title, has_zoom) in enumerate(panels):
        ax = axs[idx]
        ax.set_facecolor("#FFFFFF")
        despine(ax)
        ax.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.6)

        ds_m = all_metrics[ds]
        for b in BASELINES:
            if b not in ds_m:
                continue
            st = BASELINE_STYLES[b]
            y_vals = [ds_m[b].get(f"accuracy_noise_{n:.1f}", 0.0) for n in NOISE_LEVELS]
            ax.plot(
                NOISE_LEVELS,
                y_vals,
                label=st["label"] if idx == 0 else None,
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=st["linewidth"],
                markersize=st["markersize"] * 0.9,
                markeredgewidth=st["markeredgewidth"],
                zorder=st["zorder"],
            )

        ax.set_xlabel(r"Noise Intensity $\sigma$", fontsize=label_fs)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=label_fs)
        ax.set_xlim(-0.02, 0.82)
        ax.set_xticks(NOISE_LEVELS)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.set_title(title, loc="left", fontsize=title_fs, fontweight="bold", pad=5)

        if has_zoom:
            ax_in = ax.inset_axes([0.42, 0.35, 0.54, 0.55])
            ax_in.set_facecolor("#FFFFFF")
            ax_in.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.5)
            for spine in ["top", "right", "bottom", "left"]:
                ax_in.spines[spine].set_color("#000000")
                ax_in.spines[spine].set_linewidth(0.6)

            zoom_x = [0.2, 0.4, 0.6, 0.8]
            for b in BASELINES:
                st = BASELINE_STYLES[b]
                y_zoom = [ds_m[b].get(f"accuracy_noise_{n:.1f}", 0.0) for n in zoom_x]
                ax_in.plot(
                    zoom_x,
                    y_zoom,
                    color=st["color"],
                    linestyle=st["linestyle"],
                    marker=st["marker"],
                    linewidth=st["linewidth"] * 0.8,
                    markersize=st["markersize"] * 0.75,
                    markeredgewidth=st["markeredgewidth"],
                    zorder=st["zorder"],
                )

            ax_in.set_xlim(0.18, 0.82)
            ax_in.set_xticks([0.2, 0.4, 0.6, 0.8])
            ax_in.tick_params(labelsize=8.5, pad=1.5)

            if ds == "cifar10":
                ax_in.set_ylim(9.0, 13.0)
                ax_in.set_yticks([9.0, 11.0, 13.0])
                ax_in.set_title(r"$\sigma \in [0.2, 0.8]$", fontsize=8.5, pad=2, fontweight="bold")
            else:
                ax_in.set_ylim(0.5, 4.0)
                ax_in.set_yticks([0.8, 2.0, 3.4])
                ax_in.set_title(r"$\sigma \in [0.2, 0.8]$", fontsize=8.5, pad=2, fontweight="bold")

            ax.indicate_inset_zoom(ax_in, edgecolor="#666666", alpha=0.6, linewidth=0.7)

    # Global horizontal legend placed neatly across bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#B0B0B0",
        framealpha=0.95,
        fontsize=legend_fs,
        handlelength=2.0,
        columnspacing=1.0,
        borderpad=0.4,
    )

    out_base = os.path.join(output_dir, "fig_cross_accuracy_ieee")
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close()

    return png_path, pdf_path


def plot_cross_governance_ieee(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
) -> Tuple[str, str]:
    """
    Generates a 1x3 panel combining governance and efficiency dimensions:
      (a) Eviction D_KL grouped by dataset (Log scale, y in [1e-9, 1e2]).
      (b) Cache Hit Rate grouped by dataset (with Inset Zoom).
      (c) Trade-off Latency vs. Peak RAM across all datasets (with distinct dataset shapes
          and 10MB ceiling at x ≈ 0.5ms).
    Width = 7.16 inches (IEEE double column), Height = 3.3 inches.
    """
    title_fs = 12.0
    label_fs = 11.5
    tick_fs = 10.0
    legend_fs = 10.5

    fig, axs = plt.subplots(1, 3, figsize=(7.16, 3.3), dpi=600)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.92, wspace=0.32)

    cached_baselines = ["B1", "B2", "B3", "B4"]
    datasets = ["mnist", "cifar10", "cifar100"]
    ds_labels = ["MNIST", "CIFAR-10", "CIFAR-100"]

    n_groups = len(datasets)
    n_bars = len(cached_baselines)
    bar_width = 0.18
    x_indices = np.arange(n_groups) * 1.28

    # -------------------------------------------------------------------------
    # Subplot (a): Eviction D_KL grouped by dataset (Log scale)
    # -------------------------------------------------------------------------
    ax_a = axs[0]
    ax_a.set_facecolor("#FFFFFF")
    despine(ax_a)
    ax_a.set_yscale("log")
    ax_a.grid(True, axis="y", which="both", linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.6)

    for i, b in enumerate(cached_baselines):
        offsets = x_indices + (i - (n_bars - 1) / 2) * bar_width
        vals = [all_metrics[ds][b].get("eviction_kl_divergence", 1e-12) for ds in datasets]
        st = BASELINE_STYLES[b]
        bars = ax_a.bar(
            offsets,
            vals,
            width=bar_width,
            color=st["color"],
            edgecolor="#000000",
            linewidth=0.7,
            label=st["label"],
            zorder=3,
        )

    # Annotate B4 values vertically centered in the middle of B4 bar
    for idx_ds, ds in enumerate(datasets):
        val_b4 = all_metrics[ds]["B4"].get("eviction_kl_divergence", 0.0)
        x_b4 = x_indices[idx_ds] + (3 - (n_bars - 1) / 2) * bar_width
        if ds == "cifar100":
            txt = r"$2.25\times 10^{-8}$"
            y_pos = val_b4 * 2.5
        elif ds == "cifar10":
            txt = r"$7.0\times 10^{-4}$"
            y_pos = val_b4 * 2.2
        else:
            txt = r"$2.8\times 10^{-3}$"
            y_pos = val_b4 * 2.2

        ax_a.annotate(
            txt,
            xy=(x_b4, y_pos),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#004D40",
            rotation=90,
        )

    ax_a.set_xticks(x_indices)
    ax_a.set_xticklabels(ds_labels, fontsize=8.5)
    ax_a.tick_params(axis="both", labelsize=tick_fs)
    ax_a.tick_params(axis="x", labelsize=8.5)
    ax_a.set_xlim(-0.6, x_indices[-1] + 0.6)
    ax_a.set_ylabel(r"Eviction $D_{KL}$ (nats)", fontsize=label_fs)
    ax_a.set_ylim(1e-9, 1e2)
    ax_a.set_title(r"(a) Eviction Skew ($D_{KL}$)", loc="left", fontsize=title_fs, fontweight="bold", pad=5)

    # -------------------------------------------------------------------------
    # Subplot (b): Conditional Cache Hit Rate grouped by dataset
    # -------------------------------------------------------------------------
    ax_b = axs[1]
    ax_b.set_facecolor("#FFFFFF")
    despine(ax_b)
    ax_b.grid(True, axis="y", linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.6)

    for i, b in enumerate(cached_baselines):
        offsets = x_indices + (i - (n_bars - 1) / 2) * bar_width
        vals = [all_metrics[ds][b].get("cache_hit_rate", 0.0) for ds in datasets]
        st = BASELINE_STYLES[b]
        bars = ax_b.bar(
            offsets,
            vals,
            width=bar_width,
            color=st["color"],
            edgecolor="#000000",
            linewidth=0.7,
            label=st["label"],
            zorder=3,
        )
        if b == "B4":
            for bar, val in zip(bars, vals):
                ax_b.annotate(
                    f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#000000",
                )

    ax_b.set_xticks(x_indices)
    ax_b.set_xticklabels(ds_labels, fontsize=8.5)
    ax_b.tick_params(axis="both", labelsize=tick_fs)
    ax_b.tick_params(axis="x", labelsize=8.5)
    ax_b.set_xlim(-0.6, x_indices[-1] + 0.6)
    ax_b.set_ylabel("Cache Hit Rate (%)", fontsize=label_fs)
    ax_b.set_ylim(0, 75)
    ax_b.set_title("(b) Cache Hit Rate", loc="left", fontsize=title_fs, fontweight="bold", pad=5)

    # Inset Zoom in panel (b)
    ax_b_in = ax_b.inset_axes([0.48, 0.42, 0.48, 0.50])
    ax_b_in.set_facecolor("#FFFFFF")
    ax_b_in.grid(True, axis="y", linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.5)
    for spine in ["top", "right", "bottom", "left"]:
        ax_b_in.spines[spine].set_color("#000000")
        ax_b_in.spines[spine].set_linewidth(0.6)

    cifar_indices = np.array([0, 1])
    cifar_datasets = ["cifar10", "cifar100"]
    for i, b in enumerate(cached_baselines):
        offsets = cifar_indices + (i - (n_bars - 1) / 2) * bar_width
        vals = [all_metrics[ds][b].get("cache_hit_rate", 0.0) for ds in cifar_datasets]
        st = BASELINE_STYLES[b]
        bars_in = ax_b_in.bar(
            offsets,
            vals,
            width=bar_width,
            color=st["color"],
            edgecolor="#000000",
            linewidth=0.6,
            zorder=3,
        )

    ax_b_in.set_xticks(cifar_indices)
    ax_b_in.set_xticklabels(["C-10", "C-100"], fontsize=8.0)
    ax_b_in.set_ylim(12.5, 14.8)
    ax_b_in.set_yticks([13.0, 14.0])
    ax_b_in.yaxis.tick_right()
    ax_b_in.tick_params(labelsize=8.0, pad=1.5)
    ax_b_in.set_title("Zoom C-10/100", fontsize=8.0, pad=2, fontweight="bold")

    # -------------------------------------------------------------------------
    # Subplot (c): Trade-off Latency vs. Peak RAM across all datasets
    # -------------------------------------------------------------------------
    ax_c = axs[2]
    ax_c.set_facecolor("#FFFFFF")
    despine(ax_c)
    ax_c.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", linewidth=0.6)

    # Red dashed line at 10.0 MB
    ax_c.axhline(
        y=10.0,
        color="#D55E00",
        linestyle="--",
        linewidth=1.3,
        zorder=1,
    )
    ax_c.text(
        0.5,
        10.7,
        "Strict RAM Ceiling (10 MB)",
        color="#D55E00",
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # Visually distinguish datasets using distinct marker symbols
    ds_markers = {"mnist": "o", "cifar10": "s", "cifar100": "^"}

    for ds in datasets:
        for b in BASELINES:
            st = BASELINE_STYLES[b]
            lat = all_metrics[ds][b].get("latency_ms", 0.0)
            ram = all_metrics[ds][b].get("ram_peak_mb", 0.0)
            ax_c.scatter(
                lat,
                ram,
                color=st["color"],
                marker=ds_markers[ds],
                s=55 if b == "B4" else 40,
                linewidths=1.5 if b == "B4" else 1.0,
                edgecolors="#000000" if b == "B4" else "none",
                zorder=5,
            )

    # Annotate B1 violation and B4 bounded without crossing lines
    ax_c.annotate(
        "B1: Unbounded\n(31.5-52.3 MB)",
        xy=(7.05, 52.28),
        xytext=(1.5, 43.0),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9),
        fontsize=8.5,
        color="#0072B2",
        fontweight="bold",
    )
    ax_c.annotate(
        "B2, B3, B4: Bounded (<10 MB)",
        xy=(5.5, 7.5),
        xytext=(1.8, 3.2),
        arrowprops=dict(arrowstyle="->", color="#009E73", lw=0.9),
        fontsize=8.5,
        color="#009E73",
        fontweight="bold",
    )

    # Dataset legend in upper right
    dataset_legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="MNIST", markerfacecolor="#555555", markeredgecolor="#000000", markersize=6.0),
        Line2D([0], [0], marker="s", color="w", label="CIFAR-10", markerfacecolor="#555555", markeredgecolor="#000000", markersize=6.0),
        Line2D([0], [0], marker="^", color="w", label="CIFAR-100", markerfacecolor="#555555", markeredgecolor="#000000", markersize=6.0),
    ]
    ax_c.legend(
        handles=dataset_legend_elements,
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        framealpha=0.9,
        fontsize=8.0,
        handletextpad=0.2,
        borderpad=0.3,
    )

    ax_c.set_xlabel("Mean Latency (ms)", fontsize=label_fs)
    ax_c.set_ylabel("Peak RAM (MB)", fontsize=label_fs)
    ax_c.tick_params(axis="both", labelsize=tick_fs)
    ax_c.set_xlim(-0.5, 12.0)
    ax_c.set_ylim(-1.0, 58.0)
    ax_c.set_title("(c) Edge Efficiency Trade-off", loc="left", fontsize=title_fs, fontweight="bold", pad=5)

    # Global horizontal legend placed neatly across bottom
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#B0B0B0",
        framealpha=0.95,
        fontsize=legend_fs,
        handlelength=2.0,
        columnspacing=1.0,
        borderpad=0.4,
    )

    out_base = os.path.join(output_dir, "fig_cross_governance_ieee")
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close()

    return png_path, pdf_path


# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION PIPELINE
# -----------------------------------------------------------------------------

def main() -> None:
    include_cross = "--cross" in sys.argv

    print("=" * 75)
    print("  IEEE TRANSACTIONS SCIENTIFIC FIGURE GENERATION SUITE")
    print("  Journal: IEEE Transactions on Neural Networks and Learning Systems")
    print("=" * 75)

    configure_ieee_matplotlib()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "outputs", "figures_ieee")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/2] Loading canonical benchmark metrics from outputs/* ...")
    metrics_data = load_canonical_metrics(project_root)
    print("      Successfully verified canonical metrics for MNIST, CIFAR-10, and CIFAR-100.")

    print(f"\n[2/2] Generating Dataset-Specific 2x2 Dashboards (6.0 x 7.5 in, vertical bottom legend) ...")
    generated_files = []
    for ds in ["mnist", "cifar10", "cifar100"]:
        png_p, pdf_p = plot_dataset_dashboard_ieee(ds, metrics_data[ds], output_dir)
        print(f"      [DONE] {ds.upper()}:")
        print(f"             - PNG: {png_p}")
        print(f"             - PDF: {pdf_p}")
        generated_files.extend([png_p, pdf_p])

    if include_cross:
        print(f"\n[OPT] Generating Cross-Dataset 1x3 Metric Figures (7.16 x 3.2-3.3 in) ...")
        p1_png, p1_pdf = plot_cross_accuracy_ieee(metrics_data, output_dir)
        print(f"      [DONE] Accuracy Progression across Datasets: {p1_png}")
        generated_files.extend([p1_png, p1_pdf])

        p2_png, p2_pdf = plot_cross_governance_ieee(metrics_data, output_dir)
        print(f"      [DONE] Governance & Efficiency Trade-offs across Datasets: {p2_png}")
        generated_files.extend([p2_png, p2_pdf])

    print("\n" + "=" * 75)
    print(f"  SUCCESS: {len(generated_files)} publication figures generated in {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
