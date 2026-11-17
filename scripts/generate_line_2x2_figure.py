"""
Generate publication-quality 2x2 line plot visualization combining Tables 3-4-5-6
(MD(GT) across all 4 epsilon thresholds: 0.01, 0.05, 0.10, 0.20 and 9 scenarios).

User specifications:
1. Retain the line plot design (çizgi grafiği).
2. X-axis labels: NO 45 degree rotation (horizontal 0°). Show ONLY variance symbols and values:
   r"$\sigma^2{=}1$", r"$\sigma^2{=}4$", r"$\sigma^2{=}9$" for each setup.
3. Alternating background shading per scenario setup (Setup 1, 2, 3) to clearly distinguish setups.
4. Clean titles without 'Panel (a)' expressions: directly $\varepsilon = 0.01$, $\varepsilon = 0.05$, etc.
5. Single, centered Y-axis label on the left (no repetitive Y-axis labels across subplots).
6. Distinct line/marker color for each method, with B1 (Best-model) solid black.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load 160-model 2880s master results
with open("outputs/master_analysis_results_2880s_160m.json") as f:
    data = json.load(f)

EPS_LIST = ["eps_0.01", "eps_0.05", "eps_0.1", "eps_0.2"]
EPS_TITLES = [r"$\varepsilon = 0.01$", r"$\varepsilon = 0.05$", r"$\varepsilon = 0.10$", r"$\varepsilon = 0.20$"]

SCENARIOS = [f"S{s}_nv{nv}" for s in [1, 2, 3] for nv in [1, 4, 9]]
# Only variance labels on x-axis:
X_TICK_LABELS = [
    r"$\sigma^2{=}1$", r"$\sigma^2{=}4$", r"$\sigma^2{=}9$",
    r"$\sigma^2{=}1$", r"$\sigma^2{=}4$", r"$\sigma^2{=}9$",
    r"$\sigma^2{=}1$", r"$\sigma^2{=}4$", r"$\sigma^2{=}9$"
]

METHODS = [
    "B0:SimpleAvg", "B1:Best-model", "B2:PerfWeighted",
    "B3:AGGrobust",  "B4:AVA-PDP",    "B8:AGREE-Rank",
    "Ö1:AGGfaith",   "Ö2:AGGopt"
]

METHOD_DISPLAY = [
    "B0: SimpleAvg", "B1: Best-model", "B2: PerfWeighted",
    "B3: AGGrobust",  "B4: AVA-PDP",    "B8: AGREE-Rank",
    "Ö1: AGGfaith",   "Ö2: AGGopt"
]

# Color palette: B1 is solid black, other methods are clearly distinct
COLORS = {
    "B0:SimpleAvg":    "#1F77B4",  # Blue
    "B1:Best-model":   "#000000",  # Solid Black
    "B2:PerfWeighted": "#2CA02C",  # Green
    "B3:AGGrobust":    "#FF7F0E",  # Orange
    "B4:AVA-PDP":      "#9467BD",  # Purple
    "B8:AGREE-Rank":   "#8C564B",  # Brown
    "Ö1:AGGfaith":     "#17BECF",  # Cyan
    "Ö2:AGGopt":       "#D62728",  # Red
}

MARKERS = {
    "B0:SimpleAvg":    "o",
    "B1:Best-model":   "s",
    "B2:PerfWeighted": "^",
    "B3:AGGrobust":    "D",
    "B4:AVA-PDP":      "v",
    "B8:AGREE-Rank":   "P",
    "Ö1:AGGfaith":     "<",
    "Ö2:AGGopt":       "X",
}

LINESTYLES = {
    "B0:SimpleAvg":    "-",
    "B1:Best-model":   "-",
    "B2:PerfWeighted": "-.",
    "B3:AGGrobust":    "-",
    "B4:AVA-PDP":      "-",
    "B8:AGREE-Rank":   "--",
    "Ö1:AGGfaith":     ":",
    "Ö2:AGGopt":       "-.",
}

LINEWIDTHS = {
    "B0:SimpleAvg":    1.8,
    "B1:Best-model":   2.6,
    "B2:PerfWeighted": 1.8,
    "B3:AGGrobust":    2.0,
    "B4:AVA-PDP":      2.2,
    "B8:AGREE-Rank":   2.0,
    "Ö1:AGGfaith":     1.8,
    "Ö2:AGGopt":       1.8,
}

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

x_indices = np.arange(len(SCENARIOS))

# Background shading blocks for setups
# Setup 1: x in [-0.5, 2.5]
# Setup 2: x in [2.5, 5.5]
# Setup 3: x in [5.5, 8.5]
bg_colors = ["#F3F7FA", "#FFFFFF", "#F3F7FA"]

for p_idx, (eps_key, eps_title) in enumerate(zip(EPS_LIST, EPS_TITLES)):
    ax = axes[p_idx]
    
    # Alternating background shading per setup
    ax.axvspan(-0.5, 2.5, facecolor=bg_colors[0], alpha=0.9, zorder=0)
    ax.axvspan(2.5, 5.5, facecolor=bg_colors[1], alpha=0.9, zorder=0)
    ax.axvspan(5.5, 8.5, facecolor=bg_colors[2], alpha=0.9, zorder=0)
    
    # Vertical divider lines between setups
    ax.axvline(2.5, color="#4682B4", linestyle="--", lw=1.2, alpha=0.55, zorder=1)
    ax.axvline(5.5, color="#4682B4", linestyle="--", lw=1.2, alpha=0.55, zorder=1)
    
    # Grid lines
    ax.grid(True, linestyle=":", alpha=0.6, color="#AAAAAA", zorder=1)
    
    # Setup title headers inside plot
    header_y = 0.355
    ax.text(1.0, header_y, "Setup 1 (Linear/Poly)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1A365D",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#B0C4DE", alpha=0.9, lw=0.8),
            zorder=5)
    ax.text(4.0, header_y, "Setup 2 (Trigonometric)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1A365D",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#B0C4DE", alpha=0.9, lw=0.8),
            zorder=5)
    ax.text(7.0, header_y, "Setup 3 (Nonlinear)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1A365D",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#B0C4DE", alpha=0.9, lw=0.8),
            zorder=5)
    
    # Plot curves for each method
    for m, m_label in zip(METHODS, METHOD_DISPLAY):
        y_vals = [data[sc]["by_epsilon"][eps_key]["results"][m]["md"] for sc in SCENARIOS]
        
        # Best-model on top (zorder=4), other lines zorder=3
        zo = 4 if m == "B1:Best-model" else 3
        ax.plot(x_indices, y_vals, label=m_label,
                color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m],
                linewidth=LINEWIDTHS[m], markersize=6.5, alpha=0.95, zorder=zo)
    
    # Panel title (no "Panel (a)" expression)
    ax.set_title(eps_title, fontweight="bold", pad=9)
    
    # Limits and ticks
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(0.05, 0.38)
    
    if p_idx in [2, 3]:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(X_TICK_LABELS, rotation=0, ha="center")
    else:
        ax.set_xticks(x_indices)

# Single centered Y-axis label on the left
fig.supylabel("Mean Deviation from Ground Truth, MD(GT)", fontsize=13, fontweight="bold", x=0.015)

# Unified horizontal legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           ncol=4, frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=11)

plt.tight_layout()
fig.subplots_adjust(bottom=0.10, left=0.065, right=0.985, top=0.95, wspace=0.08, hspace=0.15)

out_pdf = "latex/figures/combined_md_all_epsilons.pdf"
out_png = "latex/figures/combined_md_all_epsilons.png"
os.makedirs("latex/figures", exist_ok=True)
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close()
print(f"Generated 2x2 Line Plot: {out_pdf}")
