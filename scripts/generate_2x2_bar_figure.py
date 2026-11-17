"""
Generate 2x2 grouped bar chart for Rashomon PDP aggregation:
- Layout: 2x2 grid corresponding to the 4 Rashomon thresholds:
    Top-left:     epsilon = 0.01
    Top-right:    epsilon = 0.05
    Bottom-left:  epsilon = 0.10
    Bottom-right: epsilon = 0.20
- NO 'Panel (a)' labels -> Use clean titles: r"$\varepsilon = 0.01$", etc.
- In each panel, 9 scenarios (S1-sigma^2=1,4,9; S2-sigma^2=1,4,9; S3-sigma^2=1,4,9):
    Delineated by alternating background color shading for Setup 1, Setup 2, Setup 3!
- X-axis tick labels: NOT 45 degrees! Fully horizontal (0 deg).
    Display only variance symbols and values: r"$\sigma^2{=}1$", r"$\sigma^2{=}4$", r"$\sigma^2{=}9$" (repeated for each setup).
    Setup names placed clearly at the top or bottom of the setup sections.
- Shared Y-axis: Single central Y-axis label on the left: 'Mean Deviation from Ground Truth, MD(GT)' (no duplicate Y labels).
- Bars: Distinct color bars for each method.
    B1: Best-model is SOLID BLACK.
    Other 7 methods have distinct, distinguishable colors.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load 160-model 2880s master results
with open("outputs/master_analysis_results_2880s_160m.json") as f:
    data = json.load(f)

EPS_KEYS = ["eps_0.01", "eps_0.05", "eps_0.1", "eps_0.2"]
EPS_TITLES = [r"$\varepsilon = 0.01$", r"$\varepsilon = 0.05$", r"$\varepsilon = 0.10$", r"$\varepsilon = 0.20$"]

SCENARIOS = [f"S{s}_nv{nv}" for s in [1, 2, 3] for nv in [1, 4, 9]]
# Just variance symbol & value: horizontal!
X_LABELS = [r"$\sigma^2{=}1$", r"$\sigma^2{=}4$", r"$\sigma^2{=}9$"] * 3

METHODS = [
    "B0:SimpleAvg", "B1:Best-model", "B2:PerfWeighted",
    "B3:AGGrobust",  "B4:AVA-PDP",    "B8:AGREE-Rank",
    "Ö1:AGGfaith",   "Ö2:AGGopt"
]

METHOD_LABELS = {
    "B0:SimpleAvg":    "B0: SimpleAvg",
    "B1:Best-model":   "B1: Best-model",
    "B2:PerfWeighted": "B2: PerfWeighted",
    "B3:AGGrobust":    "B3: AGGrobust",
    "B4:AVA-PDP":      "B4: AVA-PDP",
    "B8:AGREE-Rank":   "B8: AGREE-Rank",
    "Ö1:AGGfaith":     "Ö1: AGGfaith",
    "Ö2:AGGopt":       "Ö2: AGGopt",
}

# Colors: B1 is solid black, other 7 methods distinct colors
METHOD_COLORS = {
    "B0:SimpleAvg":    "#1F77B4",  # Blue
    "B1:Best-model":   "#111111",  # Solid Black
    "B2:PerfWeighted": "#2CA02C",  # Green
    "B3:AGGrobust":    "#FF7F0E",  # Orange
    "B4:AVA-PDP":      "#9467BD",  # Purple
    "B8:AGREE-Rank":   "#8C564B",  # Brown
    "Ö1:AGGfaith":     "#17BECF",  # Cyan/Teal
    "Ö2:AGGopt":       "#D62728",  # Crimson Red
}

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14
})

# 2x2 Grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), sharex=True, sharey=True)
axes_flat = axes.flatten()

n_scenarios = len(SCENARIOS)  # 9
n_methods = len(METHODS)      # 8
total_group_width = 0.82
bar_width = total_group_width / n_methods
x_indices = np.arange(n_scenarios)

# Background shading colors for the 3 Setups
SETUP_BG_COLORS = ["#FFFFFF", "#F3F7FA", "#EBF1F7"]
SETUP_NAMES = ["Setup 1 (Linear/Poly)", "Setup 2 (Trigonometric)", "Setup 3 (Nonlinear)"]

for p_idx, (eps_key, eps_title) in enumerate(zip(EPS_KEYS, EPS_TITLES)):
    ax = axes_flat[p_idx]
    
    # 1. Background shading for Setup 1 (0..2), Setup 2 (3..5), Setup 3 (6..8)
    for s_i in range(3):
        x_start = s_i * 3 - 0.5
        x_end = (s_i + 1) * 3 - 0.5
        ax.axvspan(x_start, x_end, color=SETUP_BG_COLORS[s_i], alpha=1.0, zorder=0)
        
        # Vertical divider between setups
        if s_i > 0:
            ax.axvline(x_start, color="#A8BACB", linestyle="--", linewidth=1.1, zorder=1)
            
        # Setup label header at top of each setup block
        mid_x = (x_start + x_end) / 2
        ax.text(mid_x, 0.358, SETUP_NAMES[s_i], ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="#1B3B6F",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#B0C4DE", lw=1.0, alpha=0.95))
        
    ax.grid(axis="y", linestyle=":", alpha=0.55, color="#777777", zorder=1)
    
    # 2. Draw Grouped Bars for all 9 scenarios
    for sc_i, sc in enumerate(SCENARIOS):
        sc_data = data[sc]["by_epsilon"][eps_key]["results"]
        x_center = sc_i
        
        for m_i, m in enumerate(METHODS):
            val = sc_data[m]["md"]
            bar_x = x_center - (total_group_width / 2) + (m_i + 0.5) * bar_width
            
            edge_color = "#000000" if m == "B1:Best-model" else "white"
            lw = 0.9 if m == "B1:Best-model" else 0.3
            ax.bar(bar_x, val, width=bar_width, color=METHOD_COLORS[m],
                   edgecolor=edge_color, linewidth=lw, zorder=3)
            
    ax.set_title(eps_title, fontweight="bold", pad=10)
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(0.0, 0.385)
    
    # X-axis ticks (fully horizontal!)
    if p_idx in [2, 3]:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(X_LABELS, rotation=0, fontweight="medium")

# Single shared Y-axis label in the center
fig.text(0.045, 0.53, "Mean Deviation from Ground Truth, MD(GT)",
         va="center", ha="center", rotation="vertical", fontsize=12, fontweight="bold")

# Unified Legend at the bottom
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=METHOD_COLORS[m], edgecolor="#000000" if m == "B1:Best-model" else "#CCCCCC",
          label=METHOD_LABELS[m]) for m in METHODS
]
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.53, -0.035),
           ncol=8, frameon=True, framealpha=0.98, edgecolor="#C0C0C0", fontsize=9.5)

plt.tight_layout()
fig.subplots_adjust(left=0.08, bottom=0.10, hspace=0.18, wspace=0.10)

out_fig = "latex/figures/grouped_bar_2x2_all_epsilons.pdf"
out_png = "latex/figures/grouped_bar_2x2_all_epsilons.png"
plt.savefig(out_fig, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close()
print(f"Generated 2x2 Grouped Bar Chart: {out_fig}")
