"""
Refined grouped bar chart for Rashomon PDP aggregation:
- 3 panels (Setup 1, Setup 2, Setup 3)
- Background shading for sigma^2 in {1, 4, 9}
- Centered Y-axis label without overlapping ticks
- Horizontal X-ticks clearly showing epsilon values
- Top banners for sigma^2
- Solid black bar for B1, distinct colors for other 7 aggregation methods
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
EPS_LABELS = ["0.01", "0.05", "0.10", "0.20"]
NOISE_VALS = [1, 4, 9]

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

# Distinguishable colors: B1 is solid black
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

SETUP_TITLES = {
    1: "Setup 1: Linear + Polynomial",
    2: "Setup 2: Trigonometric + Interaction",
    3: "Setup 3: Complex Nonlinear"
}

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14
})

fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), sharey=True)

total_group_width = 0.82
bar_width = total_group_width / len(METHODS)

for s_idx, snum in enumerate([1, 2, 3]):
    ax = axes[s_idx]
    
    # Alternating background shading for noise variances
    bg_colors = ["#FFFFFF", "#F4F7FA", "#E9EFF5"]
    for nv_i in range(3):
        x_start = nv_i * 4 - 0.5
        x_end = (nv_i + 1) * 4 - 0.5
        ax.axvspan(x_start, x_end, color=bg_colors[nv_i], alpha=1.0, zorder=0)
        if nv_i > 0:
            ax.axvline(x_start, color="#A8BACB", linestyle="--", linewidth=1.1, zorder=1)
            
        # Top banner for noise variance
        mid_x = (x_start + x_end) / 2
        ax.text(mid_x, 0.360, rf"$\mathbf{{\sigma^2 = {NOISE_VALS[nv_i]}}}$",
                ha="center", va="top", fontsize=11,
                color="#1B3B6F", bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#B0C4DE", lw=1.2, alpha=0.98))

    ax.grid(axis="y", linestyle=":", alpha=0.55, color="#777777", zorder=1)
    
    cond_idx = 0
    x_ticks = []
    x_tick_labels = []
    
    for nv in NOISE_VALS:
        sc_key = f"S{snum}_nv{nv}"
        sc_data = data[sc_key]["by_epsilon"]
        
        for eps_i, eps_key in enumerate(EPS_LIST):
            x_center = cond_idx
            x_ticks.append(x_center)
            x_tick_labels.append(rf"$\varepsilon={EPS_LABELS[eps_i]}$")
            
            for m_i, m in enumerate(METHODS):
                val = sc_data[eps_key]["results"][m]["md"]
                bar_x = x_center - (total_group_width / 2) + (m_i + 0.5) * bar_width
                
                edge_color = "#000000" if m == "B1:Best-model" else "white"
                lw = 0.9 if m == "B1:Best-model" else 0.3
                ax.bar(bar_x, val, width=bar_width, color=METHOD_COLORS[m],
                       edgecolor=edge_color, linewidth=lw, zorder=3)
                
            cond_idx += 1
            
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=8.5)
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylim(0.0, 0.385)
    ax.set_title(SETUP_TITLES[snum], fontweight="bold", pad=12)

# Single common Y-axis label positioned clearly on the left without overlapping ticks
fig.text(0.045, 0.53, "Mean Deviation from Ground Truth, MD(GT)",
         va="center", ha="center", rotation="vertical", fontsize=12, fontweight="bold")

# Unified Legend at the bottom
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=METHOD_COLORS[m], edgecolor="#000000" if m == "B1:Best-model" else "#CCCCCC",
          label=METHOD_LABELS[m]) for m in METHODS
]
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.53, -0.05),
           ncol=8, frameon=True, framealpha=0.98, edgecolor="#C0C0C0", fontsize=9.5)

plt.tight_layout()
fig.subplots_adjust(left=0.08, bottom=0.14, wspace=0.10)

out_fig = "latex/figures/grouped_bar_md_by_setup.pdf"
out_png = "latex/figures/grouped_bar_md_by_setup.png"
plt.savefig(out_fig, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close()
print(f"Generated Refined Grouped Bar Chart: {out_fig}")
