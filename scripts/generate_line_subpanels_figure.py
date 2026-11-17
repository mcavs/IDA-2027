"""
Generate clean, professional line chart visualization:
- Main Layout: 3 horizontal panels (one per Setup: Setup 1, Setup 2, Setup 3)
- Sub-panels within each Setup: 3 sub-panels horizontally (one per noise variance: sigma^2 = 1, sigma^2 = 4, sigma^2 = 9) -> Total 1x3 subplots per Setup, or 3 rows x 3 columns of subplots!
- Or 3 major columns (Setups), each divided into 3 sub-columns (sigma^2 = 1, 4, 9) = 1 row x 9 subplots (or 3 rows x 3 columns).
Let's see: "Her bir panel kendi içerisinde yatayda 3 alt panele ayrılsın. Çizgiler kullanılsın."
In the previous design, there were 4 panels for epsilons or 3 panels for setups.
If each Setup is a main panel (row or major panel), and inside it horizontally divided into 3 sub-panels for sigma^2 in {1, 4, 9}:
On x-axis of each sub-panel: the 4 epsilon values (0.01, 0.05, 0.10, 0.20) completely horizontal!
Lines: 8 lines for 8 methods across epsilons.
B1 is solid black line with markers. Other methods have distinct colors.
Background color differentiation for the 3 sub-panels (sigma^2 = 1, 4, 9).
Single common Y-axis label in the center.
No 'Panel (a)' labels.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("outputs/master_analysis_results_2880s_160m.json") as f:
    data = json.load(f)

EPS_LIST = ["eps_0.01", "eps_0.05", "eps_0.1", "eps_0.2"]
EPS_X = [0.01, 0.05, 0.10, 0.20]
EPS_TICKS = ["0.01", "0.05", "0.10", "0.20"]
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

COLORS = {
    "B0:SimpleAvg":    "#1F77B4",
    "B1:Best-model":   "#000000",  # Solid Black
    "B2:PerfWeighted": "#2CA02C",
    "B3:AGGrobust":    "#FF7F0E",
    "B4:AVA-PDP":      "#9467BD",
    "B8:AGREE-Rank":   "#8C564B",
    "Ö1:AGGfaith":     "#17BECF",
    "Ö2:AGGopt":       "#D62728",
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
    "B0:SimpleAvg":    "--",
    "B1:Best-model":   "-",
    "B2:PerfWeighted": "-.",
    "B3:AGGrobust":    "-",
    "B4:AVA-PDP":      "-",
    "B8:AGREE-Rank":   ":",
    "Ö1:AGGfaith":     "--",
    "Ö2:AGGopt":       "-.",
}

SETUP_NAMES = {
    1: "Setup 1: Linear + Polynomial",
    2: "Setup 2: Trigonometric + Interaction",
    3: "Setup 3: Complex Nonlinear"
}

# Typography
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14
})

# 3 rows (Setups) x 3 columns (Noise variances) = 9 sub-panels
# This cleanly gives: each Setup panel is horizontally split into 3 sub-panels for sigma^2 in {1, 4, 9}
fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, sharey=True)

# Subtle background tint for the 3 noise variance columns to easily distinguish them
BG_COLORS = ["#FFFFFF", "#F5F8FA", "#EDF3F8"]

x_pos = np.arange(len(EPS_LIST))

for row_i, snum in enumerate([1, 2, 3]):
    for col_i, nv in enumerate(NOISE_VALS):
        ax = axes[row_i, col_i]
        
        # Background shading
        ax.set_facecolor(BG_COLORS[col_i])
        ax.grid(True, linestyle="--", alpha=0.5, color="#888888", zorder=1)
        
        sc_key = f"S{snum}_nv{nv}"
        sc_data = data[sc_key]["by_epsilon"]
        
        # Plot lines for all 8 methods
        for m in METHODS:
            y_vals = [sc_data[eps]["results"][m]["md"] for eps in EPS_LIST]
            
            # Formatting
            if m == "B1:Best-model":
                lw = 2.6
                ms = 7.5
                zorder = 5
                alpha = 1.0
            elif m in ("B4:AVA-PDP", "B3:AGGrobust", "B8:AGREE-Rank"):
                lw = 2.0
                ms = 6.5
                zorder = 4
                alpha = 0.95
            else:
                lw = 1.4
                ms = 5.0
                zorder = 3
                alpha = 0.85
                
            ax.plot(x_pos, y_vals, label=METHOD_LABELS[m],
                    color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m],
                    linewidth=lw, markersize=ms, alpha=alpha, zorder=zorder)
            
        # Titles: column top shows Noise, row right or title shows Setup
        if row_i == 0:
            ax.set_title(rf"$\mathbf{{\sigma^2 = {nv}}}$", fontsize=12, fontweight="bold", pad=8, color="#1B3B6F")
            
        if col_i == 0:
            # Put setup name clearly as a bold annotation
            ax.text(-0.35, 0.5, SETUP_NAMES[snum], transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=11, fontweight="bold", color="#222222")
            
        if row_i == 2:
            ax.set_xticks(x_pos)
            ax.set_xticklabels([rf"$\varepsilon={t}$" for t in EPS_TICKS], rotation=0)
            ax.set_xlabel(r"Rashomon parameter $\varepsilon$", fontsize=10.5)
            
        ax.set_ylim(0.04, 0.38)
        ax.set_xlim(-0.25, 3.25)

# Single common Y-axis label in the center
fig.text(0.025, 0.53, "Mean Deviation from Ground Truth, MD(GT)",
         va="center", ha="center", rotation="vertical", fontsize=12, fontweight="bold")

# Unified bottom legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, -0.04),
           ncol=8, frameon=True, framealpha=0.98, edgecolor="#C0C0C0", fontsize=9.5)

plt.tight_layout()
fig.subplots_adjust(left=0.08, bottom=0.09, hspace=0.18, wspace=0.10)

out_fig = "latex/figures/line_md_by_setup_and_noise.pdf"
out_png = "latex/figures/line_md_by_setup_and_noise.png"
plt.savefig(out_fig, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close()
print(f"Generated 3x3 Line Chart Figure: {out_fig}")
