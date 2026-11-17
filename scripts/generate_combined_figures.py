"""
Generate publication-quality multi-panel visualization combining Tables 3-4-5-6
(MD(GT) across all 4 epsilon thresholds: 0.01, 0.05, 0.10, 0.20 and 9 scenarios).
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
EPS_LABELS = [r"$\varepsilon = 0.01$", r"$\varepsilon = 0.05$", r"$\varepsilon = 0.10$", r"$\varepsilon = 0.20$"]
SCENARIOS = [f"S{s}_nv{nv}" for s in [1, 2, 3] for nv in [1, 4, 9]]
SCEN_LABELS = [f"S{s}-" + r"$\sigma^2{=}" + f"{nv}$" for s in [1, 2, 3] for nv in [1, 4, 9]]

METHODS = [
    "B0:SimpleAvg", "B1:Best-model", "B2:PerfWeighted",
    "B3:AGGrobust",  "B4:AVA-PDP",    "B8:AGREE-Rank",
    "Ö1:AGGfaith",   "Ö2:AGGopt"
]
METHOD_SHORT = ["B0", "B1", "B2", "B3", "B4", "B8", "Ö1", "Ö2"]

COLORS = {
    "B0:SimpleAvg":    "#2C7BB6",
    "B1:Best-model":   "#111111",
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

# -------------------------------------------------------------------------
# Figure 1: 2x2 Subplots (One per Epsilon threshold)
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14
})

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

x_indices = np.arange(len(SCENARIOS))

for p_idx, (eps_key, eps_label) in enumerate(zip(EPS_LIST, EPS_LABELS)):
    ax = axes[p_idx]
    
    # Grid and background styling
    ax.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.set_facecolor("#FAFAFA")
    
    # Vertical separators between setups
    ax.axvline(2.5, color="#888888", linestyle=":", lw=1.2, alpha=0.7)
    ax.axvline(5.5, color="#888888", linestyle=":", lw=1.2, alpha=0.7)
    
    # Text headers for setups
    y_top = 0.355
    ax.text(1.0, y_top, "Setup 1 (Linear/Poly)", ha="center", va="top", fontsize=9, fontweight="bold", color="#444444")
    ax.text(4.0, y_top, "Setup 2 (Trigonometric)", ha="center", va="top", fontsize=9, fontweight="bold", color="#444444")
    ax.text(7.0, y_top, "Setup 3 (Nonlinear)", ha="center", va="top", fontsize=9, fontweight="bold", color="#444444")
    
    for m in METHODS:
        y_vals = []
        for sc in SCENARIOS:
            md_val = data[sc]["by_epsilon"][eps_key]["results"][m]["md"]
            y_vals.append(md_val)
            
        lw = 2.4 if m in ("B1:Best-model", "B4:AVA-PDP", "B8:AGREE-Rank") else 1.4
        ms = 7 if m in ("B1:Best-model", "B4:AVA-PDP", "B8:AGREE-Rank") else 5
        alpha = 0.95 if m in ("B1:Best-model", "B4:AVA-PDP", "B8:AGREE-Rank", "B3:AGGrobust") else 0.8
        
        ax.plot(x_indices, y_vals, label=m.split(":")[0] + f" ({m.split(':')[1]})",
                color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m],
                linewidth=lw, markersize=ms, alpha=alpha, zorder=4)
        
    ax.set_title(f"Panel ({chr(97+p_idx)}): {eps_label}", fontweight="bold", pad=8)
    if p_idx in [0, 2]:
        ax.set_ylabel("Mean Deviation from Ground Truth, MD(GT)")
    if p_idx in [2, 3]:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(SCEN_LABELS, rotation=35, ha="right")
    ax.set_ylim(0.05, 0.38)

# Unified Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
           ncol=4, frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

plt.tight_layout()
fig.subplots_adjust(bottom=0.13)

out_fig1 = "latex/figures/combined_md_all_epsilons.pdf"
out_png1 = "latex/figures/combined_md_all_epsilons.png"
plt.savefig(out_fig1, bbox_inches="tight", dpi=300)
plt.savefig(out_png1, bbox_inches="tight", dpi=200)
plt.close()
print(f"Generated 2x2 Panel Figure: {out_fig1}")


# -------------------------------------------------------------------------
# Figure 2: Heatmap Matrix (4 panels x heatmap: Scenario vs Method)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)

for p_idx, (eps_key, eps_label) in enumerate(zip(EPS_LIST, EPS_LABELS)):
    ax = axes[p_idx]
    mat = np.zeros((len(SCENARIOS), len(METHODS)))
    
    for r_idx, sc in enumerate(SCENARIOS):
        for c_idx, m in enumerate(METHODS):
            mat[r_idx, c_idx] = data[sc]["by_epsilon"][eps_key]["results"][m]["md"]
            
    im = ax.imshow(mat, cmap="YlGnBu_r", aspect="auto", vmin=0.07, vmax=0.35)
    
    ax.set_xticks(np.arange(len(METHOD_SHORT)))
    ax.set_xticklabels(METHOD_SHORT, rotation=45, ha="right", fontsize=9)
    if p_idx == 0:
        ax.set_yticks(np.arange(len(SCEN_LABELS)))
        ax.set_yticklabels(SCEN_LABELS, fontsize=9)
        ax.set_ylabel("Scenario")
    ax.set_title(eps_label, fontweight="bold", fontsize=11)
    
    # Text in cells
    for r in range(len(SCENARIOS)):
        best_c = np.argmin(mat[r, :])
        for c in range(len(METHODS)):
            val = mat[r, c]
            txt = f"{val:.2f}"
            bold = "bold" if c == best_c else "normal"
            star = "★" if c == best_c else ""
            txt_full = f"{txt}{star}"
            color = "white" if val < 0.14 or val > 0.28 else "black"
            ax.text(c, r, txt_full, ha="center", va="center", fontsize=7.5, fontweight=bold, color=color)

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="MD(GT) [Lower is better, ★=Best]")
plt.suptitle("MD(GT) Comparison across Rashomon Thresholds and Aggregation Methods", fontweight="bold", fontsize=13, y=1.02)

out_fig2 = "latex/figures/heatmap_md_all_epsilons.pdf"
out_png2 = "latex/figures/heatmap_md_all_epsilons.png"
plt.savefig(out_fig2, bbox_inches="tight", dpi=300)
plt.savefig(out_png2, bbox_inches="tight", dpi=200)
plt.close()
print(f"Generated 4-Panel Heatmap Figure: {out_fig2}")
