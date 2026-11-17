import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import studentized_range, rankdata, friedmanchisquare
import os

with open('outputs/master_analysis_results_classification_27scenarios.json') as f:
    data = json.load(f)

# Methods and clean publication labels
methods = [
    'B1:Best-model',
    'B0:SimpleAvg',
    'B2:PerfWeighted',
    'M1:SoftmaxBMA',
    'M2:PointMedian',
    'M3:TrimMean',
    'M4:FuncMedian',
    'M5:PairwiseAlign'
]

clean_names = {
    'B1:Best-model':    'Single Best (B1)',
    'B0:SimpleAvg':     'Simple Average (B0)',
    'B2:PerfWeighted':  'Performance-Weighted (B2)',
    'M1:SoftmaxBMA':    'Softmax BMA (M1)',
    'M2:PointMedian':   'Pointwise Median (M2)',
    'M3:TrimMean':      'Trimmed Mean (M3)',
    'M4:FuncMedian':    'Fréchet Median (M4)',
    'M5:PairwiseAlign': 'Pairwise Alignment (M5)'
}

# 36 Table 3 conditions (3 setups x 3 noise regimes x 4 epsilons)
regimes = {'low': [1,2,3], 'mid': [4,5,6], 'high': [7,8,9]}
tab3_mat = []
for snum in [1, 2, 3]:
    for reg, nvs in regimes.items():
        for eps in [0.01, 0.05, 0.10, 0.20]:
            eps_key = f'eps_{eps:.2f}'
            row = [np.mean([data[f'S{snum}_nv{nv}']['by_epsilon'][eps_key]['results'][m]['md'] for nv in nvs]) for m in methods]
            tab3_mat.append(row)

tab3_mat = np.array(tab3_mat) # shape: (36, 8)
N, k = tab3_mat.shape

# Compute Friedman test
stat, p_val = friedmanchisquare(*[tab3_mat[:, i] for i in range(k)])

# Compute average ranks (lower is better)
ranks = np.array([rankdata(tab3_mat[i, :]) for i in range(N)])
avg_ranks = np.mean(ranks, axis=0)

# Sort methods by average rank
sorted_idx = np.argsort(avg_ranks)
sorted_ranks = avg_ranks[sorted_idx]
sorted_names = [clean_names[methods[i]] for i in sorted_idx]

# Critical Distance calculation for Nemenyi test (alpha = 0.05)
alpha = 0.05
q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

print(f"Friedman chi2 = {stat:.2f}, p-value = {p_val:.4e}")
print(f"N = {N}, k = {k}, CD = {CD:.3f}")
for name, r in zip(sorted_names, sorted_ranks):
    print(f"  {name:25s}: {r:.2f}")

# Find cliques (groups of methods that are not significantly different, i.e. max rank - min rank <= CD)
cliques = []
for i in range(k):
    group = [i]
    for j in range(i + 1, k):
        if sorted_ranks[j] - sorted_ranks[i] <= CD:
            group.append(j)
    if len(group) > 1:
        # Check if this group is already a subset of an existing clique
        is_subset = False
        for c in cliques:
            if set(group).issubset(set(c)):
                is_subset = True
                break
        if not is_subset:
            cliques.append(group)

print(f"Cliques (indices in sorted list): {cliques}")

# ─── DRAW CRITICAL DISTANCE DIAGRAM ──────────────────────────────────────────
# Publication-quality Demsar-style Critical Distance diagram
fig, ax = plt.subplots(figsize=(12, 4.8), dpi=300)

# Axis limits: rank scale from 1 to k
lowv = 1
highv = k
margin = 0.5
ax.set_xlim(lowv - margin, highv + margin)
ax.set_ylim(-0.35, 1.25)
ax.axis('off')

# Main axis line
y_axis = 0.75
ax.plot([lowv, highv], [y_axis, y_axis], color='black', lw=2.0)

# Ticks
for r in range(lowv, highv + 1):
    ax.plot([r, r], [y_axis, y_axis + 0.04], color='black', lw=1.5)
    ax.text(r, y_axis + 0.08, str(r), ha='center', va='bottom', fontsize=15, fontweight='bold')

# Critical Distance Bar at top
cd_left = lowv
cd_right = lowv + CD
ax.plot([cd_left, cd_right], [y_axis + 0.35, y_axis + 0.35], color='#b2182b', lw=3.0)
ax.plot([cd_left, cd_left], [y_axis + 0.31, y_axis + 0.39], color='#b2182b', lw=2.0)
ax.plot([cd_right, cd_right], [y_axis + 0.31, y_axis + 0.39], color='#b2182b', lw=2.0)
ax.text((cd_left + cd_right) / 2, y_axis + 0.40, f"Critical Distance (CD = {CD:.2f})", 
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#b2182b')

# Rank scale label
ax.text((lowv + highv) / 2, y_axis + 0.22, "Average Rank (Lower is Better)", ha='center', va='bottom', fontsize=14, fontstyle='italic')

# Methods on Left (lower ranks / better) vs Right (higher ranks)
# Split methods: first half on left, second half on right
n_left = k // 2
left_idx = list(range(n_left))
right_idx = list(range(n_left, k))

# Left side items
y_step = 0.12
for rank_i, idx in enumerate(left_idx):
    r_val = sorted_ranks[idx]
    label = f"{sorted_names[idx]} ({r_val:.2f})"
    y_target = y_axis - 0.22 - (rank_i * y_step)
    
    # Marker on axis
    ax.plot(r_val, y_axis, marker='o', markersize=6, color='#1f77b4')
    # Connecting line: vertical down, then horizontal left
    ax.plot([r_val, r_val], [y_axis, y_target], color='#888888', lw=1.0, ls=':')
    ax.plot([r_val, lowv - 0.2], [y_target, y_target], color='#888888', lw=1.0, ls=':')
    ax.text(lowv - 0.25, y_target, label, ha='right', va='center', fontsize=13, fontweight='medium')

# Right side items
for rank_i, idx in enumerate(right_idx):
    r_val = sorted_ranks[idx]
    label = f"{sorted_names[idx]} ({r_val:.2f})"
    y_target = y_axis - 0.22 - ((k - 1 - idx) * y_step)
    
    # Marker on axis
    ax.plot(r_val, y_axis, marker='o', markersize=6, color='#1f77b4' if idx < k-1 else '#d95f02')
    # Connecting line: vertical down, then horizontal right
    ax.plot([r_val, r_val], [y_axis, y_target], color='#888888', lw=1.0, ls=':')
    ax.plot([r_val, highv + 0.2], [y_target, y_target], color='#888888', lw=1.0, ls=':')
    ax.text(highv + 0.25, y_target, label, ha='left', va='center', fontsize=13, fontweight='medium')

# Draw clique lines connecting non-significantly different methods
# Position clique bars below axis line but above text lines
clique_y_base = y_axis - 0.08
for c_i, c in enumerate(cliques):
    c_min_rank = sorted_ranks[c[0]]
    c_max_rank = sorted_ranks[c[-1]]
    bar_y = clique_y_base - (c_i * 0.05)
    ax.plot([c_min_rank, c_max_rank], [bar_y, bar_y], color='#2166ac', lw=4.5, solid_capstyle='round')

plt.tight_layout()

os.makedirs('latex/figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

out_pdf = 'figures/nemenyi_cd_diagram.pdf'
out_png = 'figures/nemenyi_cd_diagram.png'
plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
plt.savefig(out_png, bbox_inches='tight', dpi=300)
plt.savefig('latex/figures/nemenyi_cd_diagram.pdf', bbox_inches='tight', dpi=300)
plt.close()

print(f"CD Diagram successfully generated and saved to {out_pdf} and {out_png}")
