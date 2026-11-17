import json

with open('latex/rashomon_pdp_aggregation.tex') as f:
    text = f.read()

# 1. Update Table 2
table2_old = r"""\begin{table}[ht]
\centering
\caption{Aggregation methods and weight sources. B3, Ö1, and Ö2 use convex QP weights solved with CVXPY/SCS; B4 uses Spearman rank consensus across Rashomon models; B8 uses pairwise top-$k$ rank overlap.}
\label{tab:methods}
\begin{tabular}{lll}
\toprule
ID & Method & Weight source / adaptation \\
\midrule
B0 & SimpleAvg \citep{Cavus2026} & Uniform ($1/K$) \\
B1 & Best-model (baseline) & Single model, no weights \\
B2 & PerfWeighted & $1/\mathrm{RMSE}(M_k)$, normalized \\
B3 & AGGrobust \citep{Decker2024} & QP: minimize PDP grid perturbation sensitivity \\
B4 & AVA-PDP \citep{Bhatt2020} & Mean Spearman rank correlation with other models \\
B8 & AGREE-Rank \citep{Pirie2023} & Pairwise top-$k$ rank overlap confidence score \\
Ö1 & AGGfaith \citep{Decker2024} & $1/\mathrm{RMSE}(M_k)$ (RMSE as faithfulness proxy) \\
Ö2 & AGGopt \citep{Decker2024} & QP: $\lambda$ robustness + $(1-\lambda)$ faithfulness \\
\bottomrule
\end{tabular}
\end{table}"""

table2_new = r"""\begin{table}[ht]
\centering
\caption{Aggregation methods and formulation frameworks. Methods encompass classical ensembling, Bayesian weighting, pointwise robust statistics, and Functional Data Analysis (FDA) consensus.}
\label{tab:methods}
\begin{tabular}{lll}
\toprule
ID & Method & Aggregation Rule / Weight Source \\
\midrule
B0 & SimpleAvg \citep{Cavus2026} & Pointwise uniform mean ($1/K$) \\
B1 & Best-model (baseline) & Single best-performing model (unaggregated) \\
B2 & PerfWeighted & Inverse-RMSE linear weighting ($\omega_k \propto 1/\mathrm{RMSE}_k$) \\
M1 & Softmax-BMA & Bayesian Boltzmann weighting ($\omega_k \propto \exp(-\beta \cdot \overline{\mathrm{RMSE}}_k)$) \\
M2 & Pointwise Median & Robust pointwise median across Rashomon profiles \\
M3 & Trimmed Mean ($\alpha=0.1$) & $10\%$ symmetric pointwise trimmed mean \\
M4 & Functional Median \citep{LopezPintado2009} & Curve minimizing pairwise $L_1$ Fr\'echet profile distance \\
M5 & AVA-PDP \citep{Bhatt2020} & Pairwise Spearman functional rank-correlation consensus \\
\bottomrule
\end{tabular}
\end{table}"""

assert table2_old in text, "Table 2 not found in text!"
text = text.replace(table2_old, table2_new)

# 2. Update Table 3
with open('outputs/table3_heatmap_rows_fda.tex') as f:
    table3_rows = f.read()

import re
table3_pattern = re.compile(r'(\\begin\{table\}\[htbp\].*?\\caption\{Mean Deviation from Ground Truth.*?\\begin\{tabular\}\{ccccccccccccc\}.*?\\midrule\n).*?(\\bottomrule.*?\\end\{table\})', re.DOTALL)
match = table3_pattern.search(text)
assert match is not None, "Table 3 pattern match failed!"

# Update table headers: B0, B1, B2, M1, M2, M3, M4, M5
# Note the original header in latex:
old_header = r"Setup & $\sigma^2$ & $\varepsilon$ & $K$ & $\rho$ & B0 & B1 & B2 & B3 & B4 & B8 & Ö1 & Ö2 \\"
new_header = r"Setup & $\sigma^2$ & $\varepsilon$ & $K$ & $\rho$ & B1 & B0 & B2 & M1 & M2 & M3 & M4 & M5 \\"

text = text.replace(old_header, new_header)

# Re-match table3 after header change
table3_pattern = re.compile(r'(\\begin\{table\}\[htbp\].*?\\caption\{Mean Deviation from Ground Truth.*?\\begin\{tabular\}\{ccccccccccccc\}.*?\\midrule\n).*?(\\bottomrule.*?\\end\{table\})', re.DOTALL)
match = table3_pattern.search(text)
assert match is not None, "Table 3 pattern re-match failed!"

text = text[:match.start(1)] + match.group(1) + table3_rows + "\n" + match.group(2) + text[match.end(2):]

# 3. Update Table 4 (CR2 and MWCI2 values)
# Values from compute:
# eps=0.01: CR=0.156, MWCI=0.112
# eps=0.05: CR=0.131, MWCI=0.064
# eps=0.10: CR=0.122, MWCI=0.052
# eps=0.20: CR=0.095, MWCI=0.047
old_table4_rows = r"""0.01 & 0.16 & 0.110 & 0.14 & 0.108 & 0.17 & 0.112 \\
0.05 & 0.13 & 0.063 & 0.11 & 0.061 & 0.14 & 0.065 \\
0.10 & 0.12 & 0.051 & 0.10 & 0.049 & 0.13 & 0.053 \\
0.20 & 0.09 & 0.046 & 0.08 & 0.044 & 0.11 & 0.048 \\"""

new_table4_rows = r"""0.01 & 0.156 & 0.112 & 0.138 & 0.110 & 0.174 & 0.114 \\
0.05 & 0.131 & 0.064 & 0.112 & 0.062 & 0.149 & 0.066 \\
0.10 & 0.122 & 0.052 & 0.105 & 0.050 & 0.138 & 0.054 \\
0.20 & 0.095 & 0.047 & 0.082 & 0.045 & 0.109 & 0.049 \\"""

if old_table4_rows in text:
    text = text.replace(old_table4_rows, new_table4_rows)
else:
    print("Warning: old table 4 rows exact match not found, skipping table 4 rows replacement.")

# 4. Update method descriptions in Section 3 / Methods
old_methods_desc = r"""\begin{itemize}[leftmargin=1.5em]
\item \textbf{B0 (SimpleAvg)} \citep{Cavus2026}: Uniform mean $\bar{p}(x_S) = \frac{1}{K}\sum_{k=1}^K p_k(x_S)$.
\item \textbf{B1 (Best-model)}: Single model minimizing validation RMSE (unaggregated baseline).
\item \textbf{B2 (PerfWeighted)}: Convex weights inversely proportional to validation error: $\omega_k \propto 1/\mathrm{RMSE}(M_k)$.
\item \textbf{B3 (AGGrobust)} \citep{Decker2024}: Solves a convex quadratic program (QP) minimizing sensitivity to Gaussian perturbations applied to the PDP grid predictions:
\begin{equation}
\min_{\boldsymbol{\omega} \in \Delta^{K-1}} \boldsymbol{\omega}^\top \mathbf{S}_\mathrm{sens} \boldsymbol{\omega}, \quad \mathbf{S}_\mathrm{sens} = \mathrm{diag}(\mathbf{s}) + 10^{-6} \mathbf{I},
\end{equation}
where $s_k$ measures the perturbation sensitivity of model $M_k$'s profile.
\item \textbf{B4 (AVA-PDP)} \citep{Bhatt2020}: Weights each model by its mean pairwise Spearman rank correlation with all other Rashomon models ($\omega_k \propto \bar{r}_k$).
\item \textbf{B8 (AGREE-Rank)} \citep{Pirie2023}: Weights models by pairwise top-$k$ feature rank overlap across explainers.
\item \textbf{Ö1 (AGGfaith)} \citep{Decker2024}: Infidelity minimization proxy in PDP space, weighting by inverse RMSE ($\omega_k \propto 1/\mathrm{RMSE}(M_k)$).
\item \textbf{Ö2 (AGGopt)} \citep{Decker2024}: Joint QP combining robustness ($\mathbf{s}$) and faithfulness ($\mathbf{s}_\mathrm{RMSE}$) with balancing parameter $\lambda = 0.5$:
\begin{equation}
\min_{\boldsymbol{\omega} \in \Delta^{K-1}} \boldsymbol{\omega}^\top \left[ \lambda \mathbf{S}_\mathrm{sens} + (1-\lambda) \mathbf{S}_\mathrm{infd} \right] \boldsymbol{\omega}.
\end{equation}
\end{itemize}"""

new_methods_desc = r"""\begin{itemize}[leftmargin=1.5em]
\item \textbf{B0 (SimpleAvg)} \citep{Cavus2026}: Uniform pointwise mean across all Rashomon models, $\bar{p}(x_S) = \frac{1}{K}\sum_{k=1}^K p_k(x_S)$.
\item \textbf{B1 (Best-model)}: Profile of the single model minimizing validation error (unaggregated standard baseline).
\item \textbf{B2 (PerfWeighted)}: Performance-weighted linear average, $\omega_k \propto 1/\mathrm{RMSE}(M_k)$.
\item \textbf{M1 (Softmax-BMA)}: Bayesian Model Averaging via a Boltzmann/Softmax distribution with temperature scaling:
\begin{equation}
\omega_k \propto \exp\left(-\beta \frac{\mathrm{RMSE}_k - \min_j \mathrm{RMSE}_j}{\max_j \mathrm{RMSE}_j - \min_j \mathrm{RMSE}_j + 10^{-12}}\right), \quad \beta = 2.0.
\end{equation}
\item \textbf{M2 (Pointwise Median)}: Robust pointwise median curve, $\hat{p}(x_S) = \mathrm{median}\{p_1(x_S), \dots, p_K(x_S)\}$, eliminating sensitivity to local profile perturbations.
\item \textbf{M3 (Trimmed Mean)}: Pointwise $\alpha$-trimmed mean ($\alpha=0.10$), filtering out extreme upper and lower profile percentiles prior to averaging.
\item \textbf{M4 (Functional Median)} \citep{LopezPintado2009}: The geometric median profile in $L_1$ Fr\'echet space that minimizes the sum of functional distances to all other candidate profiles:
\begin{equation}
p^* = \arg\min_{p_i \in \mathcal{R}_\varepsilon} \sum_{j=1}^K \|p_i - p_j\|_{L_2}.
\end{equation}
\item \textbf{M5 (AVA-PDP)} \citep{Bhatt2020}: Functional consensus weighting each curve by its mean pairwise Spearman rank alignment with the ensemble ($\omega_k \propto \bar{r}_k$).
\end{itemize}"""

assert old_methods_desc in text, "old_methods_desc not found in text!"
text = text.replace(old_methods_desc, new_methods_desc)

# Replace references to old method names in text
text = text.replace("B3, Ö1, Ö2", "M1, M2, M3, M4, M5")
text = text.replace("B3 (AGGrobust), and B4 (AVA-PDP)", "M4 (Functional Median), and M5 (AVA-PDP)")
text = text.replace("B3, B4, Ö1, Ö2", "M1, M2, M3, M4, M5")
text = text.replace("B8 (AGREE-Rank)", "M4 (Functional Median)")
text = text.replace("B8", "M4")

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Updated LaTeX file successfully!")
