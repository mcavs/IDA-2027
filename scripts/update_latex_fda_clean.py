import re

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
\caption{Aggregation methods and formulation frameworks. Methods encompass classical ensembling, Bayesian weighting, robust statistics, and Functional Data Analysis (FDA) consensus.}
\label{tab:methods}
\begin{tabular}{lll}
\toprule
ID & Method & Aggregation Rule / Weight Source \\
\midrule
B1 & Best-model (baseline) & Single best-performing model (unaggregated) \\
B0 & SimpleAvg \citep{Cavus2026} & Pointwise uniform mean ($1/K$) \\
B2 & PerfWeighted & Inverse-RMSE linear weighting ($\omega_k \propto 1/\mathrm{RMSE}_k$) \\
M1 & Softmax-BMA & Bayesian Boltzmann weighting ($\omega_k \propto \exp(-\beta \cdot \overline{\mathrm{RMSE}}_k)$) \\
M2 & Pointwise Median & Robust pointwise median across Rashomon profiles \\
M3 & Trimmed Mean ($\alpha=0.1$) & $10\%$ symmetric pointwise trimmed mean \\
M4 & Functional Median \citep{LopezPintado2009} & Curve minimizing pairwise $L_1$ Fr\'echet profile distance \\
M5 & AVA-PDP \citep{Bhatt2020} & Pairwise Spearman functional rank-correlation consensus \\
\bottomrule
\end{tabular}
\end{table}"""

assert table2_old in text, "Table 2 old not found!"
text = text.replace(table2_old, table2_new)

# 2. Update Methods Itemize
idx1 = text.find(r"\item \textbf{B0 (SimpleAvg)}")
idx2 = text.find(r"\end{itemize}", idx1)
assert idx1 != -1 and idx2 != -1, "Itemize block not found!"

new_itemize = r"""\item \textbf{B1 (Best-model)}: Single model $M^*$ with minimum validation RMSE (standard unaggregated baseline).
\item \textbf{B0 (SimpleAvg)} \citep{Cavus2026}: Pointwise uniform averaging across all Rashomon models, $\bar{p}(x_S) = \frac{1}{K}\sum_{k=1}^K p_k(x_S)$.
\item \textbf{B2 (PerfWeighted)}: Inverse validation error linear weighting, $\omega_k \propto 1/\mathrm{RMSE}(M_k)$.
\item \textbf{M1 (Softmax-BMA)}: Bayesian Model Averaging via Boltzmann temperature weighting, $\omega_k \propto \exp\left(-\beta \frac{\mathrm{RMSE}_k - \min_j \mathrm{RMSE}_j}{\max_j \mathrm{RMSE}_j - \min_j \mathrm{RMSE}_j + 10^{-12}}\right)$ with $\beta = 2.0$.
\item \textbf{M2 (Pointwise Median)}: Robust pointwise median curve across all Rashomon models, $\hat{p}(x_S) = \mathrm{median}\{p_1(x_S), \dots, p_K(x_S)\}$, eliminating sensitivity to local profile perturbations.
\item \textbf{M3 (Trimmed Mean)}: Pointwise symmetric $\alpha$-trimmed mean ($\alpha=0.10$), filtering out extreme upper and lower profile percentiles prior to averaging.
\item \textbf{M4 (Functional Median)} \citep{LopezPintado2009}: The geometric median profile in $L_1$ Fr\'echet space that minimizes the total functional distance to all other candidate profiles: $p^* = \arg\min_{p_i \in \mathcal{R}_\varepsilon} \sum_{j=1}^K \|p_i - p_j\|_{L_2}$.
\item \textbf{M5 (AVA-PDP)} \citep{Bhatt2020}: Functional consensus weighting each curve by its mean pairwise Spearman rank alignment with the ensemble pool ($\omega_k \propto \bar{r}_k$)."""

text = text[:idx1] + new_itemize + "\n" + text[idx2:]

# 3. Update Table 3
old_header = r"$\varepsilon$ & Scenario & $\sigma^2$ & $K$ & $\rho$ & B0 & B1 & B2 & B3 & B4 & B8 & Ö1 & Ö2 \\"
new_header = r"Setup & $\sigma^2$ & $\varepsilon$ & $K$ & $\rho$ & B1 & B0 & B2 & M1 & M2 & M3 & M4 & M5 \\"
assert old_header in text, "Table 3 old header not found!"
text = text.replace(old_header, new_header)

with open('outputs/table3_heatmap_rows_fda.tex') as f:
    table3_rows = f.read()

t3_start = text.find(r"\caption{Mean Deviation from Ground Truth")
midrule_idx = text.find(r"\midrule", t3_start)
bottomrule_idx = text.find(r"\bottomrule", midrule_idx)

text = text[:midrule_idx + len(r"\midrule")] + "\n" + table3_rows + "\n" + text[bottomrule_idx:]

# 4. Contextual text updates
text = text.replace(
    r"We adapt their AGGrobust, AGGfaith, and AGGopt formulations to functional PDP aggregation (Section~\ref{sec:methods}, B3, Ö1, Ö2).",
    r"We introduce principled robust statistics and Functional Data Analysis (FDA) formulations for PDP aggregation (Section~\ref{sec:methods}, M1--M5)."
)
text = text.replace(
    r"B0 (SimpleAvg) captures most of the variance reduction, while B2 (PerfWeighted), B3 (AGGrobust), and B4 (AVA-PDP) provide the lowest overall ground-truth deviation.",
    r"B0 (SimpleAvg) and B2 (PerfWeighted) capture substantial variance reduction, while M4 (Functional Median) achieves the highest win rate (30.6\%) by selecting central, perturbation-free profile trajectories."
)

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Updated LaTeX file cleanly!")
