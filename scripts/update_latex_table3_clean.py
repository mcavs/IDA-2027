with open('latex/rashomon_pdp_aggregation.tex') as f:
    text = f.read()

# Locate longtable block in LaTeX
idx1 = text.find(r"\begin{longtable}{ccccccccccccc}")
idx_group = text.rfind(r"\begingroup", 0, idx1)
idx_end = text.find(r"\end{longtable}", idx1) + len(r"\end{longtable}")
idx_end_group = text.find(r"\endgroup", idx_end) + len(r"\endgroup")

assert idx_group != -1 and idx_end_group != -1, "Longtable block not found!"

with open('outputs/table3_clean_rows.tex') as f:
    table3_rows = f.read()

new_longtable = r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{1.2pt}
\begin{longtable}{ccccccccccccc}
\caption{Mean Deviation from Ground Truth ($\mathrm{MD}(\mathrm{GT})$) across Rashomon thresholds $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$ grouped into noise variance regimes: low ($\sigma^2 \in \{1,2,3\}$), mid ($\sigma^2 \in \{4,5,6\}$), and high ($\sigma^2 \in \{7,8,9\}$). Values report mean $\pm$ standard deviation across the three noise levels per regime ($n=1{,}500$, 160 models, 2880s budget per run). Bold denotes the lowest mean MD within each row.} \label{tab:md_combined} \\
\toprule
Setup & Noise & $\varepsilon$ & $\bar{K}$ & $\bar{\rho}$ & B1 & B0 & B2 & M1 & M2 & M3 & M4 & M5 \\
\midrule
\endfirsthead
\caption[]{Mean Deviation from Ground Truth ($\mathrm{MD}(\mathrm{GT})$) -- Continued from previous page} \\
\toprule
Setup & Noise & $\varepsilon$ & $\bar{K}$ & $\bar{\rho}$ & B1 & B0 & B2 & M1 & M2 & M3 & M4 & M5 \\
\midrule
\endhead
\midrule
\multicolumn{13}{r}{\footnotesize Continued on next page} \\
\endfoot
\bottomrule
\endlastfoot
""" + table3_rows + r"""
\end{longtable}
\endgroup"""

text = text[:idx_group] + new_longtable + text[idx_end_group:]

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Updated LaTeX with clean Table 3 successfully!")
