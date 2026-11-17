with open('latex/rashomon_pdp_aggregation.tex') as f:
    text = f.read()

# 1. Add \usepackage{pdflscape} to preamble if not present
if r'\usepackage{pdflscape}' not in text:
    text = text.replace(r'\usepackage{longtable}', "\\usepackage{longtable}\n\\usepackage{pdflscape}")

# 2. Locate the longtable block and wrap it in \begin{landscape} ... \end{landscape}
idx1 = text.find(r"\begin{longtable}{ccccccccccccc}")
idx_group = text.rfind(r"\begingroup", 0, idx1)
idx_end = text.find(r"\end{longtable}", idx1) + len(r"\end{longtable}")
idx_end_group = text.find(r"\endgroup", idx_end) + len(r"\endgroup")

assert idx_group != -1 and idx_end_group != -1, "Longtable block not found!"

# Also restore comfortable font size and column padding since landscape has plenty of width!
# \small and \setlength{\tabcolsep}{3.5pt} fits comfortably in landscape textwidth (~24.7cm)
with open('outputs/table3_clean_rows.tex') as f:
    table3_rows = f.read()

new_landscape_longtable = r"""\begin{landscape}
\begingroup
\small
\setlength{\tabcolsep}{3.2pt}
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
\endgroup
\end{landscape}"""

text = text[:idx_group] + new_landscape_longtable + text[idx_end_group:]

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Applied pdflscape to Table 3 successfully!")
