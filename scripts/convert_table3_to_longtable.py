with open('latex/rashomon_pdp_aggregation.tex') as f:
    text = f.read()

# Add \usepackage{longtable} to preamble
if r'\usepackage{longtable}' not in text:
    text = text.replace(r'\usepackage{booktabs}', "\\usepackage{booktabs}\n\\usepackage{longtable}")

# Locate Table 3 block
t3_start = text.find(r"\begin{table}[htbp]" + "\n" + r"\centering\small" + "\n" + r"\caption{Mean Deviation from Ground Truth")
if t3_start == -1:
    t3_start = text.find(r"\begin{table}[htbp]")
    # find where caption is
    while t3_start != -1:
        if "Mean Deviation from Ground Truth" in text[t3_start:t3_start+500]:
            break
        t3_start = text.find(r"\begin{table}[htbp]", t3_start+1)

assert t3_start != -1, "Table 3 start not found!"
t3_end = text.find(r"\end{table}", t3_start) + len(r"\end{table}")

with open('outputs/table3_heatmap_rows_fda_27scenarios.tex') as f:
    table3_rows = f.read()

longtable_block = r"""\begingroup
\small
\setlength{\tabcolsep}{3.6pt}
\begin{longtable}{ccccccccccccc}
\caption{Mean Deviation from Ground Truth ($\mathrm{MD}(\mathrm{GT})$) across all four Rashomon thresholds $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$ and all nine noise variance levels $\sigma^2 \in \{1,\dots,9\}$ ($n=1{,}500$, 160 models, 2880s budget, 27 scenarios, 108 evaluations). Bold = lowest MD per scenario row; color gradient reflects relative deviation from ground truth.} \label{tab:md_combined} \\
\toprule
Setup & $\sigma^2$ & $\varepsilon$ & $K$ & $\rho$ & B1 & B0 & B2 & M1 & M2 & M3 & M4 & M5 \\
\midrule
\endfirsthead
\caption[]{Mean Deviation from Ground Truth ($\mathrm{MD}(\mathrm{GT})$) -- Continued from previous page} \\
\toprule
Setup & $\sigma^2$ & $\varepsilon$ & $K$ & $\rho$ & B1 & B0 & B2 & M1 & M2 & M3 & M4 & M5 \\
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

text = text[:t3_start] + longtable_block + text[t3_end:]

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Converted Table 3 to longtable cleanly!")
