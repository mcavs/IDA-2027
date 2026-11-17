import re

with open('latex/rashomon_pdp_aggregation.tex') as f:
    text = f.read()

# Update Table 3 caption and rows
with open('outputs/table3_heatmap_rows_fda_27scenarios.tex') as f:
    table3_rows = f.read()

# Replace caption
old_caption_sub = r"across all four Rashomon thresholds $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$ under the updated aggregation methods ($n=1{,}500$, 160 models, 2880s budget)"
new_caption_sub = r"across all four Rashomon thresholds $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$ and all nine noise variance levels $\sigma^2 \in \{1,\dots,9\}$ ($n=1{,}500$, 160 models, 2880s budget, 27 scenarios, 108 evaluations)"

if old_caption_sub in text:
    text = text.replace(old_caption_sub, new_caption_sub)

# Replace rows in Table 3
t3_start = text.find(r"\caption{Mean Deviation from Ground Truth")
midrule_idx = text.find(r"\midrule", t3_start)
bottomrule_idx = text.find(r"\bottomrule", midrule_idx)

text = text[:midrule_idx + len(r"\midrule")] + "\n" + table3_rows + "\n" + text[bottomrule_idx:]

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Updated LaTeX with 27-scenario Table 3 successfully!")
