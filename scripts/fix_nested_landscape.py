with open('latex/rashomon_pdp_aggregation.tex') as f:
    text = f.read()

# Replace double landscape with single landscape
text = text.replace(r"\begin{landscape}" + "\n" + r"\begin{landscape}", r"\begin{landscape}")
text = text.replace(r"\end{landscape}" + "\n" + r"\end{landscape}", r"\end{landscape}")

# Ensure pdflscape is used
text = text.replace(r'\usepackage{lscape}', r'\usepackage{pdflscape}')

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Fixed nested landscape cleanly!")
