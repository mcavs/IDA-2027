with open('latex/rashomon_pdp_aggregation.tex') as f:
    text = f.read()

# Replace \usepackage{pdflscape} with \usepackage{lscape}
text = text.replace(r'\usepackage{pdflscape}', r'\usepackage{lscape}')

with open('latex/rashomon_pdp_aggregation.tex', 'w') as f:
    f.write(text)

print("Switched pdflscape to lscape!")
