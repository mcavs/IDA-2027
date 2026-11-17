import json, glob, pandas as pd

master5 = json.load(open('outputs/real_world_5datasets/real_world_5datasets_master.json'))
screen10 = {}
for f in sorted(glob.glob('outputs/real_world_screening/*.json')):
    d = json.load(open(f))
    screen10[d['dataset']] = d

titles = {
    'churn': 'Customer Churn',
    'wdbc': 'Breast Cancer (WDBC)',
    'german_credit': 'German Credit',
    'heart_statlog': 'Heart Statlog',
    'climate_crashes': 'Climate Model Crashes',
    'ilpd_liver': 'Indian Liver Patient (ILPD)',
    'wilt': 'Wilt Remote Sensing',
    'phoneme': 'Phoneme Recognition',
    'australian': 'Australian Credit',
    'diabetes': 'Pima Diabetes',
    'spambase': 'Spambase',
    'bank_marketing': 'Bank Marketing',
    'adult': 'Adult Census Income',
    'blood_transfusion': 'Blood Transfusion',
    'kc1': 'KC1 Software Defect'
}

all_data = {}
for k, v in master5.items():
    all_data[k] = {'n': v['n_rows'], 'K': v['K'], 'rho': v['rho'], 'features': v['features']}
for k, v in screen10.items():
    all_data[k] = {'n': v['n'], 'K': v['K'], 'rho': v['K'] / v['total_models'], 'features': v['features']}

latex = []
latex.append(r"\begin{table*}[htbp]")
latex.append(r"\centering")
latex.append(r"\small")
latex.append(r"\caption{Empirical profile ambiguity summary across fifteen real-world classification benchmarks ($\varepsilon=0.05$). For each benchmark, we report total observations ($n$), Rashomon set size ($K$), Rashomon ratio ($\rho$), continuous feature count, mean ambiguity ($\bar{A}$), and the features exhibiting minimal and maximal empirical ambiguity.}")
latex.append(r"\label{tab:real_world_ambiguity}")
latex.append(r"\vspace{2pt}")
latex.append(r"\begin{tabularx}{\linewidth}{l ccc c c cc}")
latex.append(r"\toprule")
latex.append(r"\textbf{Dataset} & $n$ & $K$ & $\rho$ & Features & Mean $\bar{A}$ & Min Ambiguity Feature ($\bar{A}_j$) & Max Ambiguity Feature ($\bar{A}_j$) \\")
latex.append(r"\midrule")

for k in titles:
    d = all_data[k]
    feats = d['features']
    ambs = [fdata['ambiguity'] for fdata in feats.values()]
    mean_a = sum(ambs) / len(ambs)
    min_f = min(feats.items(), key=lambda x: x[1]['ambiguity'])
    max_f = max(feats.items(), key=lambda x: x[1]['ambiguity'])
    min_name = min_f[0].replace('_', r'\_').replace('-', r'\textendash ')
    max_name = max_f[0].replace('_', r'\_').replace('-', r'\textendash ')
    latex.append(f"{titles[k]} & {d['n']:,} & {d['K']} & {d['rho']:.2f} & {len(feats)} & {mean_a:.5f} & {min_name} ({min_f[1]['ambiguity']:.6f}) & {max_name} ({max_f[1]['ambiguity']:.6f}) \\\\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabularx}")
latex.append(r"\end{table*}")

out_tex = "\n".join(latex)
with open("outputs/real_world_screening/table_15datasets.tex", "w") as f:
    f.write(out_tex)
print(out_tex)
