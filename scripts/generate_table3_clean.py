import json
import numpy as np

with open('outputs/master_analysis_results_fda_27scenarios.json') as f:
    master = json.load(f)

method_keys = [
    'B1:Best-model', 'B0:SimpleAvg', 'B2:PerfWeighted',
    'M1:SoftmaxBMA', 'M2:PointMedian', 'M3:TrimMean',
    'M4:FuncMedian', 'M5:AVA-PDP'
]

noise_groups = [
    ('low', [1, 2, 3]),
    ('mid', [4, 5, 6]),
    ('high', [7, 8, 9])
]

eps_list = ['eps_0.01', 'eps_0.05', 'eps_0.10', 'eps_0.20']
setups = [1, 2, 3]

grouped_data = []

for s in setups:
    for ng_name, nvs in noise_groups:
        for eps_key in eps_list:
            eps_val = eps_key.replace('eps_', '')
            k_vals = []
            rho_vals = []
            m_vals = {m: [] for m in method_keys}
            for nv in nvs:
                scen_id = f'S{s}_nv{nv}'
                eps_data = master[scen_id]['by_epsilon'][eps_key]
                k_vals.append(eps_data['n_rash'])
                rho_vals.append(eps_data['rho'])
                for m in method_keys:
                    m_vals[m].append(eps_data['results'][m]['md'])

            k_mean = np.mean(k_vals)
            rho_mean = np.mean(rho_vals)
            means = {m: np.mean(m_vals[m]) for m in method_keys}
            stds = {m: np.std(m_vals[m]) for m in method_keys}

            grouped_data.append({
                'setup': s,
                'noise_group': ng_name,
                'eps': eps_val,
                'k_mean': k_mean,
                'rho_mean': rho_mean,
                'means': means,
                'stds': stds
            })

latex_rows = []
prev_setup = None

for row in grouped_data:
    s = row['setup']
    ng = row['noise_group']
    eps = row['eps']
    k_str = f"{row['k_mean']:.1f}"
    rho_str = f"{row['rho_mean']:.3f}"
    
    best_m = min(method_keys, key=lambda m: row['means'][m])
    
    if s != prev_setup:
        setup_col = f"({s})"
        prev_setup = s
    else:
        setup_col = ""

    noise_col = ng if eps == '0.01' else ""

    cell_strs = []
    for m in method_keys:
        mean_v = row['means'][m]
        std_v = row['stds'][m]
        val_str = f"{mean_v:.3f} $\\pm$ {std_v:.3f}"
        if m == best_m:
            formatted_val = f"\\textbf{{{mean_v:.3f}}} $\\pm$ {std_v:.3f}"
        else:
            formatted_val = val_str
        cell_strs.append(formatted_val)

    line = f"{setup_col} & {noise_col} & {eps} & {k_str} & {rho_str} & " + " & ".join(cell_strs) + " \\\\"
    latex_rows.append(line)
    
    if eps == '0.20':
        latex_rows.append("\\midrule")

if latex_rows[-1] == "\\midrule":
    latex_rows.pop()

output_tex = "\n".join(latex_rows)
with open('outputs/table3_clean_rows.tex', 'w') as f:
    f.write(output_tex)

print("Generated valid math mode Table 3 rows!")
