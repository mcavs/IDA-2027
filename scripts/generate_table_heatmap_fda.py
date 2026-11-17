import json
import numpy as np

with open('outputs/master_analysis_results_fda_27scenarios.json') as f:
    master = json.load(f)

method_keys = [
    'B1:Best-model', 'B0:SimpleAvg', 'B2:PerfWeighted',
    'M1:SoftmaxBMA', 'M2:PointMedian', 'M3:TrimMean',
    'M4:FuncMedian', 'M5:AVA-PDP'
]

all_mds = []
for scen_id, scen_data in master.items():
    for eps_key, eps_data in scen_data['by_epsilon'].items():
        res = eps_data['results']
        for m in method_keys:
            all_mds.append(res[m]['md'])

min_md = min(all_mds)
max_md = max(all_mds)

def get_color_macro(val, min_v, max_v):
    norm = (val - min_v) / (max_v - min_v + 1e-12)
    pct = int(round(norm * 100))
    pct = max(0, min(100, pct))
    return f"\\cellcolor{{heatcolor!{pct}!white}}"

latex_rows = []
setups = [1, 2, 3]
all_nvs = list(range(1, 10))
eps_list = ['eps_0.01', 'eps_0.05', 'eps_0.10', 'eps_0.20']

for snum in setups:
    for nv in all_nvs:
        scen_id = f"S{snum}_nv{nv}"
        scen_data = master[scen_id]
        setup_str = f"Setup {snum}"
        sig_str = str(nv)

        first_scen_row = True
        for eps_key in eps_list:
            eps_data = scen_data['by_epsilon'][eps_key]
            eps_val = eps_key.replace('eps_', '')
            k_val = eps_data['n_rash']
            rho_val = eps_data['rho']
            res = eps_data['results']

            row_mds = {m: res[m]['md'] for m in method_keys}
            best_m = min(row_mds, key=row_mds.get)

            cell_strs = []
            for m in method_keys:
                val = row_mds[m]
                col_macro = get_color_macro(val, min_md, max_md)
                val_str = f"{val:.3f}"
                if m == best_m:
                    formatted_val = f"\\textbf{{{val_str}}}"
                else:
                    formatted_val = val_str
                cell_strs.append(f"{col_macro}{formatted_val}")

            row_line = f"{setup_str if (nv==1 and eps_key == 'eps_0.01') else ''} & {sig_str if eps_key == 'eps_0.01' else ''} & {eps_val} & {k_val} & {rho_val:.3f} & " + " & ".join(cell_strs) + " \\\\"
            latex_rows.append(row_line)
        latex_rows.append("\\midrule")

if latex_rows[-1] == "\\midrule":
    latex_rows.pop()

output_tex = "\n".join(latex_rows)
with open('outputs/table3_heatmap_rows_fda_27scenarios.tex', 'w') as f:
    f.write(output_tex)

print("Generated Table 3 (27 scenarios, 108 rows) heatmap rows successfully!")
