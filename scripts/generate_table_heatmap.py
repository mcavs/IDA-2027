import json
import matplotlib.colors as mcolors

with open('outputs/master_analysis_results_2880s_160m.json') as f:
    d = json.load(f)

# Gradient from Soft Mint Green (#A5D6A7) to Soft Cream (#FFFFF0) to Soft Coral/Rose (#EF9A9A)
cmap = mcolors.LinearSegmentedColormap.from_list('err_cmap', ['#A5D6A7', '#FFFFF0', '#EF9A9A'])
vmin, vmax = 0.075, 0.355

def get_hex(val):
    norm = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    rgba = cmap(norm)
    return f"{int(rgba[0]*255):02X}{int(rgba[1]*255):02X}{int(rgba[2]*255):02X}"

methods = ['B0:SimpleAvg', 'B1:Best-model', 'B2:PerfWeighted', 'B3:AGGrobust', 'B4:AVA-PDP', 'B8:AGREE-Rank', 'Ö1:AGGfaith', 'Ö2:AGGopt']

lines = []
eps_map = {'eps_0.01': '0.01', 'eps_0.05': '0.05', 'eps_0.1': '0.10', 'eps_0.2': '0.20'}
for eps in ['eps_0.01', 'eps_0.05', 'eps_0.1', 'eps_0.2']:
    eps_val = eps_map[eps]
    for s_idx, s in enumerate([1, 2, 3]):
        sc_name = f"Setup {s}"
        for nv_idx, nv in enumerate([1, 4, 9]):
            sc_key = f"S{s}_nv{nv}"
            res = d[sc_key]['by_epsilon'][eps]
            k_val = res['n_rash']
            rho_val = f"{res['rho']:.2f}"
            
            cell_strs = []
            vals = [res['results'][m]['md'] for m in methods]
            min_val = min(vals)
            
            for m in methods:
                v = res['results'][m]['md']
                hx = get_hex(v)
                v_str = f"{v:.3f}"
                if abs(v - min_val) < 1e-4:
                    cell = r"\cellcolor[HTML]{" + hx + r"}\textbf{" + v_str + "}"
                elif m != 'B1:Best-model' and abs(min_val - res['results']['B1:Best-model']['md']) < 1e-4:
                    agg_vals = [res['results'][m2]['md'] for m2 in methods if m2 != 'B1:Best-model']
                    if abs(v - min(agg_vals)) < 1e-4:
                        cell = r"\cellcolor[HTML]{" + hx + r"}\underline{" + v_str + "}"
                    else:
                        cell = r"\cellcolor[HTML]{" + hx + "}" + v_str
                else:
                    cell = r"\cellcolor[HTML]{" + hx + "}" + v_str
                cell_strs.append(cell)
            
            e_col = eps_val if (s_idx == 0 and nv_idx == 0) else "    "
            s_col = sc_name if nv_idx == 0 else "       "
            row_str = f"{e_col} & {s_col} & {nv} & {k_val} & {rho_val} & " + " & ".join(cell_strs) + r" \\"
            lines.append(row_str)
        if s_idx < 2:
            lines.append(r"\cmidrule(lr){2-13}")
    if eps != 'eps_0.2':
        lines.append(r"\midrule")

with open("outputs/table3_heatmap_rows.tex", "w") as out:
    out.write("\n".join(lines) + "\n")
print("Done! Lines written:", len(lines))
