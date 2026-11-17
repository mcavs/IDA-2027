import json
import numpy as np

with open('outputs/master_analysis_results_classification_27scenarios.json') as f:
    data = json.load(f)

regimes = [
    ('low', [1, 2, 3]),
    ('mid', [4, 5, 6]),
    ('high', [7, 8, 9])
]

methods = ['B1:Best-model', 'B0:SimpleAvg', 'B2:PerfWeighted', 'M1:SoftmaxBMA', 'M2:PointMedian', 'M3:TrimMean', 'M4:FuncMedian', 'M5:PairwiseAlign']
eps_list = [0.01, 0.05, 0.10, 0.20]

print("=== TABLE 3: MEAN DEVIATION LATEX ROWS ===")
for snum in [1, 2, 3]:
    s_label = f"({snum})"
    for r_idx, (reg_name, nvs) in enumerate(regimes):
        for eps in eps_list:
            eps_key = f"eps_{eps:.2f}"
            
            # compute mean K and mean rho
            ks = [data[f"S{snum}_nv{nv}"]["by_epsilon"][eps_key]["n_rash"] for nv in nvs]
            rhos = [data[f"S{snum}_nv{nv}"]["by_epsilon"][eps_key]["rho"] for nv in nvs]
            mean_k = int(round(np.mean(ks)))
            mean_rho = np.mean(rhos)
            
            # compute method mds
            mds = {}
            for m in methods:
                mds[m] = np.mean([data[f"S{snum}_nv{nv}"]["by_epsilon"][eps_key]["results"][m]["md"] for nv in nvs])
            
            # find min md among methods
            min_m = min(methods, key=lambda m: mds[m])
            
            # format columns
            setup_col = s_label if (r_idx == 0 and eps == 0.01) else "{}"
            noise_col = reg_name if eps == 0.01 else ""
            
            vals_str = []
            for m in methods:
                val = mds[m]
                if m == min_m:
                    vals_str.append(f"\\textbf{{{val:.3f}}}")
                else:
                    vals_str.append(f"{val:.3f}")
            
            row_str = f"{setup_col} & {noise_col} & {eps:.2f} & {mean_k} & {mean_rho:.2f} & " + " & ".join(vals_str) + " \\\\"
            print(row_str)
        if r_idx < 2:
            print("\\cmidrule{2-13}")
    if snum < 3:
        print("\\midrule")

print("\n=== TABLE 4: COVERAGE AND WIDTH LATEX ROWS ===")
# Table 4: for each eps, list methods and their CR and MWCI
methods_t4 = ['B0:SimpleAvg', 'B2:PerfWeighted', 'M1:SoftmaxBMA', 'M2:PointMedian', 'M3:TrimMean', 'M4:FuncMedian', 'M5:PairwiseAlign']
t4_labels = {
    'B0:SimpleAvg': 'B0 (Simple Average)',
    'B2:PerfWeighted': 'B2 (Performance-Weighted)',
    'M1:SoftmaxBMA': 'M1 (Softmax BMA)',
    'M2:PointMedian': 'M2 (Pointwise Median)',
    'M3:TrimMean': 'M3 (Trimmed Mean)',
    'M4:FuncMedian': 'M4 (Functional Median)',
    'M5:PairwiseAlign': 'M5 (Pairwise Alignment)'
}

for eps in eps_list:
    eps_key = f"eps_{eps:.2f}"
    print(f"% Epsilon = {eps:.2f}")
    cr_dict = {}
    mwci_dict = {}
    for m in methods_t4:
        cr_dict[m] = np.mean([data[scen]["by_epsilon"][eps_key]["results"][m]["cr1"] for scen in data])
        mwci_dict[m] = np.mean([data[scen]["by_epsilon"][eps_key]["results"][m]["mwci1"] for scen in data])
    
    max_cr = max(cr_dict.values())
    for m in methods_t4:
        cr_val = cr_dict[m]
        mwci_val = mwci_dict[m]
        cr_str = f"\\textbf{{{cr_val:.3f}}}" if cr_val == max_cr else f"{cr_val:.3f}"
        print(f"\\varepsilon = {eps:.2f} & {t4_labels[m]} & {cr_str} & {mwci_val:.4f} \\\\")
    print("\\midrule")
