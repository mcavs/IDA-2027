import json
import numpy as np

with open('outputs/master_analysis_results_classification_27scenarios.json') as f:
    data = json.load(f)

methods_t4 = ['B0:SimpleAvg', 'B2:PerfWeighted', 'M1:SoftmaxBMA', 'M2:PointMedian', 'M3:TrimMean', 'M4:FuncMedian', 'M5:PairwiseAlign']
eps_list = [0.01, 0.05, 0.10, 0.20]

for eps in eps_list:
    eps_key = f"eps_{eps:.2f}"
    cr_dict = {}
    mwci_dict = {}
    for m in methods_t4:
        cr_dict[m] = np.mean([data[scen]["by_epsilon"][eps_key]["results"][m]["cr1"] for scen in data])
        mwci_dict[m] = np.mean([data[scen]["by_epsilon"][eps_key]["results"][m]["mwci1"] for scen in data])
    
    max_cr = max(cr_dict.values())
    
    cr_cells = []
    mwci_cells = []
    for m in methods_t4:
        cr_val = cr_dict[m]
        mwci_val = mwci_dict[m]
        if cr_val == max_cr:
            cr_cells.append(f"\\textbf{{{cr_val:.3f}}}")
        else:
            cr_cells.append(f"{cr_val:.3f}")
        mwci_cells.append(f"{mwci_val:.3f}")
    
    print(f"\\multirow{{2}}{{*}}{{{eps:.2f}}} & $\\overline{{\\mathrm{{CR}}}}$ & " + " & ".join(cr_cells) + " \\\\")
    print(f" & $\\overline{{\\mathrm{{MWCI}}}}$ & " + " & ".join(mwci_cells) + " \\\\")
    if eps < 0.20:
        print("\\cmidrule{1-9}")
