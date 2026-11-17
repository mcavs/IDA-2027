"""
Kaydedilmis PDP nesneleri uzerinden sabit B1 referansiyla tum analizleri yeniden hesaplama.
AutoML'i sifirdan calistirmaya gerek YOKTUR (outputs/pdps/ altindaki pickle'lar kullanilir).
gt = gt_raw - gt_raw.mean() + best_pdp.mean()
Bu sayede:
1) B1: Best-model tum epsilon tablolarinda TAMAMEN AYNI kalir.
2) Agregasyon yontemleri de ayni sabit ground-truth ekseninde degerlendirilir.
"""
import os, json, pickle, time
import numpy as np
from analysis import (
    METHOD_NAMES, AGG_FNS, compute_fixed_weights, bootstrap_ci,
    coverage_rate, mwci, OUT
)

PDP_DIR = os.path.join(OUT, "pdps")
EPS_LIST = [0.01, 0.05, 0.10, 0.20]
scenarios = [f"S{s}_nv{nv}" for s in [1, 2, 3] for nv in [1, 4, 9]]

recomputed_results = {}

for sc in scenarios:
    pdp_path = os.path.join(PDP_DIR, f"{sc}_pdp_data.pkl")
    print(f"Islem yapiliyor: {sc} ({pdp_path})")
    with open(pdp_path, "rb") as f:
        data = pickle.load(f)
        
    vals = np.array(data["vals"])
    best_pdp = np.array(data["best_pdp"])
    pdp_all = np.array(data["pdp_all"])
    all_rmses = data["all_rmses"]
    model_ids = data["model_ids"]
    gt_raw = np.array(data["gt_raw"])
    
    # SABIT GROUND TRUTH (B1 referansiyla merkezlenmis)
    gt = gt_raw - gt_raw.mean() + best_pdp.mean()
    b1_md = float(np.mean(np.abs(best_pdp - gt)))
    
    best_rmse = all_rmses[0]
    
    sc_res = {
        "scenario": sc,
        "b1_md_fixed": b1_md,
        "by_epsilon": {}
    }
    
    for eps in EPS_LIST:
        rash_idx = [i for i, r in enumerate(all_rmses) if r <= best_rmse * (1 + eps)]
        k = len(rash_idx)
        rho = k / len(all_rmses)
        
        if k < 2:
            res_eps = {
                "n_rash": k,
                "rho": rho,
                "results": {
                    "B1:Best-model": {"md": b1_md, "cr1": None, "cr2": None, "mwci1": None, "mwci2": None}
                }
            }
            for m in AGG_FNS:
                res_eps["results"][m] = {"md": b1_md, "cr1": 0.0, "cr2": 0.0, "mwci1": 0.0, "mwci2": 0.0}
            sc_res["by_epsilon"][f"eps_{eps}"] = res_eps
            continue
            
        P = pdp_all[rash_idx]
        rmses = [all_rmses[i] for i in rash_idx]
        ref = P.mean(0)
        
        res_eps = {
            "n_rash": k,
            "rho": rho,
            "results": {
                "B1:Best-model": {
                    "md": b1_md,
                    "cr1": None, "cr2": None, "mwci1": None, "mwci2": None
                }
            }
        }
        
        for method, fn in AGG_FNS.items():
            agg = fn(P=P, rmses=rmses, ref=ref)
            fw = compute_fixed_weights(method, P, rmses, ref)
            (lo1, hi1), (lo2, hi2) = bootstrap_ci(method, P, rmses, ref, fw)
            res_eps["results"][method] = {
                "md": float(np.mean(np.abs(agg - gt))),
                "cr1": coverage_rate(gt, lo1, hi1),
                "cr2": coverage_rate(gt, lo2, hi2),
                "mwci1": mwci(lo1, hi1),
                "mwci2": mwci(lo2, hi2)
            }
        sc_res["by_epsilon"][f"eps_{eps}"] = res_eps
        
    recomputed_results[sc] = sc_res
    print(f"  ✓ {sc} bitti (B1={b1_md:.4f})")

OUT_FILE = os.path.join(OUT, "master_analysis_results_fixed_gt.json")
with open(OUT_FILE, "w") as f:
    json.dump(recomputed_results, f, indent=2)

print(f"\nTum sonuclar basariyla kaydedildi: {OUT_FILE}")
