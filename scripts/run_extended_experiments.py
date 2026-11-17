"""
Genisletilmis Rashomon PDP Deneyleri
- Max models: 80
- Max runtime: 720 s
- Epsilon degerleri: 0.01, 0.05, 0.10, 0.20
- Senaryolar: 3 Setup x 3 Gurultu (sigma^2 in {1, 4, 9})
"""
import warnings; warnings.filterwarnings("ignore")
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
import cvxpy as cp
import json, time, os
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare

# Analysis modulunden fonksiyonlari al
from analysis import (
    SETUP_FNS, METHOD_NAMES, AGG_FNS,
    compute_pdp, compute_fixed_weights, bootstrap_ci,
    coverage_rate, mwci, rashomon_set, OUT
)

EPS_LIST = [0.01, 0.05, 0.10, 0.20]
MAX_MODELS = 80
MAX_RUNTIME = 720

def run_experiment_multieps(snum, nv, eps_list=EPS_LIST, max_models=MAX_MODELS, max_runtime=MAX_RUNTIME):
    print(f"\n==========================================")
    print(f"Calisiyor: Setup {snum}, sigma^2 = {nv}")
    print(f"==========================================")
    t0 = time.time()
    
    df, feat, gt_fn = SETUP_FNS[snum](nv=nv)
    
    # AutoML egitimi
    hf = h2o.H2OFrame(df)
    feat_cols = [c for c in df.columns if c != "y"]
    tr, _ = hf.split_frame(ratios=[0.8], seed=42)
    
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime,
        seed=42,
        sort_metric="RMSE",
        exclude_algos=["StackedEnsemble", "DeepLearning"]
    )
    aml.train(x=feat_cols, y="y", training_frame=tr)
    lb = aml.leaderboard.as_data_frame()[["model_id", "rmse"]].dropna()
    print(f"AutoML tamamlandi ({len(lb)} model, {time.time()-t0:.1f}s)")
    
    # Modelleri ve PDP'leri hesapla
    df_tr = tr.as_data_frame()
    all_models = []
    all_rmses = []
    for _, row in lb.iterrows():
        try:
            m = h2o.get_model(row["model_id"])
            all_models.append(m)
            all_rmses.append(row["rmse"])
        except Exception as e:
            pass
            
    best_rmse = all_rmses[0]
    vals, best_pdp = compute_pdp(all_models[0], df_tr, feat)
    pdp_all = np.array([compute_pdp(m, df_tr, feat, gp=len(vals))[1] for m in all_models])
    
    gt_raw = gt_fn(vals)
    
    scenario_res = {}
    
    for eps in eps_list:
        print(f"  --- Epsilon = {eps:.2f} ---")
        rash_idx = [i for i, r in enumerate(all_rmses) if r <= best_rmse * (1 + eps)]
        n_rash = len(rash_idx)
        rho = n_rash / len(all_rmses)
        print(f"      K = {n_rash}, rho = {rho:.3f}")
        
        if n_rash < 2:
            print(f"      UYARI: K < 2 (K={n_rash}), tekil model fallback")
            # K=1 ise tum metodlar Best-model ile ayni
            P = pdp_all[rash_idx]
            gt = gt_raw - gt_raw.mean() + P.mean()
            md_val = float(np.mean(np.abs(best_pdp - gt)))
            res_eps = {
                "vals": vals.tolist(), "gt": gt.tolist(),
                "best_pdp": best_pdp.tolist(), "rho": rho, "n_rash": n_rash,
                "results": {
                    "B1:Best-model": {"md": md_val, "cr1": None, "cr2": None, "mwci1": None, "mwci2": None}
                }
            }
            for m in AGG_FNS:
                res_eps["results"][m] = {"md": md_val, "cr1": 0.0, "cr2": 0.0, "mwci1": 0.0, "mwci2": 0.0}
            scenario_res[f"eps_{eps}"] = res_eps
            continue
            
        P = pdp_all[rash_idx]
        rmses = [all_rmses[i] for i in rash_idx]
        ref = P.mean(0)
        gt = gt_raw - gt_raw.mean() + P.mean()
        
        res_eps = {
            "vals": vals.tolist(),
            "gt": gt.tolist(),
            "best_pdp": best_pdp.tolist(),
            "rho": rho,
            "n_rash": n_rash,
            "results": {
                "B1:Best-model": {
                    "md": float(np.mean(np.abs(best_pdp - gt))),
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
            
        b1_md = res_eps["results"]["B1:Best-model"]["md"]
        b0_md = res_eps["results"]["B0:SimpleAvg"]["md"]
        b3_md = res_eps["results"]["B3:AGGrobust"]["md"]
        print(f"      MD: B1={b1_md:.4f}, B0={b0_md:.4f}, B3={b3_md:.4f}")
        scenario_res[f"eps_{eps}"] = res_eps
        
    return scenario_res

def main():
    print("H2O baslatiliyor...")
    h2o.init(nthreads=4, max_mem_size="4G", verbose=False)
    h2o.no_progress()
    
    os.makedirs(OUT, exist_ok=True)
    all_experiments = {}
    CKPT_PATH = os.path.join(OUT, "extended_experiments.json")
    
    if os.path.exists(CKPT_PATH):
        try:
            with open(CKPT_PATH, "r") as f:
                all_experiments = json.load(f)
            print(f"Mevcut sonuclar yuklendi: {list(all_experiments.keys())}")
        except:
            pass
            
    for snum in [1, 2, 3]:
        for nv in [1, 4, 9]:
            key = f"S{snum}_nv{nv}"
            if key in all_experiments:
                print(f"{key} zaten mevcut, atlaniyor.")
                continue
            res = run_experiment_multieps(snum, nv)
            all_experiments[key] = res
            with open(CKPT_PATH, "w") as f:
                json.dump(all_experiments, f, indent=2)
            print(f"{key} kaydedildi.")
            
    h2o.cluster().shutdown()
    print("\nTum 9 senaryo x 4 epsilon deneyi basariyla tamamlandi!")

if __name__ == "__main__":
    main()
