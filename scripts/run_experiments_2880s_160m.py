"""
AutoML Bütçesi 2 Katına Çıkarılmış Deneyler:
- Max models: 160 (80 -> 160)
- Max runtime: 2880 saniye (48 dakika) (1440 -> 2880)
- Rashomon epsilon parametreleri: 0.01, 0.05, 0.10, 0.20
- Senaryolar: 3 Setup x 3 Gurultu (sigma^2 in {1, 4, 9})
- Model PDP nesneleri ve verileri outputs/pdps/<Scenario>_pdp_data.pkl altina kaydedilir.
- H2O modelleri outputs/models/<Scenario>/ altina kaydedilir.
- Analizlerde B1 referansi sabit tutulur (gt = gt_raw - gt_raw.mean() + best_pdp.mean()).
"""
import warnings; warnings.filterwarnings("ignore")
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
import cvxpy as cp
import json, time, os, pickle

from analysis import (
    SETUP_FNS, METHOD_NAMES, AGG_FNS,
    compute_pdp, compute_fixed_weights, bootstrap_ci,
    coverage_rate, mwci, OUT
)

EPS_LIST = [0.01, 0.05, 0.10, 0.20]
MAX_MODELS = 160
MAX_RUNTIME = 2880

PDP_DIR = os.path.join(OUT, "pdps")
MODEL_DIR = os.path.join(OUT, "models")
os.makedirs(PDP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_scenario(snum, nv, eps_list=EPS_LIST, max_models=MAX_MODELS, max_runtime=MAX_RUNTIME):
    scenario_key = f"S{snum}_nv{nv}"
    print(f"\n{'='*55}")
    print(f"Basliyor: {scenario_key} | Max Models: {max_models} | Max Runtime: {max_runtime}s")
    print(f"{'='*55}")
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
    lb_df = aml.leaderboard.as_data_frame()
    lb = lb_df[["model_id", "rmse"]].dropna()
    print(f"AutoML tamamlandi ({len(lb)} model egitildi/inceleniyor, sure: {time.time()-t0:.1f}s)")
    
    df_tr = tr.as_data_frame()
    all_models = []
    all_rmses = []
    model_ids = []
    
    sc_model_dir = os.path.join(MODEL_DIR, scenario_key)
    os.makedirs(sc_model_dir, exist_ok=True)
    
    for _, row in lb.iterrows():
        mid = row["model_id"]
        try:
            m = h2o.get_model(mid)
            all_models.append(m)
            all_rmses.append(row["rmse"])
            model_ids.append(mid)
            h2o.save_model(model=m, path=sc_model_dir, force=True)
        except Exception as e:
            pass
            
    best_rmse = all_rmses[0]
    vals, best_pdp = compute_pdp(all_models[0], df_tr, feat)
    
    print(f"{len(all_models)} model icin PDP hesaplaniyor...")
    pdp_dict = {}
    pdp_list = []
    for m, mid in zip(all_models, model_ids):
        _, pdp_curve = compute_pdp(m, df_tr, feat, gp=len(vals))
        pdp_dict[mid] = pdp_curve
        pdp_list.append(pdp_curve)
        
    pdp_all = np.array(pdp_list)
    gt_raw = gt_fn(vals)
    
    # Sabit ground truth referansi (B1'e gore merkezlenmis)
    gt = gt_raw - gt_raw.mean() + best_pdp.mean()
    b1_md = float(np.mean(np.abs(best_pdp - gt)))
    
    pdp_save_path = os.path.join(PDP_DIR, f"{scenario_key}_pdp_data.pkl")
    pdp_payload = {
        "scenario": scenario_key,
        "setup": snum,
        "noise_var": nv,
        "feature": feat,
        "vals": vals,
        "model_ids": model_ids,
        "all_rmses": all_rmses,
        "pdp_dict": pdp_dict,
        "pdp_all": pdp_all,
        "best_pdp": best_pdp,
        "gt_raw": gt_raw,
        "gt_fixed": gt,
        "b1_md_fixed": b1_md,
        "leaderboard": lb_df
    }
    with open(pdp_save_path, "wb") as f:
        pickle.dump(pdp_payload, f)
    print(f"-> Model PDP nesneleri ve verileri kaydedildi: {pdp_save_path}")
    
    scenario_res = {
        "scenario": scenario_key,
        "setup": snum,
        "noise_var": nv,
        "feature": feat,
        "vals": vals.tolist(),
        "model_ids": model_ids,
        "all_rmses": all_rmses,
        "b1_md_fixed": b1_md,
        "pdp_file": pdp_save_path,
        "by_epsilon": {}
    }
    
    for eps in eps_list:
        print(f"  --- Epsilon = {eps:.2f} ---")
        rash_idx = [i for i, r in enumerate(all_rmses) if r <= best_rmse * (1 + eps)]
        n_rash = len(rash_idx)
        rho = n_rash / len(all_rmses)
        print(f"      K = {n_rash}, rho = {rho:.3f}")
        
        if n_rash < 2:
            res_eps = {
                "rho": rho, "n_rash": n_rash, "rash_models": [model_ids[i] for i in rash_idx],
                "results": {
                    "B1:Best-model": {"md": b1_md, "cr1": None, "cr2": None, "mwci1": None, "mwci2": None}
                }
            }
            for m in AGG_FNS:
                res_eps["results"][m] = {"md": b1_md, "cr1": 0.0, "cr2": 0.0, "mwci1": 0.0, "mwci2": 0.0}
            scenario_res["by_epsilon"][f"eps_{eps}"] = res_eps
            continue
            
        P = pdp_all[rash_idx]
        rmses = [all_rmses[i] for i in rash_idx]
        ref = P.mean(0)
        
        res_eps = {
            "rho": rho,
            "n_rash": n_rash,
            "rash_models": [model_ids[i] for i in rash_idx],
            "results": {
                "B1:Best-model": {
                    "md": b1_md,
                    "agg": best_pdp.tolist(),
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
                "agg": agg.tolist(),
                "cr1": coverage_rate(gt, lo1, hi1),
                "cr2": coverage_rate(gt, lo2, hi2),
                "mwci1": mwci(lo1, hi1),
                "mwci2": mwci(lo2, hi2)
            }
            
        b0_md = res_eps["results"]["B0:SimpleAvg"]["md"]
        b3_md = res_eps["results"]["B3:AGGrobust"]["md"]
        print(f"      MD: B1={b1_md:.4f}, B0={b0_md:.4f}, B3={b3_md:.4f}")
        scenario_res["by_epsilon"][f"eps_{eps}"] = res_eps
        
    return scenario_res

def main():
    print("H2O baslatiliyor...")
    h2o.init(nthreads=4, max_mem_size="4G", verbose=False)
    h2o.no_progress()
    
    MASTER_RESULTS_PATH = os.path.join(OUT, "master_analysis_results_2880s_160m.json")
    all_experiments = {}
    
    for snum in [1, 2, 3]:
        for nv in [1, 4, 9]:
            key = f"S{snum}_nv{nv}"
            res = run_scenario(snum, nv)
            all_experiments[key] = res
            with open(MASTER_RESULTS_PATH, "w") as f:
                json.dump(all_experiments, f, indent=2)
            print(f"✓ {key} kaydedildi.")
            
    h2o.cluster().shutdown()
    print("\n" + "="*60)
    print("✓ Tum 9 senaryo 160 model ve 2880s bütçesiyle tamamlandi!")
    print(f"✓ Sonuclar: {MASTER_RESULTS_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()
