"""
run_recompute_all.py
--------------------
Yeniden hesaplama betiği:
- codes/analysis_updated.py içerisindeki güncel metodolojiyi (B3, B4, B8, Ö1, Ö2) kullanır.
- outputs/pdps/<Scenario>_pdp_data.pkl içerisindeki kaydedilmiş 160-model / 2880s verilerini kullanır.
- B1'e göre merkezlenmiş sabit ground-truth referansını korur.
- Hem MD değerlerini hem de B=100 bootstrap CI1 / CI2 değerlerini hesaplar.
- outputs/master_analysis_results_2880s_160m_updated.json dosyasına kaydeder.
"""
import os, sys, json, pickle, time
import numpy as np
from scipy.stats import rankdata

import analysis_updated as au

PDP_DIR = "outputs/pdps"
OUT_DIR = "outputs"
EPS_LIST = [0.01, 0.05, 0.10, 0.20]
SCENARIOS = [f"S{s}_nv{nv}" for s in [1, 2, 3] for nv in [1, 4, 9]]
N_BOOTS = 100

def fast_agg_ava(P):
    k = len(P)
    if k == 1:
        return P[0]
    ranks = np.apply_along_axis(rankdata, 1, P)
    ranks_c = ranks - ranks.mean(axis=1, keepdims=True)
    cov = ranks_c @ ranks_c.T
    std = np.sqrt(np.diag(cov)) + 1e-12
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 0)
    avg_corr = corr.sum(axis=1) / (k - 1)
    w = np.clip(avg_corr + 1e-8, 1e-8, None)
    return (P * (w / w.sum())[:, None]).sum(0)

def fast_compute_fw(method, P, rmses, ref):
    if method == "B4:AVA-PDP":
        k = len(P)
        if k == 1:
            return np.ones(1)
        ranks = np.apply_along_axis(rankdata, 1, P)
        ranks_c = ranks - ranks.mean(axis=1, keepdims=True)
        cov = ranks_c @ ranks_c.T
        std = np.sqrt(np.diag(cov)) + 1e-12
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 0)
        avg_corr = corr.sum(axis=1) / (k - 1)
        w = np.clip(avg_corr + 1e-8, 1e-8, None)
        return w / w.sum()
    return au.compute_fixed_weights(method, P, rmses, ref)

def fast_bootstrap_ci(method, P, rmses, ref, fixed_weights, B=N_BOOTS):
    k, m = P.shape
    boots1, boots2 = [], []
    for _ in range(B):
        idx = np.random.choice(k, k, replace=True)
        Pb = P[idx]
        rb = [rmses[i] for i in idx]
        refb = Pb.mean(0)
        if method == "B0:SimpleAvg":
            a1 = Pb.mean(0)
        elif method == "B2:PerfWeighted":
            w = 1.0 / (np.array(rb) + 1e-8); w /= w.sum(); a1 = (Pb * w[:, None]).sum(0)
        elif method == "B3:AGGrobust":
            a1 = au.agg_robust(Pb, rmses=rb, np_=5)
        elif method == "B4:AVA-PDP":
            a1 = fast_agg_ava(Pb)
        elif method == "B8:AGREE-Rank":
            a1 = au.agg_rank(Pb)
        elif method == "Ö1:AGGfaith":
            a1 = au.agg_faith(Pb, rmses=rb)
        elif method == "Ö2:AGGopt":
            a1 = au.agg_opt(Pb, rmses=rb, np_=5)
        else:
            a1 = Pb.mean(0)
        boots1.append(a1)
        
        wb = fixed_weights[idx]
        wb = wb / wb.sum() if wb.sum() > 0 else np.ones(k) / k
        boots2.append((Pb * wb[:, None]).sum(0))
        
    b1 = np.array(boots1); b2 = np.array(boots2)
    return (np.percentile(b1, 2.5, 0), np.percentile(b1, 97.5, 0)), \
           (np.percentile(b2, 2.5, 0), np.percentile(b2, 97.5, 0))

def main():
    print("=" * 65)
    print("YENIDEN HESAPLAMA BASLADI (analysis_updated.py adaptasyonlari)")
    print("=" * 65)
    
    master_results = {}
    t_start = time.time()
    
    for sc in SCENARIOS:
        pdp_file = os.path.join(PDP_DIR, f"{sc}_pdp_data.pkl")
        print(f"\n>>> Senaryo: {sc} ({pdp_file})")
        with open(pdp_file, "rb") as f:
            data = pickle.load(f)
            
        vals = np.array(data["vals"])
        best_pdp = np.array(data["best_pdp"])
        pdp_all = np.array(data["pdp_all"])
        all_rmses = data["all_rmses"]
        model_ids = data["model_ids"]
        gt_raw = np.array(data["gt_raw"])
        
        gt = gt_raw - gt_raw.mean() + best_pdp.mean()
        b1_md = float(np.mean(np.abs(best_pdp - gt)))
        best_rmse = all_rmses[0]
        
        sc_res = {
            "scenario": sc,
            "setup": data.get("setup", int(sc[1])),
            "noise_var": data.get("noise_var", int(sc.split("nv")[1])),
            "feature": data.get("feature", "X1"),
            "vals": vals.tolist(),
            "model_ids": model_ids,
            "all_rmses": all_rmses,
            "b1_md_fixed": b1_md,
            "pdp_file": pdp_file,
            "by_epsilon": {}
        }
        
        for eps in EPS_LIST:
            rash_idx = [i for i, r in enumerate(all_rmses) if r <= best_rmse * (1 + eps)]
            n_rash = len(rash_idx)
            rho = n_rash / len(all_rmses)
            
            if n_rash < 2:
                res_eps = {
                    "rho": rho,
                    "n_rash": n_rash,
                    "rash_models": [model_ids[i] for i in rash_idx],
                    "results": {
                        "B1:Best-model": {"md": b1_md, "cr1": None, "cr2": None, "mwci1": None, "mwci2": None}
                    }
                }
                for m in au.AGG_FNS:
                    res_eps["results"][m] = {"md": b1_md, "cr1": 0.0, "cr2": 0.0, "mwci1": 0.0, "mwci2": 0.0}
                sc_res["by_epsilon"][f"eps_{eps}"] = res_eps
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
            
            for m in au.AGG_FNS:
                if m == "B4:AVA-PDP":
                    agg = fast_agg_ava(P)
                else:
                    agg = au.AGG_FNS[m](P=P, rmses=rmses, ref=ref)
                
                fw = fast_compute_fw(m, P, rmses, ref)
                (lo1, hi1), (lo2, hi2) = fast_bootstrap_ci(m, P, rmses, ref, fw, B=N_BOOTS)
                
                res_eps["results"][m] = {
                    "md": float(np.mean(np.abs(agg - gt))),
                    "agg": agg.tolist(),
                    "cr1": au.coverage_rate(gt, lo1, hi1),
                    "cr2": au.coverage_rate(gt, lo2, hi2),
                    "mwci1": au.mwci(lo1, hi1),
                    "mwci2": au.mwci(lo2, hi2)
                }
                
            sc_res["by_epsilon"][f"eps_{eps}"] = res_eps
            print(f"  eps={eps:.2f} | K={n_rash:3d} | MD: B1={b1_md:.4f}, B0={res_eps['results']['B0:SimpleAvg']['md']:.4f}, B4={res_eps['results']['B4:AVA-PDP']['md']:.4f}, B8={res_eps['results']['B8:AGREE-Rank']['md']:.4f}")
            
        master_results[sc] = sc_res
        
    out_path = os.path.join(OUT_DIR, "master_analysis_results_2880s_160m_updated.json")
    with open(out_path, "w") as f:
        json.dump(master_results, f, indent=2)
        
    print("\n" + "=" * 65)
    print(f"HESAPLAMA TAMAMLANDI (Toplam sure: {time.time()-t_start:.1f}s)")
    print(f"Kaydedildi: {out_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
