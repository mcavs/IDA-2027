import pickle, json, time, glob, os
import numpy as np
from scipy.stats import trim_mean, rankdata

def agg_simple(P, **_):
    return P.mean(axis=0)

def agg_perf(P, loglosses, **_):
    ll = np.array(loglosses, dtype=float)
    # Inverse logloss weighting (lower logloss -> higher weight)
    # Add small epsilon to prevent division by zero
    w = 1.0 / (ll + 1e-12)
    w /= w.sum()
    return (P * w[:, None]).sum(axis=0)

def agg_softmax_bma(P, loglosses, beta=2.0, **_):
    ll = np.array(loglosses, dtype=float)
    ptp = np.ptp(ll)
    if ptp < 1e-12:
        return P.mean(axis=0)
    ll_sc = (ll - np.min(ll)) / ptp
    w = np.exp(-beta * ll_sc)
    w /= w.sum()
    return (P * w[:, None]).sum(axis=0)

def agg_pointwise_median(P, **_):
    return np.median(P, axis=0)

def agg_trimmed_mean(P, alpha=0.1, **_):
    if len(P) < 5:
        return P.mean(axis=0)
    return trim_mean(P, proportiontocut=alpha, axis=0)

def agg_functional_median(P, **_):
    k = len(P)
    if k <= 2:
        return P.mean(axis=0)
    dists = np.sum(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1), axis=1)
    return P[np.argmin(dists)]

def agg_pairwise_alignment(P, **_):
    k, m = P.shape
    if k <= 1:
        return P[0]
    ranks = rankdata(P, axis=1)
    ranks_centered = ranks - ranks.mean(axis=1, keepdims=True)
    stds = np.std(ranks, axis=1, keepdims=True)
    cov = np.dot(ranks_centered, ranks_centered.T) / (m - 1)
    denom = np.dot(stds, stds.T) + 1e-12
    corr_matrix = cov / denom
    np.fill_diagonal(corr_matrix, 0)
    avg_corr = np.clip(corr_matrix.sum(axis=1) / (k - 1), 1e-8, None)
    w = avg_corr / avg_corr.sum()
    return (P * w[:, None]).sum(axis=0)

METHODS = {
    "B0:SimpleAvg":      agg_simple,
    "B2:PerfWeighted":   agg_perf,
    "M1:SoftmaxBMA":     agg_softmax_bma,
    "M2:PointMedian":    agg_pointwise_median,
    "M3:TrimMean":       agg_trimmed_mean,
    "M4:FuncMedian":     agg_functional_median,
    "M5:PairwiseAlign":  agg_pairwise_alignment,
}

def compute_weights(method, P, loglosses):
    k = len(P)
    if method == "B0:SimpleAvg":
        return np.ones(k) / k
    if method == "B2:PerfWeighted":
        ll = np.array(loglosses, dtype=float)
        w = 1.0 / (ll + 1e-12); return w / w.sum()
    if method == "M1:SoftmaxBMA":
        ll = np.array(loglosses, dtype=float)
        ptp = np.ptp(ll)
        if ptp < 1e-12: return np.ones(k) / k
        w = np.exp(-2.0 * (ll - np.min(ll)) / ptp); return w / w.sum()
    if method == "M5:PairwiseAlign":
        if k <= 1: return np.ones(1)
        ranks = rankdata(P, axis=1)
        ranks_c = ranks - ranks.mean(axis=1, keepdims=True)
        stds = np.std(ranks, axis=1, keepdims=True)
        cov = np.dot(ranks_c, ranks_c.T) / (P.shape[1] - 1)
        denom = np.dot(stds, stds.T) + 1e-12
        cm = cov / denom
        np.fill_diagonal(cm, 0)
        ac = np.clip(cm.sum(axis=1) / (k - 1), 1e-8, None)
        return ac / ac.sum()
    return np.ones(k) / k

def bootstrap_ci(method, fn, P, loglosses, fixed_w, B=100, alpha=0.05):
    k = len(P)
    boots1, boots2 = [], []
    for _ in range(B):
        idx = np.random.choice(k, k, replace=True)
        Pb = P[idx]
        ll_b = [loglosses[i] for i in idx]
        if method in ["B0:SimpleAvg", "M2:PointMedian", "M3:TrimMean", "M4:FuncMedian", "M5:PairwiseAlign"]:
            a1 = fn(Pb)
        else:
            a1 = fn(Pb, loglosses=ll_b)
        boots1.append(a1)
        boots2.append((Pb * fixed_w[:, None]).sum(axis=0))
    b1 = np.array(boots1)
    b2 = np.array(boots2)
    lo1, hi1 = np.percentile(b1, 100 * alpha / 2, axis=0), np.percentile(b1, 100 * (1 - alpha / 2), axis=0)
    lo2, hi2 = np.percentile(b2, 100 * alpha / 2, axis=0), np.percentile(b2, 100 * (1 - alpha / 2), axis=0)
    return (lo1, hi1), (lo2, hi2)

def main():
    t0 = time.time()
    setups = [1, 2, 3]
    all_nvs = list(range(1, 10))
    eps_list = [0.01, 0.05, 0.10, 0.20]

    master_results = {}

    for snum in setups:
        for nv in all_nvs:
            scen_id = f"S{snum}_nv{nv}"
            fpath = f"outputs/cls_pdps/{scen_id}_cls_pdp_data.pkl"
            with open(fpath, "rb") as f:
                d = pickle.load(f)

            all_loglosses = np.array(d["all_loglosses"])
            all_aucs = np.array(d["all_aucs"])
            pdp_all = np.array(d["pdp_all"])
            gt = d["gt_pdp"]
            b1_md = float(d["b1_md"])
            best_ll = float(np.min(all_loglosses))
            best_auc = float(np.max(all_aucs))
            N_tot = len(all_loglosses)

            scen_dict = {
                "scenario": scen_id,
                "setup": snum,
                "noise_var": nv,
                "b1_md": b1_md,
                "best_logloss": best_ll,
                "best_auc": best_auc,
                "pdp_file": fpath,
                "by_epsilon": {}
            }

            for eps in eps_list:
                eps_key = f"eps_{eps:.2f}"
                # Rashomon set defined by logloss within (1 + eps) * best_ll
                rash_idx = np.where(all_loglosses <= best_ll * (1.0 + eps))[0]
                if len(rash_idx) == 0:
                    rash_idx = np.array([np.argmin(all_loglosses)])
                
                P = pdp_all[rash_idx]
                loglosses = all_loglosses[rash_idx]
                K = len(P)
                rho = float(K / N_tot)

                eps_res = {
                    "rho": rho,
                    "n_rash": K,
                    "results": {}
                }

                eps_res["results"]["B1:Best-model"] = {
                    "md": b1_md,
                    "cr1": 0.0, "mwci1": 0.0,
                    "cr2": 0.0, "mwci2": 0.0
                }

                for m_name, fn in METHODS.items():
                    if m_name in ["B0:SimpleAvg", "M2:PointMedian", "M3:TrimMean", "M4:FuncMedian", "M5:PairwiseAlign"]:
                        agg_curve = fn(P)
                    else:
                        agg_curve = fn(P, loglosses=loglosses)

                    md = float(np.mean(np.abs(agg_curve - gt)))
                    fixed_w = compute_weights(m_name, P, loglosses)

                    # Bootstrap CI
                    (lo1, hi1), (lo2, hi2) = bootstrap_ci(m_name, fn, P, loglosses, fixed_w, B=100)
                    cov1 = np.mean((gt >= lo1) & (gt <= hi1))
                    w1 = np.mean(hi1 - lo1)
                    cov2 = np.mean((gt >= lo2) & (gt <= hi2))
                    w2 = np.mean(hi2 - lo2)

                    eps_res["results"][m_name] = {
                        "md": md,
                        "cr1": float(cov1), "mwci1": float(w1),
                        "cr2": float(cov2), "mwci2": float(w2)
                    }

                scen_dict["by_epsilon"][eps_key] = eps_res

            master_results[scen_id] = scen_dict
            print(f"[{scen_id}] Best AUC={best_auc:.4f}, Best LL={best_ll:.4f}, B1_MD={b1_md:.4f}, S_Avg(0.05)={scen_dict['by_epsilon']['eps_0.05']['results']['B0:SimpleAvg']['md']:.4f}")

    out_file = "outputs/master_analysis_results_classification_27scenarios.json"
    with open(out_file, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\nAnalysis complete in {time.time()-t0:.1f}s. Saved to {out_file}")

if __name__ == "__main__":
    main()
