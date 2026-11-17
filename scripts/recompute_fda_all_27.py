import pickle, json, time, glob
import numpy as np
from scipy.stats import trim_mean, rankdata

def agg_simple(P, **_):
    return P.mean(axis=0)

def agg_perf(P, rmses, **_):
    w = 1.0 / np.array(rmses, dtype=float)
    w /= w.sum()
    return (P * w[:, None]).sum(axis=0)

def agg_softmax_bma(P, rmses, beta=2.0, **_):
    r = np.array(rmses, dtype=float)
    ptp = np.ptp(r)
    if ptp < 1e-12:
        return P.mean(axis=0)
    r_sc = (r - np.min(r)) / ptp
    w = np.exp(-beta * r_sc)
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

def agg_ava_fast(P, **_):
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
    "M5:AVA-PDP":        agg_ava_fast,
}

def compute_weights(method, P, rmses):
    k = len(P)
    if method == "B0:SimpleAvg":
        return np.ones(k) / k
    if method == "B2:PerfWeighted":
        w = 1.0 / np.array(rmses, dtype=float); return w / w.sum()
    if method == "M1:SoftmaxBMA":
        r = np.array(rmses, dtype=float)
        ptp = np.ptp(r)
        if ptp < 1e-12: return np.ones(k) / k
        w = np.exp(-2.0 * (r - np.min(r)) / ptp); return w / w.sum()
    if method == "M5:AVA-PDP":
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

def bootstrap_ci(method, fn, P, rmses, fixed_w, B=100, alpha=0.05):
    k = len(P)
    boots1, boots2 = [], []
    for _ in range(B):
        idx = np.random.choice(k, k, replace=True)
        Pb = P[idx]
        rb = [rmses[i] for i in idx]
        if method in ["B0:SimpleAvg", "M2:PointMedian", "M3:TrimMean", "M4:FuncMedian", "M5:AVA-PDP"]:
            a1 = fn(Pb)
        else:
            a1 = fn(Pb, rmses=rb)
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
            fpath = f"outputs/pdps/{scen_id}_pdp_data.pkl"
            with open(fpath, "rb") as f:
                d = pickle.load(f)

            all_rmses = np.array(d["all_rmses"])
            pdp_all = np.array(d["pdp_all"])
            gt = d["gt_fixed"]
            b1_md = float(d["b1_md_fixed"])
            best_rmse = float(np.min(all_rmses))
            N_tot = len(all_rmses)

            scen_dict = {
                "scenario": scen_id,
                "setup": snum,
                "noise_var": nv,
                "b1_md_fixed": b1_md,
                "pdp_file": fpath,
                "by_epsilon": {}
            }

            for eps in eps_list:
                eps_key = f"eps_{eps:.2f}"
                rash_idx = np.where(all_rmses <= best_rmse * (1 + eps))[0]
                P = pdp_all[rash_idx]
                rmses = all_rmses[rash_idx]
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
                    if m_name in ["B0:SimpleAvg", "M2:PointMedian", "M3:TrimMean", "M4:FuncMedian", "M5:AVA-PDP"]:
                        agg_curve = fn(P)
                    else:
                        agg_curve = fn(P, rmses=rmses)

                    md = float(np.mean(np.abs(agg_curve - gt)))
                    fw = compute_weights(m_name, P, rmses)
                    (lo1, hi1), (lo2, hi2) = bootstrap_ci(m_name, fn, P, rmses, fw, B=100)

                    cr1 = float(np.mean((gt >= lo1) & (gt <= hi1)))
                    mwci1 = float(np.mean(hi1 - lo1))
                    cr2 = float(np.mean((gt >= lo2) & (gt <= hi2)))
                    mwci2 = float(np.mean(hi2 - lo2))

                    eps_res["results"][m_name] = {
                        "md": md,
                        "cr1": cr1, "mwci1": mwci1,
                        "cr2": cr2, "mwci2": mwci2,
                        "agg_pdp": agg_curve.tolist()
                    }

                scen_dict["by_epsilon"][eps_key] = eps_res

            master_results[scen_id] = scen_dict
            print(f"Hesaplandı: {scen_id} (K range: {[scen_dict['by_epsilon'][k]['n_rash'] for k in scen_dict['by_epsilon']]})")

    out_file = "outputs/master_analysis_results_fda_27scenarios.json"
    with open(out_file, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\nTüm 27 senaryo (108 durum) başarıyla kaydedildi: {out_file} (Süre: {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
