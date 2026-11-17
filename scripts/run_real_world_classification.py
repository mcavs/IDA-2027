import urllib.request
import json
import io
import time
import pickle
import os
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from scipy.stats import trim_mean, rankdata

def get_arff_data(did):
    print(f"Downloading dataset did={did} from OpenML...")
    req = urllib.request.urlopen(f'https://www.openml.org/api/v1/json/data/{did}')
    meta = json.loads(req.read().decode('utf-8'))
    file_url = meta['data_set_description']['url']
    req_file = urllib.request.urlopen(file_url)
    lines = [line.decode('utf-8', errors='ignore') for line in req_file.readlines()]
    data_idx = 0
    col_names = []
    for i, l in enumerate(lines):
        if l.strip().lower().startswith('@attribute'):
            parts = l.strip().split()
            col_names.append(parts[1].replace("'", "").replace('"', ''))
        elif l.strip().lower().startswith('@data'):
            data_idx = i + 1
            break
    data_lines = ''.join(lines[data_idx:])
    df = pd.read_csv(io.StringIO(data_lines), header=None, names=col_names, na_values=['?'])
    return df

# 8 Aggregation functions
def agg_simple(P, **_):
    return P.mean(axis=0)

def agg_perf(P, loglosses, **_):
    ll = np.array(loglosses, dtype=float)
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

def compute_pdps_for_dataset(df, target_col, feature_name, max_models=50, max_runtime_secs=300, eps=0.05, sample_n=5000):
    if len(df) > sample_n:
        print(f"Subsampling {len(df)} rows to {sample_n} rows for efficient AutoML...")
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        
    # clean na
    df = df.dropna(subset=[target_col, feature_name]).reset_index(drop=True)
    
    # ensure target is categorical factor in h2o
    hf = h2o.H2OFrame(df)
    hf[target_col] = hf[target_col].asfactor()
    
    features = [c for c in df.columns if c != target_col]
    
    print(f"Running AutoML on target={target_col}, num_rows={len(df)}, features={len(features)}...")
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=42,
        exclude_algos=["StackedEnsemble", "DeepLearning"],
        sort_metric="logloss"
    )
    aml.train(x=features, y=target_col, training_frame=hf)
    
    lb = aml.leaderboard.as_data_frame()
    print(f"Trained {len(lb)} models. Best logloss: {lb['logloss'].min():.4f}, Best AUC: {lb['auc'].max():.4f}")
    
    # 20 grid values
    feature_vals = np.percentile(df[feature_name].dropna(), np.linspace(5, 95, 20))
    feature_vals = np.unique(feature_vals)
    if len(feature_vals) < 20:
        feature_vals = np.linspace(df[feature_name].min(), df[feature_name].max(), 20)
        
    # Extract PDPs for models in Rashomon set
    best_ll = lb['logloss'].min()
    rash_lb = lb[lb['logloss'] <= best_ll * (1.0 + eps)].copy()
    if len(rash_lb) == 0:
        rash_lb = lb.iloc[[0]]
        
    print(f"Rashomon set size (|R_eps| at eps={eps}): {len(rash_lb)} of {len(lb)} models (rho={len(rash_lb)/len(lb):.2f})")
    
    pdp_list = []
    logloss_list = []
    
    # Prepare background evaluation frame for PDP (sample 300 instances for speed)
    bg_df = df.sample(min(300, len(df)), random_state=42).copy()
    
    for idx, row in rash_lb.iterrows():
        mid = row['model_id']
        m = h2o.get_model(mid)
        ll = float(row['logloss'])
        logloss_list.append(ll)
        
        # Calculate PDP across grid
        preds_grid = []
        for v in feature_vals:
            temp_df = bg_df.copy()
            temp_df[feature_name] = v
            temp_hf = h2o.H2OFrame(temp_df)
            p = m.predict(temp_hf).as_data_frame()
            # positive class probability is column p1 (or second col if p0, p1)
            prob_col = p.columns[-1] if 'p1' not in p.columns else 'p1'
            preds_grid.append(float(p[prob_col].mean()))
            h2o.remove(temp_hf)
            
        pdp_list.append(preds_grid)
        
    P = np.array(pdp_list) # shape: (K, 20)
    best_pdp = P[0] # top model
    
    # Compute aggregations
    res = {
        "feature_name": feature_name,
        "grid_vals": feature_vals.tolist(),
        "K": len(rash_lb),
        "rho": float(len(rash_lb) / len(lb)),
        "best_logloss": float(best_ll),
        "best_auc": float(lb['auc'].max()),
        "curves": {
            "B1 (Single Best)": best_pdp.tolist(),
            "B0 (Simple Avg)": agg_simple(P).tolist(),
            "B2 (Perf Weighted)": agg_perf(P, logloss_list).tolist(),
            "M1 (Softmax BMA)": agg_softmax_bma(P, logloss_list).tolist(),
            "M2 (Pointwise Median)": agg_pointwise_median(P).tolist(),
            "M3 (Trimmed Mean)": agg_trimmed_mean(P).tolist(),
            "M4 (Fréchet Median)": agg_functional_median(P).tolist(),
            "M5 (Pairwise Alignment)": agg_pairwise_alignment(P).tolist()
        }
    }
    return res

def main():
    h2o.init(nthreads=-1, max_mem_size="8G")
    h2o.no_progress()
    
    os.makedirs("outputs/real_world", exist_ok=True)
    
    # 1. Adult Dataset
    print("\n==========================================")
    print("EXPERIMENT 1: adult (did=1590), feature='age'")
    print("==========================================")
    df_adult = get_arff_data(1590)
    # Target column is class
    target_adult = 'class' if 'class' in df_adult.columns else df_adult.columns[-1]
    res_adult = compute_pdps_for_dataset(df_adult, target_adult, 'age', max_models=40, max_runtime_secs=360, eps=0.05, sample_n=4000)
    
    # 2. Churn Dataset
    print("\n==========================================")
    print("EXPERIMENT 2: churn (did=40701), feature='total_day_minutes'")
    print("==========================================")
    df_churn = get_arff_data(40701)
    target_churn = 'class' if 'class' in df_churn.columns else df_churn.columns[-1]
    res_churn = compute_pdps_for_dataset(df_churn, target_churn, 'total_day_minutes', max_models=40, max_runtime_secs=360, eps=0.05)
    
    # Save results
    results_all = {
        "adult": res_adult,
        "churn": res_churn
    }
    out_json = "outputs/real_world/real_world_classification_results.json"
    with open(out_json, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nAll results saved to {out_json}")
    
    h2o.cluster().shutdown()

if __name__ == "__main__":
    main()
