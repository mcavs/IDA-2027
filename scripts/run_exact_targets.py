import urllib.request, json, io, os
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from scipy.stats import trim_mean, rankdata

def get_arff_data(did):
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

h2o.init(nthreads=-1, max_mem_size="8G")
h2o.no_progress()

def analyze_feature(df, target_col, feature_name, eps=0.10, max_models=25, max_runtime_secs=60):
    df_clean = df.dropna(subset=[target_col, feature_name]).reset_index(drop=True)
    if len(df_clean) > 2500:
        df_clean = df_clean.sample(2500, random_state=42).reset_index(drop=True)
        
    hf = h2o.H2OFrame(df_clean)
    hf[target_col] = hf[target_col].asfactor()
    features = [c for c in df_clean.columns if c != target_col]
    
    aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=42, 
                    exclude_algos=["StackedEnsemble", "DeepLearning"], sort_metric="logloss")
    aml.train(x=features, y=target_col, training_frame=hf)
    lb = aml.leaderboard.as_data_frame()
    best_ll = lb['logloss'].min()
    
    rash_lb = lb[lb['logloss'] <= best_ll * (1.0 + eps)]
    if len(rash_lb) < 3:
        rash_lb = lb.head(5)
        
    K = len(rash_lb)
    models = [h2o.get_model(mid) for mid in rash_lb['model_id']]
    logloss_list = rash_lb['logloss'].tolist()
    
    grid_vals = np.percentile(df_clean[feature_name].dropna(), np.linspace(5, 95, 20))
    grid_vals = np.unique(grid_vals)
    
    bg_df = df_clean.sample(min(200, len(df_clean)), random_state=42).copy()
    
    pdp_list = []
    for m in models:
        curve = []
        for v in grid_vals:
            sub = bg_df.copy()
            sub[feature_name] = v
            shf = h2o.H2OFrame(sub)
            preds = m.predict(shf).as_data_frame(use_multi_thread=True)
            prob_col = preds.columns[-1] if 'p1' not in preds.columns else 'p1'
            curve.append(float(preds[prob_col].mean()))
            h2o.remove(shf)
        pdp_list.append(curve)
        
    P = np.array(pdp_list)
    b1 = P[0]
    
    curves = {
        "B1 (Single Best)": b1.tolist(),
        "B0 (Simple Avg)": agg_simple(P).tolist(),
        "B2 (Perf Weighted)": agg_perf(P, logloss_list).tolist(),
        "M1 (Softmax BMA)": agg_softmax_bma(P, logloss_list).tolist(),
        "M2 (Pointwise Median)": agg_pointwise_median(P).tolist(),
        "M3 (Trimmed Mean)": agg_trimmed_mean(P).tolist(),
        "M4 (Fréchet Median)": agg_functional_median(P).tolist(),
        "M5 (Pairwise Alignment)": agg_pairwise_alignment(P).tolist()
    }
    
    h2o.remove(hf)
    return {
        "grid_vals": grid_vals.tolist(),
        "K": K,
        "best_ll": float(best_ll),
        "curves": curves
    }

results = {}
os.makedirs("outputs/real_world", exist_ok=True)

# 1. Diabetes (insu)
print("Running diabetes: insu...")
df_diab = get_arff_data(37)
target_diab = df_diab.columns[-1]
results["diabetes_insu"] = analyze_feature(df_diab, target_diab, 'insu', eps=0.10)
with open("outputs/real_world/target_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved diabetes: insu!")

# 2. Credit-g (duration)
print("Running credit-g: duration...")
df_credit = get_arff_data(31)
target_credit = df_credit.columns[-1]
results["credit_duration"] = analyze_feature(df_credit, target_credit, 'duration', eps=0.12)
with open("outputs/real_world/target_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved credit-g: duration!")

# 3. Churn (customer_service_calls)
print("Running churn: customer_service_calls...")
df_churn = get_arff_data(40701)
target_churn = df_churn.columns[-1]
results["churn_csc"] = analyze_feature(df_churn, target_churn, 'customer_service_calls', eps=0.10)
with open("outputs/real_world/target_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved churn: customer_service_calls!")

print("\nALL TARGET RUNS COMPLETE!")
h2o.cluster().shutdown()
