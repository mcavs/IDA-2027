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
def agg_simple(P, **_): return P.mean(axis=0)
def agg_perf(P, loglosses, **_):
    ll = np.array(loglosses, dtype=float)
    w = 1.0 / (ll + 1e-12)
    w /= w.sum()
    return (P * w[:, None]).sum(axis=0)
def agg_softmax_bma(P, loglosses, beta=2.0, **_):
    ll = np.array(loglosses, dtype=float)
    ptp = np.ptp(ll)
    if ptp < 1e-12: return P.mean(axis=0)
    w = np.exp(-beta * (ll - np.min(ll)) / ptp)
    w /= w.sum()
    return (P * w[:, None]).sum(axis=0)
def agg_pointwise_median(P, **_): return np.median(P, axis=0)
def agg_trimmed_mean(P, alpha=0.1, **_):
    if len(P) < 5: return P.mean(axis=0)
    return trim_mean(P, proportiontocut=alpha, axis=0)
def agg_functional_median(P, **_):
    if len(P) <= 2: return P.mean(axis=0)
    dists = np.sum(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1), axis=1)
    return P[np.argmin(dists)]
def agg_pairwise_alignment(P, **_):
    k, m = P.shape
    if k <= 1: return P[0]
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

h2o.init(nthreads=-1)
h2o.no_progress()

def run_case(did, col_name, eps=0.10):
    df = get_arff_data(did)
    target = df.columns[-1]
    df_clean = df.dropna(subset=[target, col_name]).reset_index(drop=True)
    if len(df_clean) > 2000:
        df_clean = df_clean.sample(2000, random_state=42).reset_index(drop=True)
    hf = h2o.H2OFrame(df_clean)
    hf[target] = hf[target].asfactor()
    aml = H2OAutoML(max_models=20, max_runtime_secs=40, seed=42, exclude_algos=['StackedEnsemble', 'DeepLearning'], sort_metric='logloss')
    aml.train(x=[c for c in df_clean.columns if c != target], y=target, training_frame=hf)
    lb = aml.leaderboard.as_data_frame()
    best_ll = lb['logloss'].min()
    rash_lb = lb[lb['logloss'] <= best_ll * (1.0 + eps)]
    if len(rash_lb) < 4: rash_lb = lb.head(5)
    models = [h2o.get_model(mid) for mid in rash_lb['model_id']]
    loglosses = rash_lb['logloss'].tolist()
    
    pdps = []
    grid_vals = None
    for m in models:
        res = m.partial_plot(frame=hf, cols=[col_name], plot=False, nbins=20)
        df_pdp = res[0].as_data_frame()
        if grid_vals is None:
            grid_vals = df_pdp.iloc[:, 0].values.tolist()
        mean_col = [c for c in df_pdp.columns if 'mean_response' in c][0]
        pdps.append(df_pdp[mean_col].values)
    P = np.array(pdps)
    b1 = P[0]
    curves = {
        'B1 (Single Best)': b1.tolist(),
        'B0 (Simple Avg)': agg_simple(P).tolist(),
        'B2 (Perf Weighted)': agg_perf(P, loglosses).tolist(),
        'M1 (Softmax BMA)': agg_softmax_bma(P, loglosses).tolist(),
        'M2 (Pointwise Median)': agg_pointwise_median(P).tolist(),
        'M3 (Trimmed Mean)': agg_trimmed_mean(P).tolist(),
        'M4 (Fréchet Median)': agg_functional_median(P).tolist(),
        'M5 (Pairwise Alignment)': agg_pairwise_alignment(P).tolist()
    }
    h2o.remove(hf)
    return {'grid_vals': grid_vals, 'K': len(models), 'best_ll': float(best_ll), 'curves': curves}

print("Computing Blood Transfusion V2 (Frequency)...")
res_blood_v2 = run_case(1464, 'V2', eps=0.10)

print("Computing ILPD V3 (Total Bilirubin)...")
res_ilpd_v3 = run_case(1480, 'V3', eps=0.10)

print("Computing ILPD V10 (Albumin/Globulin Ratio)...")
res_ilpd_v10 = run_case(1480, 'V10', eps=0.10)

print("Computing Diabetes Insu...")
res_diab_insu = run_case(37, 'insu', eps=0.10)

all_res = {
    'blood_v2': res_blood_v2,
    'ilpd_v3': res_ilpd_v3,
    'ilpd_v10': res_ilpd_v10,
    'diab_insu': res_diab_insu
}

with open('outputs/real_world/candidate_eval_curves.json', 'w') as f:
    json.dump(all_res, f, indent=2)

print("\nSuccessfully computed and saved all candidate curves!")
h2o.cluster().shutdown()
