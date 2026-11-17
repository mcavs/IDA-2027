import urllib.request, json, io, time, os
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

h2o.init(nthreads=-1, max_mem_size="8G")
h2o.no_progress()

datasets = [
    {"name": "diabetes", "did": 37, "features": ["plas", "mass", "age", "insu", "pedi"]},
    {"name": "credit-g", "did": 31, "features": ["duration", "credit_amount", "age"]},
    {"name": "bank-marketing", "did": 1461, "features": ["age", "duration", "campaign", "balance"]},
    {"name": "blood-transfusion", "did": 1464, "features": ["V1", "V2", "V3", "V4"]},
    {"name": "pc3", "did": 1050, "features": ["LOC_BLANK", "BRANCH_COUNT", "LOC_TOTAL", "CYCLOMATIC_COMPLEXITY", "DESIGN_COMPLEXITY"]},
    {"name": "churn", "did": 40701, "features": ["total_day_minutes", "total_eve_minutes", "customer_service_calls", "account_length"]},
    {"name": "adult", "did": 1590, "features": ["age", "education-num", "hours-per-week", "capital-gain"]}
]

candidates = []

for ds in datasets:
    name = ds["name"]
    did = ds["did"]
    print(f"\nEvaluating dataset: {name} (did={did})...")
    try:
        df = get_arff_data(did)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        continue
    
    target_col = df.columns[-1]
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=42).reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    hf = h2o.H2OFrame(df)
    hf[target_col] = hf[target_col].asfactor()
    features = [c for c in df.columns if c != target_col]
    
    aml = H2OAutoML(max_models=30, max_runtime_secs=90, seed=42, exclude_algos=["StackedEnsemble", "DeepLearning"], sort_metric="logloss")
    aml.train(x=features, y=target_col, training_frame=hf)
    lb = aml.leaderboard.as_data_frame()
    best_ll = lb['logloss'].min()
    
    # eps = 0.10 to allow genuine multiplicity
    rash_lb = lb[lb['logloss'] <= best_ll * 1.10]
    K = len(rash_lb)
    print(f"[{name}] Total models: {len(lb)}, Rashomon pool (eps=0.10): {K}")
    if K < 4:
        continue
        
    models = [h2o.get_model(mid) for mid in rash_lb['model_id']]
    bg_df = df.sample(min(200, len(df)), random_state=42).copy()
    
    for feat in ds["features"]:
        if feat not in df.columns or not np.issubdtype(df[feat].dtype, np.number):
            continue
        grid_vals = np.percentile(df[feat].dropna(), np.linspace(10, 90, 20))
        grid_vals = np.unique(grid_vals)
        if len(grid_vals) < 10:
            continue
            
        pdp_mat = []
        for m in models:
            preds_grid = []
            for v in grid_vals:
                temp_df = bg_df.copy()
                temp_df[feat] = v
                temp_hf = h2o.H2OFrame(temp_df)
                p = m.predict(temp_hf).as_data_frame()
                prob_col = p.columns[-1] if 'p1' not in p.columns else 'p1'
                preds_grid.append(float(p[prob_col].mean()))
                h2o.remove(temp_hf)
            pdp_mat.append(preds_grid)
            
        P = np.array(pdp_mat) # shape: (K, len(grid))
        b1 = P[0]
        s_avg = P.mean(axis=0)
        p_med = np.median(P, axis=0)
        
        # Distances
        d_b1_savg = np.mean(np.abs(b1 - s_avg))
        d_b1_pmed = np.mean(np.abs(b1 - p_med))
        d_savg_pmed = np.mean(np.abs(s_avg - p_med))
        
        # Max divergence of b1 from ALL others
        d_b1_others = np.mean([np.mean(np.abs(b1 - P[i])) for i in range(1, K)])
        
        # Correlation / shape difference: 1 - corr(b1, s_avg)
        corr_b1_savg = np.corrcoef(b1, s_avg)[0, 1] if np.std(b1) > 1e-6 and np.std(s_avg) > 1e-6 else 1.0
        corr_savg_pmed = np.corrcoef(s_avg, p_med)[0, 1] if np.std(s_avg) > 1e-6 and np.std(p_med) > 1e-6 else 1.0
        
        candidates.append({
            "dataset": name,
            "feature": feat,
            "K": K,
            "d_b1_savg": float(d_b1_savg),
            "d_b1_others": float(d_b1_others),
            "d_savg_pmed": float(d_savg_pmed),
            "corr_b1_savg": float(corr_b1_savg),
            "corr_savg_pmed": float(corr_savg_pmed),
            "pdp_spread": float(np.mean(np.std(P, axis=0)))
        })
        print(f"  Feature '{feat}': d(B1, SAvg)={d_b1_savg:.4f}, d(B1, others)={d_b1_others:.4f}, d(SAvg, PMed)={d_savg_pmed:.4f}, corr(B1,SAvg)={corr_b1_savg:.3f}, spread={np.mean(np.std(P, axis=0)):.4f}")

h2o.cluster().shutdown()

with open("outputs/real_world_screening.json", "w") as f:
    json.dump(candidates, f, indent=2)
