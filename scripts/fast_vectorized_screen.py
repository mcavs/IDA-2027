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

h2o.init(nthreads=-1, max_mem_size="8G")
h2o.no_progress()

datasets = [
    {"name": "diabetes", "did": 37, "features": ["plas", "mass", "insu", "age", "pedi"]},
    {"name": "churn", "did": 40701, "features": ["total_day_minutes", "customer_service_calls", "total_eve_minutes", "total_intl_calls"]},
    {"name": "pc3", "did": 1050, "features": ["LOC_BLANK", "BRANCH_COUNT", "LOC_TOTAL", "DESIGN_COMPLEXITY", "CYCLOMATIC_COMPLEXITY"]},
    {"name": "credit-g", "did": 31, "features": ["duration", "credit_amount", "age"]},
    {"name": "adult", "did": 1590, "features": ["age", "education-num", "hours-per-week", "capital-gain"]}
]

candidates = []

for ds in datasets:
    name = ds["name"]
    did = ds["did"]
    print(f"\nEvaluating dataset: {name}...")
    try:
        df = get_arff_data(did)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        continue
        
    target_col = df.columns[-1]
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    hf = h2o.H2OFrame(df)
    hf[target_col] = hf[target_col].asfactor()
    features = [c for c in df.columns if c != target_col]
    
    aml = H2OAutoML(max_models=25, max_runtime_secs=60, seed=42, exclude_algos=["StackedEnsemble", "DeepLearning"], sort_metric="logloss")
    aml.train(x=features, y=target_col, training_frame=hf)
    lb = aml.leaderboard.as_data_frame()
    best_ll = lb['logloss'].min()
    
    rash_lb = lb[lb['logloss'] <= best_ll * 1.10]
    K = len(rash_lb)
    print(f"[{name}] Total models: {len(lb)}, Rashomon pool: {K}")
    if K < 4:
        continue
        
    models = [h2o.get_model(mid) for mid in rash_lb['model_id']]
    bg_df = df.sample(min(150, len(df)), random_state=42).copy()
    n_bg = len(bg_df)
    
    for feat in ds["features"]:
        if feat not in df.columns or not np.issubdtype(df[feat].dtype, np.number):
            continue
        grid_vals = np.percentile(df[feat].dropna(), np.linspace(10, 90, 20))
        grid_vals = np.unique(grid_vals)
        if len(grid_vals) < 10:
            continue
        m_grid = len(grid_vals)
        
        # Build batch grid: duplicate bg_df for each grid val
        batch_dfs = []
        for v in grid_vals:
            sub = bg_df.copy()
            sub[feat] = v
            batch_dfs.append(sub)
        big_df = pd.concat(batch_dfs, ignore_index=True)
        big_hf = h2o.H2OFrame(big_df)
        
        pdp_mat = []
        for m in models:
            preds_h2o = m.predict(big_hf)
            prob_col = preds_h2o.columns[-1] if 'p1' not in preds_h2o.columns else 'p1'
            p1_vals = preds_h2o[prob_col].as_data_frame(use_multi_thread=True).values.flatten()
            # average across n_bg for each grid point
            pdp_curve = p1_vals.reshape(m_grid, n_bg).mean(axis=1)
            pdp_mat.append(pdp_curve)
            
        h2o.remove(big_hf)
        
        P = np.array(pdp_mat) # shape: (K, m_grid)
        b1 = P[0]
        s_avg = P.mean(axis=0)
        p_med = np.median(P, axis=0)
        
        # 1. d(B1, S_Avg) and d(B1, others)
        d_b1_savg = np.mean(np.abs(b1 - s_avg))
        d_b1_others = np.mean([np.mean(np.abs(b1 - P[i])) for i in range(1, K)])
        
        # 2. d(S_Avg, PMed) and shape differences
        d_savg_pmed = np.mean(np.abs(s_avg - p_med))
        d_b1_others_shape = 1.0 - (np.corrcoef(b1, s_avg)[0, 1] if np.std(b1)>1e-6 and np.std(s_avg)>1e-6 else 1.0)
        d_savg_pmed_shape = 1.0 - (np.corrcoef(s_avg, p_med)[0, 1] if np.std(s_avg)>1e-6 and np.std(p_med)>1e-6 else 1.0)
        
        spread = np.mean(np.std(P, axis=0))
        
        res_item = {
            "dataset": name,
            "feature": feat,
            "K": K,
            "grid_vals": grid_vals.tolist(),
            "b1": b1.tolist(),
            "s_avg": s_avg.tolist(),
            "p_med": p_med.tolist(),
            "P": P.tolist(),
            "loglosses": rash_lb['logloss'].tolist(),
            "d_b1_savg": float(d_b1_savg),
            "d_b1_others": float(d_b1_others),
            "d_savg_pmed": float(d_savg_pmed),
            "d_b1_others_shape": float(d_b1_others_shape),
            "d_savg_pmed_shape": float(d_savg_pmed_shape),
            "spread": float(spread)
        }
        candidates.append(res_item)
        print(f"  Feature '{feat:20s}': d(B1, others)={d_b1_others:.4f}, d(B1, SAvg)={d_b1_savg:.4f}, d(SAvg, PMed)={d_savg_pmed:.4f}, ShapeDiff(B1,SAvg)={d_b1_others_shape:.3f}, ShapeDiff(SAvg,PMed)={d_savg_pmed_shape:.3f}, Spread={spread:.4f}")

with open("outputs/real_world_candidates.json", "w") as f:
    json.dump(candidates, f)
print("\nScreening complete! Saved to outputs/real_world_candidates.json")
h2o.cluster().shutdown()
