import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from scipy.stats import trim_mean, rankdata
import h2o
from h2o.automl import H2OAutoML

MAX_MODELS = 160
MAX_RUNTIME = 2880
EPS = 0.05
GRID_POINTS = 20
SAMPLE_PDP_BG = 100

OUT_DIR = "outputs/real_world_5datasets"
os.makedirs(OUT_DIR, exist_ok=True)

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

DATASET_CONFIGS = [
    {
        "name": "adult",
        "csv": "data/real_world/adult.csv",
        "target": "target",
        "drop_cols": ["fnlwgt"],
        "cont_features": ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    },
    {
        "name": "german_credit",
        "csv": "data/real_world/german_credit.csv",
        "target": "target",
        "drop_cols": [],
        "cont_features": ["duration", "credit_amount", "age"]
    },
    {
        "name": "bank_marketing",
        "csv": "data/real_world/bank_marketing.csv",
        "target": "target",
        "drop_cols": [],
        "cont_features": ["V1", "V6", "V10", "V12", "V13", "V14", "V15"]
    },
    {
        "name": "churn",
        "csv": "data/real_world/churn.csv",
        "target": "target",
        "drop_cols": ["phone_number", "state", "area_code"],
        "cont_features": [
            "account_length", "number_vmail_messages", "total_day_minutes",
            "total_day_calls", "total_day_charge", "total_eve_minutes",
            "total_eve_calls", "total_eve_charge", "total_night_minutes",
            "total_night_calls", "total_night_charge", "total_intl_minutes",
            "total_intl_calls", "total_intl_charge", "number_customer_service_calls"
        ]
    },
    {
        "name": "diabetes",
        "csv": "data/real_world/diabetes.csv",
        "target": "target",
        "drop_cols": [],
        "cont_features": ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
    }
]

def compute_pdp_vector(model, bg_df, feature, vals):
    """Vectors PDP using batched evaluation frame"""
    gp = len(vals)
    rows = [bg_df.copy().assign(**{feature: v}) for v in vals]
    big_df = pd.concat(rows, ignore_index=True)
    hf_big = h2o.H2OFrame(big_df)
    try:
        pred_df = model.predict(hf_big).as_data_frame()
        # Find positive class column
        prob_col = 'p1' if 'p1' in pred_df.columns else pred_df.columns[-1]
        probs = pred_df[prob_col].values
        pdp_val = probs.reshape(gp, len(bg_df)).mean(axis=1)
    finally:
        h2o.remove(hf_big)
    return pdp_val

def run_experiment_for_dataset(cfg):
    name = cfg["name"]
    res_path = os.path.join(OUT_DIR, f"{name}_results.json")
    if os.path.exists(res_path):
        print(f"[{name}] Already completed! Loading {res_path}...")
        with open(res_path, "r") as f:
            return json.load(f)
            
    print(f"\n========================================================")
    print(f"STARTING EXPERIMENT: {name} (AutoML Budget: max_models={MAX_MODELS}, max_runtime_secs={MAX_RUNTIME})")
    print(f"========================================================")
    
    df = pd.read_csv(cfg["csv"])
    if cfg["drop_cols"]:
        df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    
    df = df.dropna().reset_index(drop=True)
    target_col = cfg["target"]
    
    # Subsample if large for reliable H2O training frame
    if len(df) > 5000:
        print(f"[{name}] Subsampling {len(df)} rows to 5000 rows for AutoML training...")
        train_df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    else:
        train_df = df.copy()
        
    hf = h2o.H2OFrame(train_df)
    hf[target_col] = hf[target_col].asfactor()
    feat_cols = [c for c in train_df.columns if c != target_col]
    
    splits = hf.split_frame(ratios=[0.8], seed=42)
    tr = splits[0]
    te = splits[1]
    
    t0 = time.time()
    aml = H2OAutoML(
        max_models=MAX_MODELS,
        max_runtime_secs=MAX_RUNTIME,
        seed=42,
        sort_metric="logloss",
        exclude_algos=["StackedEnsemble", "DeepLearning"]
    )
    aml.train(x=feat_cols, y=target_col, training_frame=tr, leaderboard_frame=te)
    elapsed = time.time() - t0
    
    lb_df = aml.leaderboard.as_data_frame()
    lb = lb_df[['model_id', 'logloss']].dropna()
    if 'auc' in lb_df.columns:
        lb['auc'] = lb_df['auc']
    else:
        lb['auc'] = np.nan
        
    print(f"[{name}] AutoML finished in {elapsed:.1f}s. Evaluated {len(lb)} models.")
    best_ll = float(lb['logloss'].min())
    print(f"[{name}] Best logloss: {best_ll:.4f}")
    
    # Rashomon set (eps=0.05)
    rash_lb = lb[lb['logloss'] <= best_ll * (1.0 + EPS)].copy()
    if len(rash_lb) == 0:
        rash_lb = lb.iloc[[0]]
    K = len(rash_lb)
    rho = float(K / len(lb))
    print(f"[{name}] Rashomon set size (|R_eps| at eps={EPS}): {K} of {len(lb)} models (rho={rho:.3f})")
    
    # Clean memory of frames not needed
    h2o.remove(tr)
    h2o.remove(te)
    h2o.remove(hf)
    
    # Prepare background frame
    bg_df = train_df.sample(min(SAMPLE_PDP_BG, len(train_df)), random_state=42).copy()
    
    # Load all models in Rashomon set
    rash_models = []
    loglosses = []
    for idx, row in rash_lb.iterrows():
        mid = row['model_id']
        try:
            m = h2o.get_model(mid)
            rash_models.append(m)
            loglosses.append(float(row['logloss']))
        except Exception as e:
            print(f"Warning: could not load model {mid}: {e}")
            
    print(f"[{name}] Successfully loaded {len(rash_models)} models for PDP extraction.")
    
    dataset_results = {
        "dataset_name": name,
        "n_rows": len(df),
        "total_models_evaluated": len(lb),
        "K": len(rash_models),
        "rho": float(len(rash_models) / len(lb)),
        "best_logloss": best_ll,
        "features": {}
    }
    
    for feat in cfg["cont_features"]:
        if feat not in df.columns:
            continue
        vals = np.percentile(df[feat].dropna(), np.linspace(5, 95, GRID_POINTS))
        vals = np.unique(vals)
        if len(vals) < GRID_POINTS:
            vals = np.linspace(df[feat].min(), df[feat].max(), GRID_POINTS)
            
        pdp_curves = []
        for m in rash_models:
            pdp_vec = compute_pdp_vector(m, bg_df, feat, vals)
            pdp_curves.append(pdp_vec)
            
        P = np.array(pdp_curves) # shape: (K, m)
        mean_pdp = P.mean(axis=0)
        
        # Empirical profile ambiguity
        diff_sq = (P - mean_pdp[None, :]) ** 2
        ambiguity_j = float(diff_sq.mean())
        
        curves_dict = {
            "B1 (Single Best)": P[0].tolist(),
            "B0 (Simple Avg)": agg_simple(P).tolist(),
            "B2 (Perf Weighted)": agg_perf(P, loglosses).tolist(),
            "M1 (Softmax BMA)": agg_softmax_bma(P, loglosses).tolist(),
            "M2 (Pointwise Median)": agg_pointwise_median(P).tolist(),
            "M3 (Trimmed Mean)": agg_trimmed_mean(P).tolist(),
            "M4 (Fréchet Median)": agg_functional_median(P).tolist(),
            "M5 (Pairwise Alignment)": agg_pairwise_alignment(P).tolist()
        }
        
        dataset_results["features"][feat] = {
            "feature_name": feat,
            "grid_vals": vals.tolist(),
            "ambiguity": ambiguity_j,
            "curves": curves_dict
        }
        print(f"  Feature '{feat}': Ambiguity A_j = {ambiguity_j:.6f}")
        
    with open(res_path, "w") as f:
        json.dump(dataset_results, f, indent=2)
        
    return dataset_results

def main():
    h2o.init(nthreads=-1, max_mem_size="10G")
    h2o.no_progress()
    
    all_results = {}
    for cfg in DATASET_CONFIGS:
        res = run_experiment_for_dataset(cfg)
        all_results[cfg["name"]] = res
        with open(os.path.join(OUT_DIR, "real_world_5datasets_master.json"), "w") as f:
            json.dump(all_results, f, indent=2)
            
    print("\nALL 5 DATASET EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
