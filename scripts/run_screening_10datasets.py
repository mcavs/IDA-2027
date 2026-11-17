import os, json, time
import numpy as np
import pandas as pd
from scipy.stats import trim_mean, rankdata
import h2o
from h2o.automl import H2OAutoML

h2o.init(nthreads=-1, max_mem_size="8G")
h2o.no_progress()

OUT_DIR = "outputs/real_world_screening"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    ("blood_transfusion", "data/real_world_10new/blood_transfusion.csv"),
    ("ilpd_liver", "data/real_world_10new/ilpd_liver.csv"),
    ("australian", "data/real_world_10new/australian.csv"),
    ("heart_statlog", "data/real_world_10new/heart_statlog.csv"),
    ("wdbc", "data/real_world_10new/wdbc.csv"),
    ("climate_crashes", "data/real_world_10new/climate_crashes.csv"),
    ("kc1", "data/real_world_10new/kc1.csv"),
    ("wilt", "data/real_world_10new/wilt.csv"),
    ("phoneme", "data/real_world_10new/phoneme.csv"),
    ("spambase", "data/real_world_10new/spambase.csv")
]

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

def compute_pdp_vector(model, bg_df, feature, vals):
    gp = len(vals)
    rows = [bg_df.copy().assign(**{feature: v}) for v in vals]
    big_df = pd.concat(rows, ignore_index=True)
    hf_big = h2o.H2OFrame(big_df)
    try:
        pred_df = model.predict(hf_big).as_data_frame()
        prob_col = 'p1' if 'p1' in pred_df.columns else pred_df.columns[-1]
        probs = pred_df[prob_col].values
        pdp_val = probs.reshape(gp, len(bg_df)).mean(axis=1)
    finally:
        h2o.remove(hf_big)
    return pdp_val

results = {}
for name, csv_path in DATASETS:
    out_file = os.path.join(OUT_DIR, f"{name}.json")
    if os.path.exists(out_file):
        print(f"[{name}] Already exists, loading...")
        with open(out_file) as f:
            results[name] = json.load(f)
        continue

    print(f"\n==========================================")
    print(f"RUNNING: {name} ({csv_path})")
    print(f"==========================================")
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    if len(df) > 3000:
        train_df = df.sample(n=3000, random_state=42).reset_index(drop=True)
    else:
        train_df = df.copy()

    hf = h2o.H2OFrame(train_df)
    hf['target'] = hf['target'].asfactor()
    feat_cols = [c for c in train_df.columns if c != 'target']
    
    splits = hf.split_frame(ratios=[0.8], seed=42)
    tr, te = splits[0], splits[1]

    aml = H2OAutoML(
        max_models=50,
        max_runtime_secs=300,
        seed=42,
        sort_metric="logloss",
        exclude_algos=["StackedEnsemble", "DeepLearning"]
    )
    aml.train(x=feat_cols, y="target", training_frame=tr, leaderboard_frame=te)
    lb_df = aml.leaderboard.as_data_frame()
    lb = lb_df[['model_id', 'logloss']].dropna()
    best_ll = float(lb['logloss'].min())

    rash_lb = lb[lb['logloss'] <= best_ll * 1.05].copy()
    if len(rash_lb) < 2:
        rash_lb = lb.head(3)

    K = len(rash_lb)
    print(f"[{name}] Total models: {len(lb)}, Rashomon set (eps=0.05): {K}")

    rash_models = []
    loglosses = []
    for _, row in rash_lb.iterrows():
        try:
            m = h2o.get_model(row['model_id'])
            rash_models.append(m)
            loglosses.append(float(row['logloss']))
        except Exception:
            pass

    num_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c != 'target']
    # If too many numeric features (e.g. spambase, wdbc), select first 8
    test_feats = num_cols[:8]

    bg_df = train_df.sample(min(100, len(train_df)), random_state=42).copy()

    d_res = {
        "dataset": name,
        "n": len(df),
        "total_models": len(lb),
        "K": len(rash_models),
        "best_ll": best_ll,
        "features": {}
    }

    for feat in test_feats:
        vals = np.percentile(df[feat].dropna(), np.linspace(5, 95, 20))
        vals = np.unique(vals)
        if len(vals) < 20:
            vals = np.linspace(df[feat].min(), df[feat].max(), 20)

        pdp_curves = []
        for m in rash_models:
            pdp_vec = compute_pdp_vector(m, bg_df, feat, vals)
            pdp_curves.append(pdp_vec)

        P = np.array(pdp_curves)
        mean_pdp = P.mean(axis=0)
        ambiguity = float(((P - mean_pdp[None, :]) ** 2).mean())

        b1_curve = P[0]
        b0_curve = agg_simple(P)
        max_b1_b0 = float(np.max(np.abs(b1_curve - b0_curve)))

        curves_dict = {
            "B1 (Single Best)": P[0].tolist(),
            "B0 (Simple Avg)": b0_curve.tolist(),
            "B2 (Perf Weighted)": agg_perf(P, loglosses).tolist(),
            "M1 (Softmax BMA)": agg_softmax_bma(P, loglosses).tolist(),
            "M2 (Pointwise Median)": agg_pointwise_median(P).tolist(),
            "M3 (Trimmed Mean)": agg_trimmed_mean(P).tolist(),
            "M4 (Fréchet Median)": agg_functional_median(P).tolist(),
            "M5 (Pairwise Alignment)": agg_pairwise_alignment(P).tolist()
        }

        d_res["features"][feat] = {
            "ambiguity": ambiguity,
            "max_b1_b0": max_b1_b0,
            "grid_vals": vals.tolist(),
            "curves": curves_dict
        }
        print(f"  {feat:20s}: A={ambiguity:.6f}, max|B1-B0|={max_b1_b0:.4f}")

    h2o.remove(hf)
    h2o.remove(tr)
    h2o.remove(te)

    with open(out_file, "w") as f:
        json.dump(d_res, f, indent=2)
    results[name] = d_res

print("\nALL SCREENING COMPLETED!")
h2o.cluster().shutdown()
