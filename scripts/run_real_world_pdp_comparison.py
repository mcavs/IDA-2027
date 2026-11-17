import os, time, pickle, json
import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from scipy.stats import trim_mean, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Start H2O
h2o.init(max_mem_size="2G", nthreads=-1)

# 2. Aggregation functions
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

def compute_pdp_for_model(model, df, feature, grid_points):
    gp = len(grid_points)
    rows = [df.copy().assign(**{feature: v}) for v in grid_points]
    big = pd.concat(rows, ignore_index=True)
    hf_big = h2o.H2OFrame(big)
    preds = model.predict(hf_big).as_data_frame()["predict"].values
    return preds.reshape(gp, len(df)).mean(axis=1)

def run_dataset(name, df_raw, target_col, feature_col, eps=0.05, max_models=20, max_runtime_secs=60):
    print(f"\n--- Running {name} (Feature: {feature_col}, Target: {target_col}) ---")
    hf = h2o.H2OFrame(df_raw)
    x_cols = [c for c in df_raw.columns if c != target_col]
    aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=42)
    aml.train(x=x_cols, y=target_col, training_frame=hf)
    
    lb = aml.leaderboard.as_data_frame()
    best_rmse = lb["rmse"].iloc[0]
    print(f"Top model RMSE: {best_rmse:.4f}")
    
    # Rashomon set (tolerance eps)
    rash_df = lb[lb["rmse"] <= best_rmse * (1.0 + eps)]
    print(f"Rashomon models (eps={eps}): {len(rash_df)} / {len(lb)}")
    
    # If fewer than 2 models meet eps=0.05, relax slightly so we have an ensemble to aggregate
    if len(rash_df) < 3:
        rash_df = lb.head(5)
        print(f"Adjusted Rashomon set to top {len(rash_df)} models for aggregation illustration")
        
    grid_vals = np.linspace(df_raw[feature_col].quantile(0.05), df_raw[feature_col].quantile(0.95), 20)
    
    # Compute PDP for all models in Rashomon set
    pdps = []
    rmses = []
    for m_id in rash_df["model_id"].values:
        m = h2o.get_model(m_id)
        pdp_curve = compute_pdp_for_model(m, df_raw, feature_col, grid_vals)
        pdps.append(pdp_curve)
        rmses.append(float(rash_df[rash_df["model_id"] == m_id]["rmse"].iloc[0]))
        
    P = np.array(pdps)
    rmses = np.array(rmses)
    
    best_model_id = lb["model_id"].iloc[0]
    best_m = h2o.get_model(best_model_id)
    best_pdp = compute_pdp_for_model(best_m, df_raw, feature_col, grid_vals)
    
    results = {
        "grid_vals": grid_vals,
        "best_pdp": best_pdp,
        "P": P,
        "rmses": rmses,
        "aggregations": {
            "Simple Average": agg_simple(P),
            "Performance-Weighted": agg_perf(P, rmses=rmses),
            "Softmax BMA": agg_softmax_bma(P, rmses=rmses),
            "Pointwise Median": agg_pointwise_median(P),
            "Trimmed Mean": agg_trimmed_mean(P),
            "Functional Median": agg_functional_median(P),
            "Alignment Consensus": agg_ava_fast(P)
        }
    }
    return results

# 1. Energy Efficiency
df_ee = pd.read_csv("/tmp/energy_efficiency.csv").drop(columns=["cooling_load"])
res_ee = run_dataset("Energy Efficiency", df_ee, target_col="heating_load", feature_col="surface_area")

# 2. Forest Fires
df_ff = pd.read_csv("/tmp/forestfires.csv")
res_ff = run_dataset("Forest Fires", df_ff, target_col="area", feature_col="temp")

h2o.cluster().shutdown()

# Save results
with open("outputs/real_world_pdp_results.pkl", "wb") as f:
    pickle.dump({"energy_efficiency": res_ee, "forest_fires": res_ff}, f)
print("\nReal world PDP results saved successfully to outputs/real_world_pdp_results.pkl")
