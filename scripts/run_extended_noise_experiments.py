"""
Tum Gurultu Seviyelerini Genisletme Scripti (sigma^2 in {2, 3, 5, 6, 7, 8}):
- 3 Setup x 6 Yeni Gurultu Seviyesi = 18 Yeni Senaryo
- Tam Butce: max_models=160, max_runtime_secs=2880 (48 dakika)
- Mevcut olanlar atlanir (resume / idempotent calisma)
- Sonuclar outputs/pdps/ ve outputs/models/ altina pkl olarak aninda kaydedilir.
"""
import warnings; warnings.filterwarnings("ignore")
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
import json, time, os, pickle

from analysis import SETUP_FNS, compute_pdp, OUT

EPS_LIST = [0.01, 0.05, 0.10, 0.20]
MAX_MODELS = 160
MAX_RUNTIME = 2880

PDP_DIR = os.path.join(OUT, "pdps")
MODEL_DIR = os.path.join(OUT, "models")
os.makedirs(PDP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_single_scenario(snum, nv):
    scenario_key = f"S{snum}_nv{nv}"
    out_pkl = os.path.join(PDP_DIR, f"{scenario_key}_pdp_data.pkl")
    
    if os.path.exists(out_pkl):
        print(f"[{scenario_key}] zaten mevcut ({out_pkl}), atlanıyor.")
        return

    print(f"\n{'='*60}")
    print(f"BAŞLIYOR: {scenario_key} (Setup {snum}, sigma^2 = {nv})")
    print(f"Bütçe: {MAX_MODELS} model, {MAX_RUNTIME}s (48 dk)")
    print(f"{'='*60}")
    t0 = time.time()

    df, feat, gt_fn = SETUP_FNS[snum](nv=nv)

    # AutoML eğitimi
    hf = h2o.H2OFrame(df)
    feat_cols = [c for c in df.columns if c != "y"]
    tr, _ = hf.split_frame(ratios=[0.8], seed=42)

    aml = H2OAutoML(
        max_models=MAX_MODELS,
        max_runtime_secs=MAX_RUNTIME,
        seed=42,
        sort_metric="RMSE",
        exclude_algos=["StackedEnsemble", "DeepLearning"]
    )
    aml.train(x=feat_cols, y="y", training_frame=tr)
    lb_df = aml.leaderboard.as_data_frame()
    lb = lb_df[["model_id", "rmse"]].dropna()
    print(f"[{scenario_key}] AutoML tamamlandı ({len(lb)} model eğitildi, süre: {time.time()-t0:.1f}s)")

    df_tr = tr.as_data_frame()
    all_models = []
    all_rmses = []
    model_ids = []

    sc_model_dir = os.path.join(MODEL_DIR, scenario_key)
    os.makedirs(sc_model_dir, exist_ok=True)

    for _, row in lb.iterrows():
        mid = row["model_id"]
        try:
            m = h2o.get_model(mid)
            all_models.append(m)
            all_rmses.append(float(row["rmse"]))
            model_ids.append(mid)
            # Model kaydet
            m_path = os.path.join(sc_model_dir, mid)
            if not os.path.exists(m_path):
                h2o.save_model(m, path=sc_model_dir, force=True)
        except Exception as e:
            continue

    if not all_models:
        print(f"[{scenario_key}] HATA: Hiç model yüklenemedi!")
        return

    # PDP hesaplama
    vals = np.linspace(df[feat].min(), df[feat].max(), 50)
    print(f"[{scenario_key}] {len(all_models)} model için PDP hesaplanıyor...")
    
    pdp_dict = {}
    pdp_list = []
    for m in all_models:
        p = compute_pdp(m, df_tr, feat, vals)
        pdp_dict[m.model_id] = p
        pdp_list.append(p)

    pdp_all = np.array(pdp_list)
    best_pdp = pdp_all[0] # Leaderboard 1.si

    # Ground truth ve centering
    gt_raw = gt_fn(vals)
    gt_fixed = gt_raw - np.mean(gt_raw) + np.mean(best_pdp)
    b1_md_fixed = float(np.mean(np.abs(best_pdp - gt_fixed)))

    # PKL kaydet
    scenario_data = {
        "scenario": scenario_key,
        "setup": snum,
        "noise_var": nv,
        "feature": feat,
        "vals": vals,
        "model_ids": model_ids,
        "all_rmses": all_rmses,
        "pdp_dict": pdp_dict,
        "pdp_all": pdp_all,
        "best_pdp": best_pdp,
        "gt_raw": gt_raw,
        "gt_fixed": gt_fixed,
        "b1_md_fixed": b1_md_fixed,
        "leaderboard": lb.to_dict(orient="records")
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(scenario_data, f)

    elapsed = time.time() - t0
    print(f"[{scenario_key}] TAMAMLANDI ve KAYDEDİLDİ: {out_pkl} (Toplam Süre: {elapsed/60:.1f} dakika)")

def main():
    h2o.init(nthreads=-1, max_mem_size="8G")
    new_noise_vars = [2, 3, 5, 6, 7, 8]
    setups = [1, 2, 3]

    print("="*60)
    print("GENİŞLETİLMİŞ GÜRÜLTÜ DENEYLERİ BAŞLIYOR")
    print(f"Kapsanan Setuplar: {setups}")
    print(f"Yeni Gürültü Seviyeleri: {new_noise_vars}")
    print(f"Toplam Yeni Senaryo Sayısı: {len(setups) * len(new_noise_vars)}")
    print("="*60)

    for snum in setups:
        for nv in new_noise_vars:
            try:
                run_single_scenario(snum, nv)
            except Exception as e:
                print(f"HATA [Setup {snum}, nv {nv}]: {e}")
                continue

    h2o.cluster().shutdown()
    print("TÜM DENEYLER BAŞARIYLA TAMAMLANDI!")

if __name__ == "__main__":
    main()
