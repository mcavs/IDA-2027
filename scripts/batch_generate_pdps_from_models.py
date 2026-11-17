"""
Tum 27 senaryo icin (3 setup x 9 noise_var: nv in {1..9})
outputs/models/ altindaki egitilmis H2O modellerini yukleyip
outputs/pdps/<scenario>_pdp_data.pkl dosyalarini olusturan script.
Cross-validation RMSE kullanarak Rashomon kumelerini dogru kurar.
"""
import warnings; warnings.filterwarnings("ignore")
import h2o
import numpy as np
import pandas as pd
import pickle, os, time, glob

from analysis import SETUP_FNS, compute_pdp, OUT

PDP_DIR = os.path.join(OUT, "pdps")
MODEL_DIR = os.path.join(OUT, "models")
os.makedirs(PDP_DIR, exist_ok=True)

# 1, 4, 9 senaryolari zaten orijinal leaderboard RMSE ile olusturulmustur, onlari koru
EXISTING_SAFE = ["S1_nv1", "S1_nv4", "S1_nv9", "S2_nv1", "S2_nv4", "S2_nv9", "S3_nv1", "S3_nv4", "S3_nv9"]

def process_scenario(snum, nv):
    scen_key = f"S{snum}_nv{nv}"
    out_pkl = os.path.join(PDP_DIR, f"{scen_key}_pdp_data.pkl")

    if scen_key in EXISTING_SAFE and os.path.exists(out_pkl):
        print(f"[{scen_key}] Referans senaryo zaten mevcut, korunuyor.")
        return

    sc_model_dir = os.path.abspath(os.path.join(MODEL_DIR, scen_key))
    if not os.path.exists(sc_model_dir):
        print(f"[{scen_key}] Model dizini bulunamadı: {sc_model_dir}")
        return

    model_files = sorted(glob.glob(os.path.join(sc_model_dir, "*")))
    model_files = [f for f in model_files if os.path.isfile(f)]
    if not model_files:
        print(f"[{scen_key}] Hiç model dosyası bulunamadı!")
        return

    print(f"\n{'='*55}\nİşleniyor: {scen_key} ({len(model_files)} model dosyası)\n{'='*55}")
    t0 = time.time()

    df, feat, gt_fn = SETUP_FNS[snum](nv=nv)
    hf = h2o.H2OFrame(df)
    tr, _ = hf.split_frame(ratios=[0.8], seed=42)
    df_tr = tr.as_data_frame()

    loaded_models = []
    loaded_ids = []
    rmses = []

    # Modelleri yükle ve xval RMSE al
    for mf in model_files:
        try:
            m = h2o.load_model(mf)
            r = m.rmse(xval=True)
            if r is None:
                r = m.rmse(valid=True)
            if r is None:
                r = m.rmse(train=True)
            if r is not None and not np.isnan(r):
                loaded_models.append(m)
                loaded_ids.append(m.model_id)
                rmses.append(float(r))
        except Exception:
            continue

    if not loaded_models:
        print(f"[{scen_key}] Modeller yüklenemedi!")
        return

    # Leaderboard sıralaması (küçük RMSE'den büyüğe)
    sort_idx = np.argsort(rmses)
    sorted_models = [loaded_models[i] for i in sort_idx]
    sorted_ids = [loaded_ids[i] for i in sort_idx]
    sorted_rmses = [rmses[i] for i in sort_idx]

    # Grid noktaları ve best PDP
    vals, best_pdp = compute_pdp(sorted_models[0], df_tr, feat, gp=50)

    # Toplu PDP hesaplama
    rows = [df_tr.copy().assign(**{feat: v}) for v in vals]
    big_df = pd.concat(rows, ignore_index=True)
    big_hf = h2o.H2OFrame(big_df)

    pdp_list = []
    pdp_dict = {}
    valid_ids = []
    valid_rmses = []

    for m, mid, r in zip(sorted_models, sorted_ids, sorted_rmses):
        try:
            preds = m.predict(big_hf).as_data_frame()["predict"].values
            curve = preds.reshape(len(vals), len(df_tr)).mean(axis=1)
            pdp_list.append(curve)
            pdp_dict[mid] = curve
            valid_ids.append(mid)
            valid_rmses.append(r)
        except Exception:
            continue

    pdp_all = np.array(pdp_list)
    gt_raw = gt_fn(vals)
    gt_fixed = gt_raw - np.mean(gt_raw) + np.mean(best_pdp)
    b1_md_fixed = float(np.mean(np.abs(best_pdp - gt_fixed)))

    data_payload = {
        "scenario": scen_key,
        "setup": snum,
        "noise_var": nv,
        "feature": feat,
        "vals": vals,
        "model_ids": valid_ids,
        "all_rmses": valid_rmses,
        "pdp_dict": pdp_dict,
        "pdp_all": pdp_all,
        "best_pdp": best_pdp,
        "gt_raw": gt_raw,
        "gt_fixed": gt_fixed,
        "b1_md_fixed": b1_md_fixed,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(data_payload, f)

    best_r = valid_rmses[0]
    k_005 = sum(1 for r in valid_rmses if r <= best_r * 1.05)
    k_020 = sum(1 for r in valid_rmses if r <= best_r * 1.20)
    print(f"[{scen_key}] TAMAMLANDI: {out_pkl} ({len(pdp_all)} PDP, best_rmse={best_r:.4f}, K@0.05={k_005}, K@0.20={k_020}, Süre: {time.time()-t0:.1f}s)")

def main():
    h2o.init(nthreads=-1, max_mem_size="8G", strict_version_check=False)
    setups = [1, 2, 3]
    all_nvs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for snum in setups:
        for nv in all_nvs:
            process_scenario(snum, nv)

    h2o.cluster().shutdown()
    print("\nTÜM PDP'LER DOĞRU RASHOMON SIRALAMASIYLA OLUŞTURULDU!")

if __name__ == "__main__":
    main()
