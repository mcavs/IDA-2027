"""
Rashomon PDP Aggregation — Siniflandirma Deneyleri Pipeline'i
==============================================================
- 3 Sentetik Setup x 9 Gurultu Seviyesi (sigma^2 in {1..9}) = 27 Senaryo
- Genisletilmis AutoML Butcesi: max_models=160, max_runtime_secs=2880 (48 dk)
  (Eger tum 160 model daha once tamamlanirsa sure dolmadan sonraki modele gecer)
- Siralama Metrigi: AUC (Rashomon toleransi: AUC >= (1 - eps) * AUC_max)
- Epsilon Değerleri: [0.01, 0.05, 0.10, 0.20]
- Kısmi Bağımlılık Profilleri (PDP): Sınıf-1 olasılığı (p1) üzerinden 20 grid noktası
- Ground Truth: Probit link entegrasyonu Phi((Y_latent - 0.5) / sigma)
- Idempotent / Resume Destegi: Her senaryonun sonuclari aninda outputs/cls_pdps/ altina kaydedilir.
"""
import warnings; warnings.filterwarnings("ignore")
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
import scipy.stats as stats
import json, time, os, pickle

EPS_LIST = [0.01, 0.05, 0.10, 0.20]
MAX_MODELS = 160
MAX_RUNTIME = 2880
GRID_POINTS = 20

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CLS_PDP = os.path.join(BASE_DIR, "outputs", "cls_pdps")
OUT_CLS_MODELS = os.path.join(BASE_DIR, "outputs", "cls_models")
os.makedirs(OUT_CLS_PDP, exist_ok=True)
os.makedirs(OUT_CLS_MODELS, exist_ok=True)

# --------------------------------------------------------------------------
# 1. VERİ ÜRETİMİ VE GROUND TRUTH FONKSİYONLARI
# --------------------------------------------------------------------------
def _base_features(n, rng):
    X1 = rng.normal(0, 1, n)
    X2 = rng.binomial(1, 0.5, n)
    X3 = rng.normal(0, 1, n)
    X4 = rng.uniform(-2, 2, n)
    X5 = rng.exponential(1, n)
    X6 = rng.choice(['A', 'B', 'C'], n)
    X7 = rng.normal(0, 1, n)
    X8 = rng.choice([0, 1], n)
    X9 = rng.normal(0, 1, n)
    return X1, X2, X3, X4, X5, X6, X7, X8, X9

def eval_latent_vector(X1, X2, X3, X4, X5, X6, X7, X8, X9, setup_num):
    if setup_num == 1:
        c6 = np.where(X6 == 'B', 2.0, np.where(X6 == 'C', -1.0, 0.0))
        return (2.9*X1 - 3.7*X2 + 1.2*X3 + 0.8*X1**2 - 1.5*X2*X3
                + 1.1*X4 - 0.5*np.log(X5 + 1) + c6 + 0.3*X9)
    elif setup_num == 2:
        c6 = np.where(X6 == 'A', -2.0, np.where(X6 == 'C', 1.0, 0.0))
        return (np.sin(X1) + np.log(np.abs(X3) + 1) + 0.8*X4*X5 - 1.2*X1*X3
                + c6 + 1.5*X7 + 2.0*(X8 == 1) + 0.4*X9**2)
    elif setup_num == 3:
        c6 = np.where(X6 == 'C', 3.0, 0.0)
        return (X1**2 + 2.0*X2 + 0.5*np.exp(-X3) + 0.7*np.sin(np.pi*X4)
                + 0.6*(np.abs(X5)**0.3) + c6 + 0.9*X7 + 1.2*X9)
    else:
        raise ValueError(f"Geçersiz setup: {setup_num}")

def make_classification_dataset(setup_num, nv, n=1500, seed=42):
    rng = np.random.default_rng(seed)
    X1, X2, X3, X4, X5, X6, X7, X8, X9 = _base_features(n, rng)
    y_latent = eval_latent_vector(X1, X2, X3, X4, X5, X6, X7, X8, X9, setup_num)
    
    noise_std = np.sqrt(max(nv, 1e-9))
    epsilon = rng.normal(0, noise_std, n)
    # Threshold = 0.5 (Cavus et al. 2026 standardı)
    y_binary = ((y_latent + epsilon) >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5,
        'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'y': y_binary
    })
    return df, 'X1', noise_std

def compute_ground_truth_pdp(df, var, vals, setup_num, noise_std):
    """Probit bağlantı fonksiyonu altında teorik marjinal sınıf olasılığı PDP'si"""
    gt_probs = []
    df_copy = df.copy()
    for v in vals:
        df_copy[var] = v
        y_lat = eval_latent_vector(
            df_copy['X1'].values, df_copy['X2'].values, df_copy['X3'].values,
            df_copy['X4'].values, df_copy['X5'].values, df_copy['X6'].values,
            df_copy['X7'].values, df_copy['X8'].values, df_copy['X9'].values,
            setup_num
        )
        p = stats.norm.cdf((y_lat - 0.5) / noise_std)
        gt_probs.append(float(np.mean(p)))
    return np.array(gt_probs)

def compute_classification_pdp(model, df_eval, feature, vals):
    """Modelin sınıf-1 tahmin olasılıkları üzerinden PDP hesaplar"""
    gp = len(vals)
    rows = [df_eval.copy().assign(**{feature: v}) for v in vals]
    big_df = pd.concat(rows, ignore_index=True)
    hf_big = h2o.H2OFrame(big_df)
    pred_df = model.predict(hf_big).as_data_frame()
    # 'p1' sınıf-1 olasılığı
    p1 = pred_df['p1'].values
    return p1.reshape(gp, len(df_eval)).mean(axis=1)

# --------------------------------------------------------------------------
# 2. TEK BİR SENARYOYU ÇALIŞTIRMA FONKSİYONU
# --------------------------------------------------------------------------
def run_single_classification_scenario(snum, nv):
    scen_id = f"S{snum}_nv{nv}"
    out_pkl = os.path.join(OUT_CLS_PDP, f"{scen_id}_cls_pdp_data.pkl")
    
    if os.path.exists(out_pkl):
        print(f"[{scen_id}] Zaten mevcut ({out_pkl}), atlanıyor.")
        return

    print(f"\n{'='*65}")
    print(f"BAŞLIYOR: Sınıflandırma {scen_id} (Setup {snum}, sigma^2 = {nv})")
    print(f"Bütçe: {MAX_MODELS} model, {MAX_RUNTIME}s (48 dk)")
    print(f"{'='*65}")
    t0 = time.time()

    df, feat, noise_std = make_classification_dataset(snum, nv, n=1500, seed=42)
    hf = h2o.H2OFrame(df)
    hf['y'] = hf['y'].asfactor()
    
    feat_cols = [c for c in df.columns if c != 'y']
    splits = hf.split_frame(ratios=[0.8], seed=42)
    tr = splits[0]
    te = splits[1]

    aml = H2OAutoML(
        max_models=MAX_MODELS,
        max_runtime_secs=MAX_RUNTIME,
        seed=42,
        sort_metric="AUC",
        exclude_algos=["StackedEnsemble", "DeepLearning"]
    )
    aml.train(x=feat_cols, y="y", training_frame=tr, leaderboard_frame=te)
    
    lb_df = aml.leaderboard.as_data_frame()
    lb = lb_df[['model_id', 'auc', 'logloss']].dropna()
    print(f"[{scen_id}] AutoML tamamlandı ({len(lb)} model eğitildi, süre: {time.time()-t0:.1f}s)")
    print(f"[{scen_id}] En iyi AUC: {lb['auc'].iloc[0]:.4f}")

    df_te = te.as_data_frame()
    all_models = []
    all_aucs = []
    all_loglosses = []
    model_ids = []

    sc_model_dir = os.path.join(OUT_CLS_MODELS, scen_id)
    os.makedirs(sc_model_dir, exist_ok=True)

    for _, row in lb.iterrows():
        mid = row["model_id"]
        try:
            m = h2o.get_model(mid)
            all_models.append(m)
            all_aucs.append(float(row["auc"]))
            all_loglosses.append(float(row["logloss"]))
            model_ids.append(mid)
            m_path = os.path.join(sc_model_dir, mid)
            if not os.path.exists(m_path):
                h2o.save_model(m, path=sc_model_dir, force=True)
        except Exception:
            continue

    if not all_models:
        print(f"[{scen_id}] HATA: Model yüklenemedi!")
        return

    # 20 grid noktası (5. ve 95. yüzdelikler arası)
    vals = np.linspace(df_te[feat].quantile(0.05), df_te[feat].quantile(0.95), GRID_POINTS)
    print(f"[{scen_id}] {len(all_models)} model için PDP profilleri çıkarılıyor...")

    pdp_list = []
    for m in all_models:
        p = compute_classification_pdp(m, df_te, feat, vals)
        pdp_list.append(p)

    pdp_all = np.array(pdp_list)
    best_pdp = pdp_all[0] # Leaderboard 1.si

    # Ground Truth PDP
    gt_pdp = compute_ground_truth_pdp(df_te, feat, vals, snum, noise_std)
    b1_md = float(np.mean(np.abs(best_pdp - gt_pdp)))

    scenario_data = {
        "scenario": scen_id,
        "setup": snum,
        "noise_var": nv,
        "noise_std": noise_std,
        "feature": feat,
        "vals": vals,
        "model_ids": model_ids,
        "all_aucs": all_aucs,
        "all_loglosses": all_loglosses,
        "pdp_all": pdp_all,
        "best_pdp": best_pdp,
        "gt_pdp": gt_pdp,
        "b1_md": b1_md,
        "leaderboard": lb.to_dict(orient="records")
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(scenario_data, f)

    elapsed = time.time() - t0
    print(f"[{scen_id}] TAMAMLANDI: {out_pkl} (Süre: {elapsed/60:.1f} dk, B1 MD: {b1_md:.4f})")

# --------------------------------------------------------------------------
# 3. ANA ÇALIŞTIRICI
# --------------------------------------------------------------------------
def main():
    h2o.init(nthreads=-1, max_mem_size="6G", verbose=False)
    h2o.no_progress()

    setups = [1, 2, 3]
    all_nvs = list(range(1, 10))

    print("\n" + "="*70)
    print("TAM GENİŞLETİLMİŞ BÜTÇELİ SINIFLANDIRMA DENEYLERİ BAŞLIYOR")
    print(f"Toplam Senaryo: {len(setups) * len(all_nvs)} (3 Setup x 9 Gürültü Seviyesi)")
    print(f"Bütçe: {MAX_MODELS} model, {MAX_RUNTIME}s (48 dk) per senaryo")
    print("="*70)

    for snum in setups:
        for nv in all_nvs:
            try:
                run_single_classification_scenario(snum, nv)
            except Exception as e:
                print(f"HATA [{snum}, {nv}]: {e}")
                continue

    h2o.cluster().shutdown()
    print("\n✓ TÜM SINIFLANDIRMA DENEYLERİ EKSİKSİZ TAMAMLANDI!")

if __name__ == "__main__":
    main()
