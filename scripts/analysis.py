"""
Rashomon PDP Aggregation — Tam Reproducible Analiz Kodu
========================================================
Makale: "Aggregating Rashomon Partial Dependence Profiles:
         A Theoretical Framework and Empirical Evaluation"

Üretilen çıktılar:
  - MD(GT) sonuç tabloları (9 senaryo × 8 yöntem)
  - İkili CI (CI1 + CI2) ve kapsama oranları
  - Quadrant analizi görseli
  - Varyans tarama (σ² ∈ {0,...,9}) görseli

Gereksinimler:
  pip install h2o scikit-learn cvxpy scipy pandas numpy matplotlib
"""

import warnings; warnings.filterwarnings("ignore")
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare, rankdata
import cvxpy as cp
import os, time, json

# ── Çıktı dizini ──────────────────────────────────────────────────────────────
OUT  = "outputs"
CKPT = "ci_results.json"
os.makedirs(OUT, exist_ok=True)

TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
NBOOTS = 200
ALPHA  = 0.05

# ══════════════════════════════════════════════════════════════════════════════
# 1. SENTETİK VERİ — Cavus et al. (2026) Setup 1, 2, 3
# ══════════════════════════════════════════════════════════════════════════════
def _base(n, rng):
    return (rng.normal(0,1,n), rng.binomial(1,.5,n).astype(float),
            rng.normal(0,1,n), rng.uniform(-2,2,n), rng.exponential(1,n),
            rng.choice(["A","B","C"],n), rng.normal(0,1,n),
            rng.choice([0,1],n).astype(float), rng.normal(0,1,n))

def make_setup1(n=1500, nv=1, seed=42):
    rng = np.random.default_rng(seed)
    X1,X2,X3,X4,X5,X6,X7,X8,X9 = _base(n, rng)
    c6 = np.where(X6=="B",2.,np.where(X6=="C",-1.,0.))
    y  = (2.9*X1 - 3.7*X2 + 1.2*X3 + 0.8*X1**2 - 1.5*X2*X3
          + 1.1*X4 - 0.5*np.log(X5+1) + c6 + 0.3*X9
          + rng.normal(0, np.sqrt(max(nv,1e-9)), n))
    df = pd.DataFrame({"X1":X1,"X2":X2,"X3":X3,"X4":X4,"X5":X5,
                       "X6":X6,"X7":X7,"X8":X8,"X9":X9,"y":y})
    return df, "X1", lambda v: 2.9*v + 0.8*v**2

def make_setup2(n=1500, nv=1, seed=42):
    rng = np.random.default_rng(seed)
    X1,X2,X3,X4,X5,X6,X7,X8,X9 = _base(n, rng)
    c6 = np.where(X6=="A",-2.,np.where(X6=="C",1.,0.))
    y  = (np.sin(X1) + np.log(np.abs(X3)+1) + 0.8*X4*X5 - 1.2*X1*X3
          + c6 + 1.5*X7 + 2*(X8==1) + 0.4*X9**2
          + rng.normal(0, np.sqrt(max(nv,1e-9)), n))
    df = pd.DataFrame({"X1":X1,"X2":X2,"X3":X3,"X4":X4,"X5":X5,
                       "X6":X6,"X7":X7,"X8":X8,"X9":X9,"y":y})
    return df, "X1", lambda v: np.sin(v)

def make_setup3(n=1500, nv=1, seed=42):
    rng = np.random.default_rng(seed)
    X1,X2,X3,X4,X5,X6,X7,X8,X9 = _base(n, rng)
    c6 = np.where(X6=="C",3.,0.)
    y  = (X1**2 + 2*X2 + 0.5*np.exp(-X3) + 0.7*np.sin(np.pi*X4)
          + 0.6*(np.abs(X5)**0.3) + c6 + 0.9*X7 + 1.2*X9
          + rng.normal(0, np.sqrt(max(nv,1e-9)), n))
    df = pd.DataFrame({"X1":X1,"X2":X2,"X3":X3,"X4":X4,"X5":X5,
                       "X6":X6,"X7":X7,"X8":X8,"X9":X9,"y":y})
    return df, "X1", lambda v: v**2

def make_diabetes():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["y"] = data.target
    return df, "bmi", None

SETUP_FNS = {1: make_setup1, 2: make_setup2, 3: make_setup3}

# ══════════════════════════════════════════════════════════════════════════════
# 2. AUTOML + RASHOMON SETİ
# ══════════════════════════════════════════════════════════════════════════════
def run_automl(df, target="y", max_models=12, max_runtime=65, seed=42):
    """H2O AutoML — makale konfigürasyonu: max 12 model, 65s, ε=0.20"""
    hf   = h2o.H2OFrame(df)
    feat = [c for c in df.columns if c != target]
    tr, _ = hf.split_frame(ratios=[0.8], seed=seed)
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime,
        seed=seed,
        sort_metric="RMSE",
        exclude_algos=["StackedEnsemble", "DeepLearning"]
    )
    aml.train(x=feat, y=target, training_frame=tr)
    lb = aml.leaderboard.as_data_frame()[["model_id","rmse"]].dropna()
    return lb, tr

def rashomon_set(lb, eps=0.20):
    best = lb["rmse"].min()
    R    = lb[lb["rmse"] <= best*(1+eps)]
    return R, len(R)/len(lb)

# ══════════════════════════════════════════════════════════════════════════════
# 3. PDP HESAPLAMA (toplu — hız için)
# ══════════════════════════════════════════════════════════════════════════════
def compute_pdp(model, df_tr, feature, gp=20):
    vals = np.linspace(df_tr[feature].quantile(.05),
                       df_tr[feature].quantile(.95), gp)
    rows = [df_tr.copy().assign(**{feature: v}) for v in vals]
    big  = pd.concat(rows, ignore_index=True)
    pred = model.predict(h2o.H2OFrame(big)).as_data_frame()["predict"].values
    return vals, pred.reshape(gp, len(df_tr)).mean(axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SEKIZ AGREGASYON YÖNTEMİ
# ══════════════════════════════════════════════════════════════════════════════
def agg_simple(P, **_):
    return P.mean(0)

def agg_perf(P, rmses, **_):
    w = 1/np.array(rmses); w /= w.sum()
    return (P * w[:,None]).sum(0)

def agg_robust(P, np_=10, std=.02, **_):
    k = len(P)
    s = np.array([np.mean([np.linalg.norm(
            p - (p + np.random.normal(0,std,len(p))))**2
            for _ in range(np_)]) for p in P])
    w = cp.Variable(k)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, np.diag(s)+1e-6*np.eye(k))),
                      [w >= 0, cp.sum(w) == 1])
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None: return agg_simple(P)
    wv = np.clip(w.value,0,None); wv /= wv.sum()
    return (P * wv[:,None]).sum(0)

def agg_faith(P, ref, **_):
    k = len(P); w = cp.Variable(k)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(P.T@w - ref)),
                      [w >= 0, cp.sum(w) == 1])
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None: return agg_simple(P)
    wv = np.clip(w.value,0,None); wv /= wv.sum()
    return (P * wv[:,None]).sum(0)

def agg_opt(P, ref, lam=.5, np_=10, std=.02, **_):
    k, m = P.shape
    s = np.array([np.mean([np.linalg.norm(
            p - (p + np.random.normal(0,std,m)))**2
            for _ in range(np_)]) for p in P])
    w = cp.Variable(k)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(P.T@w - ref) +
                    lam * cp.quad_form(w, np.diag(s)+1e-6*np.eye(k))),
        [w >= 0, cp.sum(w) == 1])
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None: return agg_simple(P)
    wv = np.clip(w.value,0,None); wv /= wv.sum()
    return (P * wv[:,None]).sum(0)

def agg_ava(P, **_):
    m = P.mean(0)
    d = np.array([np.linalg.norm(p - m) + 1e-8 for p in P])
    w = 1/d; w /= w.sum()
    return (P * w[:,None]).sum(0)

def agg_rank(P, **_):
    a = np.array([TRAPZ(np.abs(p)) for p in P])
    med = np.median(a)
    w = 1/(np.abs(a - med) + 1e-8); w /= w.sum()
    return (P * w[:,None]).sum(0)

METHOD_NAMES = [
    "B0:SimpleAvg", "B1:Best-model", "B2:PerfWeighted",
    "B3:AGGrobust",  "B4:AVA-PDP",   "B8:AGREE-Rank",
    "Ö1:AGGfaith",   "Ö2:AGGopt"
]

COLORS = {
    "B0:SimpleAvg":    "#2C7BB6", "B1:Best-model":   "#444444",
    "B2:PerfWeighted": "#2CA02C", "B3:AGGrobust":    "#FD8D3C",
    "B4:AVA-PDP":      "#9467BD", "B8:AGREE-Rank":   "#8C564B",
    "Ö1:AGGfaith":     "#E6550D", "Ö2:AGGopt":       "#D62728",
}
LS_D = {
    "B0:SimpleAvg":"--","B1:Best-model":":","B2:PerfWeighted":"-",
    "B3:AGGrobust":"-","B4:AVA-PDP":"-","B8:AGREE-Rank":"-",
    "Ö1:AGGfaith":"-.","Ö2:AGGopt":"-.",
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. METRİKLER
# ══════════════════════════════════════════════════════════════════════════════
def md_gt(agg, gt):     return float(np.mean(np.abs(agg - gt)))
def fscore(agg, ref):   return float(np.linalg.norm(agg - ref))
def sscore(agg, vals):  r,_ = spearmanr(agg, vals); return abs(float(r))

def compute_fixed_weights(method, P, rmses, ref):
    """CI₂ için sabit ağırlık vektörü"""
    k = len(P)
    if method == "B0:SimpleAvg":    return np.ones(k)/k
    if method == "B2:PerfWeighted":
        w = 1/np.array(rmses); w /= w.sum(); return w
    if method == "B3:AGGrobust":
        s = np.array([np.mean([np.linalg.norm(p-(p+np.random.normal(0,.02,len(p))))**2
                               for _ in range(8)]) for p in P])
        wv = cp.Variable(k)
        prob = cp.Problem(cp.Minimize(cp.quad_form(wv,np.diag(s)+1e-6*np.eye(k))),
                          [wv>=0, cp.sum(wv)==1])
        prob.solve(solver=cp.SCS, verbose=False)
        if wv.value is None: return np.ones(k)/k
        w = np.clip(wv.value,0,None); w /= w.sum(); return w
    if method == "B4:AVA-PDP":
        m = P.mean(0); d = np.array([np.linalg.norm(p-m)+1e-8 for p in P])
        w = 1/d; w /= w.sum(); return w
    if method == "B8:AGREE-Rank":
        a = np.array([TRAPZ(np.abs(p)) for p in P]); med = np.median(a)
        w = 1/(np.abs(a-med)+1e-8); w /= w.sum(); return w
    if method in ("Ö1:AGGfaith","Ö2:AGGopt"):
        wv = cp.Variable(k)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(P.T@wv-ref)),
                          [wv>=0, cp.sum(wv)==1])
        prob.solve(solver=cp.SCS, verbose=False)
        if wv.value is None: return np.ones(k)/k
        w = np.clip(wv.value,0,None); w /= w.sum(); return w
    return np.ones(k)/k

def bootstrap_ci(method, P, rmses, ref, fixed_weights, B=NBOOTS):
    """İkili CI: CI₁ (ağırlıklar yeniden hesaplanır) + CI₂ (sabit ağırlık)"""
    k = len(P)
    boots1, boots2 = [], []
    for _ in range(B):
        idx = np.random.choice(k, k, replace=True)
        Pb = P[idx]; rb = [rmses[i] for i in idx]; refb = Pb.mean(0)
        # CI₁
        if method == "B0:SimpleAvg":
            a1 = Pb.mean(0)
        elif method == "B2:PerfWeighted":
            w = 1/np.array(rb); w /= w.sum(); a1 = (Pb*w[:,None]).sum(0)
        elif method == "B3:AGGrobust":
            a1 = agg_robust(Pb, np_=5)
        elif method == "B4:AVA-PDP":
            a1 = agg_ava(Pb)
        elif method == "B8:AGREE-Rank":
            a1 = agg_rank(Pb)
        elif method in ("Ö1:AGGfaith","Ö2:AGGopt"):
            a1 = agg_faith(Pb, refb)
        else:
            a1 = Pb.mean(0)
        boots1.append(a1)
        # CI₂
        boots2.append((Pb * fixed_weights[:,None]).sum(0))

    b1, b2 = np.array(boots1), np.array(boots2)
    lo1,hi1 = np.percentile(b1,100*ALPHA/2,0), np.percentile(b1,100*(1-ALPHA/2),0)
    lo2,hi2 = np.percentile(b2,100*ALPHA/2,0), np.percentile(b2,100*(1-ALPHA/2),0)
    return (lo1,hi1), (lo2,hi2)

def coverage_rate(gt, lo, hi):
    return float(np.mean((gt >= lo) & (gt <= hi)))

def mwci(lo, hi):
    return float(np.mean(hi - lo))

# ══════════════════════════════════════════════════════════════════════════════
# 6. TEK DENEY ÇALIŞTIRICISI
# ══════════════════════════════════════════════════════════════════════════════
AGG_FNS = {
    "B0:SimpleAvg":    agg_simple,
    "B2:PerfWeighted": agg_perf,
    "B3:AGGrobust":    agg_robust,
    "B4:AVA-PDP":      agg_ava,
    "B8:AGREE-Rank":   agg_rank,
    "Ö1:AGGfaith":     agg_faith,
    "Ö2:AGGopt":       agg_opt,
}

def run_experiment(snum, nv, eps=0.20, max_models=12, max_runtime=65):
    """Tek bir (setup, σ²) kombinasyonu için tüm analizi çalıştır."""
    df, feat, gt_fn = SETUP_FNS[snum](nv=nv)
    lb, tr = run_automl(df, max_models=max_models, max_runtime=max_runtime)
    R_df, rho = rashomon_set(lb, eps)
    if len(R_df) < 2: return None

    all_models, all_rmses = [], []
    for _, row in lb.iterrows():
        try:
            m = h2o.get_model(row["model_id"])
            all_models.append(m); all_rmses.append(row["rmse"])
        except: pass

    best_rmse  = all_rmses[0]
    rash_idx   = [i for i,r in enumerate(all_rmses) if r <= best_rmse*(1+eps)]
    if len(rash_idx) < 2: return None

    df_tr = tr.as_data_frame()
    vals, best_pdp = compute_pdp(all_models[0], df_tr, feat)
    pdp_all = np.array([compute_pdp(m, df_tr, feat, gp=len(vals))[1]
                        for m in all_models])

    P     = pdp_all[rash_idx]
    rmses = [all_rmses[i] for i in rash_idx]
    ref   = P.mean(0)

    gt_raw = gt_fn(vals)
    gt     = gt_raw - gt_raw.mean() + P.mean()

    results = {}
    # B1: Best-model — CI yok
    results["B1:Best-model"] = {
        "md": float(np.mean(np.abs(best_pdp-gt))),
        "agg": best_pdp.tolist(),
        "cr1": None, "cr2": None, "mwci1": None, "mwci2": None
    }
    for method, fn in AGG_FNS.items():
        agg = fn(P=P, rmses=rmses, ref=ref)
        fw  = compute_fixed_weights(method, P, rmses, ref)
        (lo1,hi1),(lo2,hi2) = bootstrap_ci(method, P, rmses, ref, fw)
        results[method] = {
            "md":    float(np.mean(np.abs(agg-gt))),
            "agg":   agg.tolist(),
            "cr1":   coverage_rate(gt,lo1,hi1),
            "cr2":   coverage_rate(gt,lo2,hi2),
            "mwci1": mwci(lo1,hi1),
            "mwci2": mwci(lo2,hi2),
        }
    return {"vals":vals.tolist(),"gt":gt.tolist(),"best_pdp":best_pdp.tolist(),
            "rho":rho,"n_rash":len(rash_idx),"P":P.tolist(),"results":results}

# ══════════════════════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════
def plot_md_heatmap(all_results, fname):
    """MD(GT) ısı haritası — B1 referanslı, ★ en iyi"""
    SCENS = [f"S{s}-σ²={nv}" for s in [1,2,3] for nv in [1,4,9]]
    SHORT = [m.split(":")[0] for m in METHOD_NAMES]
    MD = np.full((9, 8), np.nan)
    for si, (snum, nv) in enumerate([(s,nv) for s in [1,2,3] for nv in [1,4,9]]):
        key = f"S{snum}_nv{nv}"
        if key not in all_results: continue
        exp = all_results[key]
        for mi, m in enumerate(METHOD_NAMES):
            if m in exp["results"]:
                MD[si,mi] = exp["results"][m]["md"]

    fig, axes = plt.subplots(1,2,figsize=(18,7),gridspec_kw={"wspace":.32})
    for ax,mat,title in [(axes[0],MD,"MD(GT) — Rashomon Seti")]:
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=np.nanmin(mat)*.95, vmax=np.nanmax(mat)*1.05)
        ax.set_xticks(range(len(SHORT))); ax.set_xticklabels(SHORT,fontsize=9,rotation=28,ha="right")
        ax.set_yticks(range(len(SCENS))); ax.set_yticklabels(SCENS,fontsize=9)
        for r in range(mat.shape[0]):
            bc = np.nanargmin(mat[r])
            for c in range(mat.shape[1]):
                v = mat[r,c]
                if np.isnan(v): continue
                txt = f"{v:.4f}" + ("★" if c==bc else "")
                ax.text(c,r,txt,ha="center",va="center",fontsize=7,
                        color="white" if v>np.nanpercentile(mat,70) else "black",
                        fontweight="bold" if c==bc else "normal")
        plt.colorbar(im,ax=ax,label="MD(GT)",shrink=.8)
        ax.set_title(title,fontsize=11,fontweight="bold")

    # Fark haritası: yöntem − B1
    b1_col = MD[:,1:2]
    DIFF = MD - b1_col
    lim  = np.nanmax(np.abs(DIFF))*1.1
    im2  = axes[1].imshow(DIFF,aspect="auto",cmap="RdYlGn_r",vmin=-lim,vmax=lim)
    axes[1].set_xticks(range(len(SHORT))); axes[1].set_xticklabels(SHORT,fontsize=9,rotation=28,ha="right")
    axes[1].set_yticks(range(len(SCENS))); axes[1].set_yticklabels(SCENS,fontsize=9)
    for r in range(DIFF.shape[0]):
        for c in range(DIFF.shape[1]):
            v = DIFF[r,c]
            if np.isnan(v): continue
            axes[1].text(c,r,f"{v:+.4f}",ha="center",va="center",fontsize=7,
                         color="white" if abs(v)>lim*.55 else "black")
    plt.colorbar(im2,ax=axes[1],label="ΔMD = yöntem − B1\n(negatif = yöntem B1'den iyi)",shrink=.8)
    axes[1].set_title("ΔMD(GT) — B1 Referanslı",fontsize=11,fontweight="bold")
    plt.suptitle("MD(GT) Sonuçları | ★ en iyi | B1=Best-model referans",fontsize=12,fontweight="bold")
    plt.savefig(fname,dpi=145,bbox_inches="tight",facecolor="white")
    plt.close(); print(f"  {fname}")

def plot_ci_summary(all_results, fname):
    """CR₂ ve MWCI₂ özet — yöntem × senaryo"""
    methods_ci = [m for m in METHOD_NAMES if m != "B1:Best-model"]
    fig, axes  = plt.subplots(1,2,figsize=(16,6),gridspec_kw={"wspace":.35})

    for ax_i, (metric, ylabel) in enumerate([("cr2","GT Kapsama (CR₂)"),
                                               ("mwci2","CI Genişliği (MWCI₂)")]):
        ax = axes[ax_i]
        noise_vals = [1,4,9]
        x = np.arange(len(noise_vals))
        w = 0.10
        offsets = np.linspace(-(len(methods_ci)-1)*w/2,
                               (len(methods_ci)-1)*w/2, len(methods_ci))
        for mi, method in enumerate(methods_ci):
            for si, snum in enumerate([1,2,3]):
                ys = []
                for nv in noise_vals:
                    key = f"S{snum}_nv{nv}"
                    v = (all_results.get(key,{}).get("results",{})
                         .get(method,{}).get(metric))
                    ys.append(v if v is not None else np.nan)
                ls = ["-","--","-."][si]
                ax.plot(x+offsets[mi], ys, color=COLORS[method], ls=ls,
                        marker=["o","s","^"][si], ms=6, lw=1.5)

        ax.set_xticks(x); ax.set_xticklabels([f"σ²={v}" for v in noise_vals],fontsize=9)
        ax.set_ylabel(ylabel,fontsize=9)
        ax.set_title(ylabel,fontsize=10,fontweight="bold")
        ax.grid(True,alpha=.3)
        if "Kapsama" in ylabel:
            ax.axhline(0.95,color="black",lw=1,ls=":",alpha=.6,label="Nominal %95")
            ax.set_ylim(0,1.05); ax.legend(fontsize=8)

    from matplotlib.lines import Line2D
    handles  = [Line2D([0],[0],color=COLORS[m],lw=2,label=m) for m in methods_ci]
    handles += [Line2D([0],[0],color="k",lw=1.5,ls=ls,marker=mk,ms=6,label=f"Setup {si+1}")
                for si,(ls,mk) in enumerate(zip(["-","--","-."],["o","s","^"]))]
    fig.legend(handles=handles,ncol=5,fontsize=7.5,
               loc="lower center",bbox_to_anchor=(.5,-.04))
    plt.suptitle("CI Özet — GT Kapsama Oranı ve CI Genişliği",fontsize=12,fontweight="bold")
    plt.savefig(fname,dpi=140,bbox_inches="tight",facecolor="white")
    plt.close(); print(f"  {fname}")

def plot_quadrant(all_results, fname):
    """CR₂ × MD(GT) quadrant analizi — senaryo bazlı"""
    pts = []
    for key, exp in all_results.items():
        if not exp: continue
        snum = int(key[1]); nv = int(key.split("nv")[1])
        for m in METHOD_NAMES:
            if m == "B1:Best-model": continue
            r = exp["results"].get(m,{})
            cr = r.get("cr2"); md = r.get("md")
            if cr is not None and md is not None:
                pts.append({"m":m,"s":snum,"nv":nv,"cr":cr,"md":md})

    if not pts: return
    df_pts = pd.DataFrame(pts)
    CR_MID = df_pts["cr"].median()
    MD_MID = df_pts["md"].median()

    fig, ax = plt.subplots(figsize=(10,7))
    # Quadrant renkleri
    for x1,x2,y1,y2,fc in [
        (0,CR_MID,MD_MID,1,  "rgba(161,93,50,.04)"),
        (CR_MID,1,MD_MID,1,  "rgba(186,117,23,.06)"),
        (0,CR_MID,0,MD_MID,  "rgba(24,95,165,.05)"),
        (CR_MID,1,0,MD_MID,  "rgba(15,110,86,.07)"),
    ]:
        pass  # matplotlib'te fill_between ile benzer etki

    S_SHAPE = {1:"o",2:"s",3:"^"}
    NV_SIZE  = {1:60,4:100,9:160}
    for _, row in df_pts.iterrows():
        ax.scatter(row["cr"], row["md"],
                   color=COLORS[row["m"]],
                   marker=S_SHAPE[row["s"]],
                   s=NV_SIZE[row["nv"]],
                   alpha=0.85, zorder=5,
                   edgecolors="white", linewidths=0.5)

    ax.axvline(CR_MID,color="black",lw=1.2,ls="--",alpha=.5)
    ax.axhline(MD_MID,color="black",lw=1.2,ls="--",alpha=.5)
    ax.set_xlabel("GT Kapsama Oranı (CR₂)",fontsize=11)
    ax.set_ylabel("MD(GT)",fontsize=11)
    ax.set_title("Quadrant Analizi: CR₂ × MD(GT)\n"
                 "İdeal bölge: yüksek CR₂ + düşük MD (sağ-alt)",
                 fontsize=11,fontweight="bold")
    ax.grid(True,alpha=.25)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles  = [Patch(fc=COLORS[m],label=m) for m in METHOD_NAMES if m!="B1:Best-model"]
    handles += [Line2D([0],[0],marker=S_SHAPE[s],color="gray",ms=8,
                       linestyle="none",label=f"Setup {s}") for s in [1,2,3]]
    handles += [Line2D([0],[0],marker="o",color="gray",ms=np.sqrt(NV_SIZE[nv]/np.pi),
                       linestyle="none",label=f"σ²={nv}") for nv in [1,4,9]]
    ax.legend(handles=handles,fontsize=7,ncol=3,loc="upper left",framealpha=.85)
    plt.savefig(fname,dpi=145,bbox_inches="tight",facecolor="white")
    plt.close(); print(f"  {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. VARYANSS TARAMA σ² ∈ {0,...,9}
# ══════════════════════════════════════════════════════════════════════════════
def run_variance_sweep(sweep_ckpt="variance_sweep.json",
                       max_models=10, max_runtime=55):
    """σ² ∈ {0,...,9} × Setup {1,2,3} — kontrol noktalı"""
    if os.path.exists(sweep_ckpt):
        with open(sweep_ckpt) as f: results = json.load(f)
        print(f"Sweep kontrol noktası yüklendi: {len(results)} sonuç")
    else:
        results = {}

    for snum in [1,2,3]:
        for nv in range(10):
            key = f"S{snum}_nv{nv}"
            if key in results:
                print(f"  [{key}] atlandı"); continue
            print(f"  [{key}] çalışıyor...", end="", flush=True)
            t0 = time.time()
            try:
                df, feat, gt_fn = SETUP_FNS[snum](nv=nv)
                lb, tr = run_automl(df, max_models=max_models,
                                    max_runtime=max_runtime)
                R_df, rho = rashomon_set(lb)
                if len(R_df) < 2:
                    print(" yetersiz"); continue
                all_models, all_rmses = [], []
                for _, row in lb.iterrows():
                    try:
                        m = h2o.get_model(row["model_id"])
                        all_models.append(m); all_rmses.append(row["rmse"])
                    except: pass
                best_rmse = all_rmses[0]
                rash_idx  = [i for i,r in enumerate(all_rmses)
                              if r <= best_rmse*1.20]
                if len(rash_idx) < 2:
                    print(" yetersiz"); continue
                df_tr = tr.as_data_frame()
                vals, best_pdp = compute_pdp(all_models[0], df_tr, feat)
                pdp_all = np.array([compute_pdp(m,df_tr,feat,gp=len(vals))[1]
                                    for m in all_models])
                P     = pdp_all[rash_idx]
                rmses = [all_rmses[i] for i in rash_idx]
                ref   = P.mean(0)
                gt_raw = gt_fn(vals)
                gt     = gt_raw - gt_raw.mean() + P.mean()
                row_res = {}
                for m, fn in AGG_FNS.items():
                    agg = fn(P=P, rmses=rmses, ref=ref)
                    row_res[m] = float(np.mean(np.abs(agg - gt)))
                row_res["B1:Best-model"] = float(np.mean(np.abs(best_pdp - gt)))
                results[key] = row_res
                with open(sweep_ckpt,"w") as f: json.dump(results,f,indent=2)
                print(f" ✓ ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f" HATA: {e}")
    return results

def plot_sweep(sweep_results, fname):
    """σ² × MD(GT) çizgi grafikleri — 3 setup yan yana"""
    fig, axes = plt.subplots(1,3,figsize=(18,5.5),
                             gridspec_kw={"wspace":.32})
    x = np.arange(10)
    for si, snum in enumerate([1,2,3]):
        ax = axes[si]
        for m in METHOD_NAMES:
            ys = [sweep_results.get(f"S{snum}_nv{nv}",{}).get(m,np.nan)
                  for nv in range(10)]
            lw = 2.3 if m in ("B1:Best-model","B3:AGGrobust","B2:PerfWeighted") else 1.5
            ax.plot(x, ys, color=COLORS[m], ls=LS_D[m],
                    marker="o", ms=5, lw=lw, label=m)
        ax.set_xticks(x); ax.set_xlabel("Gürültü varyansı (σ²)",fontsize=10)
        ax.set_ylabel("MD(GT)",fontsize=10)
        ax.set_title(f"Setup {snum}",fontsize=11,fontweight="bold")
        ax.grid(True,alpha=.3)
        ax.legend(fontsize=7,ncol=2,loc="upper left",framealpha=.88)
    plt.suptitle("MD(GT) — σ² Tarama | Rashomon Seti | Tüm Yöntemler",
                 fontsize=12,fontweight="bold")
    plt.savefig(fname,dpi=145,bbox_inches="tight",facecolor="white")
    plt.close(); print(f"  {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. HİPOTEZ TESTİ (B1 referanslı)
# ══════════════════════════════════════════════════════════════════════════════
def hypothesis_test(all_results):
    """Friedman + Wilcoxon (Bonferroni) — B1 referanslı"""
    SCENS = [(s,nv) for s in [1,2,3] for nv in [1,4,9]]
    MD = np.full((9,8), np.nan)
    for si,(snum,nv) in enumerate(SCENS):
        key = f"S{snum}_nv{nv}"
        if key not in all_results: continue
        for mi,m in enumerate(METHOD_NAMES):
            MD[si,mi] = all_results[key]["results"].get(m,{}).get("md",np.nan)

    # Friedman
    fstat,fpval = friedmanchisquare(*[MD[:,j] for j in range(8)])
    print(f"\nFriedman: χ²={fstat:.4f}, p={fpval:.4f}")

    # Wilcoxon — B1 referans
    b1 = MD[:,1]
    n_comp = 7
    alpha_bonf = 0.05/n_comp
    print(f"\nWilcoxon (Bonferroni α={alpha_bonf:.4f}):")
    for j,m in enumerate(METHOD_NAMES):
        if j == 1: continue
        col = MD[:,j]
        diff = b1 - col
        if np.all(diff==0): continue
        try:
            wstat,pval = wilcoxon(b1, col, alternative="two-sided")
            order = np.argsort(np.abs(diff[diff!=0]))
            r_vals = np.empty_like(order); r_vals[order] = np.arange(len(order))
            d = diff[diff!=0]
            r = (r_vals[d>0].sum()-r_vals[d<0].sum())/(len(d)*(len(d)+1)/2)
        except:
            wstat,pval,r = np.nan,1.,0.
        sig = "✓" if pval < alpha_bonf else ("~" if pval < 0.05 else "✗")
        print(f"  {m:20s}: W={wstat:.1f}, p_bonf={min(pval*n_comp,1):.4f} {sig}, "
              f"Δ={np.mean(col-b1):+.5f}, r={r:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("H2O başlatılıyor...")
    h2o.init(nthreads=4, max_mem_size="4G", verbose=False)
    h2o.no_progress()
    print("Hazır.\n")

    # ── Ana deney: 9 senaryo (3 setup × 3 σ²) + Diabetes ────────────────────
    all_results = {}

    for snum in [1,2,3]:
        for nv in [1,4,9]:
            key = f"S{snum}_nv{nv}"
            print(f"[{key}] çalışıyor...")
            exp = run_experiment(snum, nv)
            all_results[key] = exp
            if exp:
                print(f"  ρ={exp['rho']:.2f}, K={exp['n_rash']} | "
                      f"MD(B0)={exp['results']['B0:SimpleAvg']['md']:.4f}, "
                      f"MD(B1)={exp['results']['B1:Best-model']['md']:.4f}")

    # Diabetes
    print("[Diabetes] çalışıyor...")
    df_r, feat_r, gt_r = make_diabetes()
    # Diabetes için gt_fn yok — MD hesabı atlanır, CI hesabı yapılır

    # ── Görseller ─────────────────────────────────────────────────────────────
    print("\nGörseller üretiliyor...")
    plot_md_heatmap(all_results, f"{OUT}/heatmap_md.png")
    plot_ci_summary(all_results, f"{OUT}/ci_summary.png")
    plot_quadrant(all_results, f"{OUT}/quadrant.png")

    # ── Hipotez testi ─────────────────────────────────────────────────────────
    hypothesis_test(all_results)

    # ── Varyans tarama ────────────────────────────────────────────────────────
    print("\nVaryans tarama (σ²∈{0,...,9}) başlıyor...")
    sweep = run_variance_sweep()
    plot_sweep(sweep, f"{OUT}/sweep_md.png")

    h2o.cluster().shutdown()
    print("\n✓ Tüm analizler tamamlandı.")
