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

h2o.init(nthreads=-1)
h2o.no_progress()

test_dids = [
    (1464, 'blood-transfusion'),
    (1480, 'ilpd'),
    (1053, 'jm1'),
    (1067, 'kc1'),
    (1063, 'kc2'),
    (1050, 'pc3'),
    (1049, 'pc4'),
    (40983, 'wilt'),
    (1487, 'ozone-level-8hr')
]

for did, name in test_dids:
    try:
        df = get_arff_data(did)
        target = df.columns[-1]
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        if not num_cols: continue
        # pick up to 4 continuous columns
        num_cols = num_cols[:4]
        df_clean = df.dropna(subset=[target] + num_cols).reset_index(drop=True)
        if len(df_clean) > 1500: df_clean = df_clean.sample(1500, random_state=42).reset_index(drop=True)
        hf = h2o.H2OFrame(df_clean)
        hf[target] = hf[target].asfactor()
        aml = H2OAutoML(max_models=15, max_runtime_secs=25, seed=42, exclude_algos=['StackedEnsemble', 'DeepLearning'], sort_metric='logloss')
        aml.train(x=[c for c in df_clean.columns if c != target], y=target, training_frame=hf)
        lb = aml.leaderboard.as_data_frame()
        best_ll = lb['logloss'].min()
        rash_lb = lb[lb['logloss'] <= best_ll * 1.10]
        if len(rash_lb) < 4: rash_lb = lb.head(5)
        models = [h2o.get_model(mid) for mid in rash_lb['model_id']]
        
        for col in num_cols:
            pdps = []
            for m in models:
                res = m.partial_plot(frame=hf, cols=[col], plot=False, nbins=15)
                df_pdp = res[0].as_data_frame()
                mean_col = [c for c in df_pdp.columns if 'mean_response' in c][0]
                pdps.append(df_pdp[mean_col].values)
            P = np.array(pdps)
            b1 = P[0]
            b0 = P.mean(axis=0)
            m2 = np.median(P, axis=0) # pointwise median
            # functional median
            dists = np.sum(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1), axis=1)
            m4 = P[np.argmin(dists)]
            
            d_b1_b0 = np.mean(np.abs(b1 - b0))
            d_b1_m2 = np.mean(np.abs(b1 - m2))
            d_b0_m2 = np.mean(np.abs(b0 - m2))
            d_b0_m4 = np.mean(np.abs(b0 - m4))
            r_b1_b0 = np.corrcoef(b1, b0)[0, 1]
            r_b0_m2 = np.corrcoef(b0, m2)[0, 1]
            r_b1_m2 = np.corrcoef(b1, m2)[0, 1]
            
            # Condition for Panel b:
            # B1 and B0 close to each other (d_b1_b0 small, r_b1_b0 high)
            # Both B1 and B0 far from M2/M4 (d_b0_m2 or d_b1_m2 significant, or shape difference)
            # Or ratio of d_b0_m2 / d_b1_b0 is high!
            score = d_b0_m2 / (d_b1_b0 + 1e-4)
            print(f"[{name}] {col} | d(B1,B0)={d_b1_b0:.3f}, d(B0,M2)={d_b0_m2:.3f}, d(B0,M4)={d_b0_m4:.3f} | r(B1,B0)={r_b1_b0:.3f}, r(B0,M2)={r_b0_m2:.3f} | score={score:.2f}")
        h2o.remove(hf)
    except Exception as e:
        print(f"Error in {name}: {e}")

print("Search complete!")
