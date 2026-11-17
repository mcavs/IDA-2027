import urllib.request, json, io, os
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

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

df = get_arff_data(29)
target = df.columns[-1]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
print("Credit approval numeric cols:", num_cols)

hf = h2o.H2OFrame(df.dropna())
hf[target] = hf[target].asfactor()
aml = H2OAutoML(max_models=20, max_runtime_secs=30, seed=42, exclude_algos=['StackedEnsemble', 'DeepLearning'], sort_metric='logloss')
aml.train(x=[c for c in df.columns if c != target], y=target, training_frame=hf)
lb = aml.leaderboard.as_data_frame()
best_ll = lb['logloss'].min()
rash_lb = lb[lb['logloss'] <= best_ll * 1.10]
models = [h2o.get_model(mid) for mid in rash_lb['model_id']]
print(f"Credit Approval: K={len(models)}")

for col in num_cols:
    pdps = []
    for m in models:
        res = m.partial_plot(frame=hf, cols=[col], plot=False, nbins=15)
        df_pdp = res[0].as_data_frame()
        mean_col = [c for c in df_pdp.columns if 'mean_response' in c][0]
        pdps.append(df_pdp[mean_col].values)
    P = np.array(pdps)
    b1, b0, m2, m4 = P[0], P.mean(axis=0), np.median(P, axis=0), P[np.argmin(np.sum(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1), axis=1))]
    d_b1_b0 = np.mean(np.abs(b1 - b0))
    d_b0_m2 = np.mean(np.abs(b0 - m2))
    d_b1_m2 = np.mean(np.abs(b1 - m2))
    r_b1_b0 = np.corrcoef(b1, b0)[0, 1]
    r_b0_m2 = np.corrcoef(b0, m2)[0, 1]
    print(f"Col {col:10s} | d(B1,B0)={d_b1_b0:.3f}, d(B1,M2)={d_b1_m2:.3f}, d(B0,M2)={d_b0_m2:.3f} | r(B1,B0)={r_b1_b0:.3f}, r(B0,M2)={r_b0_m2:.3f}")
