import urllib.request, json, io, os
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

def get_arff_data(did):
    print(f"Loading did={did}...")
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

# Train AutoML on adult (did=1590, sample 3000)
df_adult = get_arff_data(1590).dropna()
if len(df_adult) > 3000:
    df_adult = df_adult.sample(3000, random_state=42).reset_index(drop=True)
target_a = df_adult.columns[-1]
hf_a = h2o.H2OFrame(df_adult)
hf_a[target_a] = hf_a[target_a].asfactor()

aml_a = H2OAutoML(max_models=20, max_runtime_secs=30, seed=42, exclude_algos=['StackedEnsemble', 'DeepLearning'], sort_metric='logloss')
aml_a.train(x=[c for c in df_adult.columns if c != target_a], y=target_a, training_frame=hf_a)
lb_a = aml_a.leaderboard.as_data_frame()
best_ll_a = lb_a['logloss'].min()
rash_a = lb_a[lb_a['logloss'] <= best_ll_a * 1.10]
print(f"Adult: {len(lb_a)} total models, {len(rash_a)} Rashomon models (|R_eps|)")

# Train AutoML on pc3 (did=1050)
df_pc3 = get_arff_data(1050).dropna()
target_p = df_pc3.columns[-1]
hf_p = h2o.H2OFrame(df_pc3)
hf_p[target_p] = hf_p[target_p].asfactor()

aml_p = H2OAutoML(max_models=20, max_runtime_secs=30, seed=42, exclude_algos=['StackedEnsemble', 'DeepLearning'], sort_metric='logloss')
aml_p.train(x=[c for c in df_pc3.columns if c != target_p], y=target_p, training_frame=hf_p)
lb_p = aml_p.leaderboard.as_data_frame()
best_ll_p = lb_p['logloss'].min()
rash_p = lb_p[lb_p['logloss'] <= best_ll_p * 1.10]
print(f"PC3: {len(lb_p)} total models, {len(rash_p)} Rashomon models (|R_eps|)")

# Save the dataset subsamples and model IDs so R DALEX script can load and compute PDPs
os.makedirs("outputs/real_world", exist_ok=True)
df_adult.to_csv("outputs/real_world/adult_subsample.csv", index=False)
df_pc3.to_csv("outputs/real_world/pc3_subsample.csv", index=False)

rash_info = {
    "adult": {
        "best_model_id": lb_a.iloc[0]['model_id'],
        "rashomon_model_ids": rash_a['model_id'].tolist(),
        "all_model_ids": lb_a['model_id'].tolist(),
        "loglosses": rash_a['logloss'].tolist()
    },
    "pc3": {
        "best_model_id": lb_p.iloc[0]['model_id'],
        "rashomon_model_ids": rash_p['model_id'].tolist(),
        "all_model_ids": lb_p['model_id'].tolist(),
        "loglosses": rash_p['logloss'].tolist()
    }
}
with open("outputs/real_world/rashomon_model_ids.json", "w") as f:
    json.dump(rash_info, f, indent=2)
print("Saved models metadata successfully!")
