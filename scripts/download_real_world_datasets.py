import urllib.request
import json
import io
import os
import pandas as pd
import numpy as np

DATA_DIR = "data/real_world"
os.makedirs(DATA_DIR, exist_ok=True)

def download_openml_dataset(did, name, target_col=None, drop_cols=None, rename_target=None):
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(csv_path):
        print(f"[{name}] Already exists at {csv_path}")
        return pd.read_csv(csv_path)
    
    print(f"[{name}] Downloading did={did} from OpenML...")
    req = urllib.request.urlopen(f'https://www.openml.org/api/v1/json/data/{did}', timeout=20)
    meta = json.loads(req.read().decode('utf-8'))
    file_url = meta['data_set_description']['url']
    req_file = urllib.request.urlopen(file_url, timeout=30)
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
    
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if rename_target:
        df = df.rename(columns=rename_target)
        
    df.to_csv(csv_path, index=False)
    print(f"[{name}] Saved to {csv_path}, shape: {df.shape}")
    return df

# 1. Adult Income (did=1590)
df_adult = download_openml_dataset(1590, "adult", rename_target={"class": "target"})

# 2. German Credit (did=31)
df_credit = download_openml_dataset(31, "german_credit", rename_target={"class": "target"})

# 3. Bank Marketing (did=1461)
df_bank = download_openml_dataset(1461, "bank_marketing", rename_target={"Class": "target"})

# 4. Customer Churn (did=40701)
df_churn = download_openml_dataset(40701, "churn", rename_target={"class": "target"})

# 5. Diabetes (did=37)
df_diabetes = download_openml_dataset(37, "diabetes", rename_target={"class": "target"})

print("\nAll 5 datasets successfully acquired and prepared.")
