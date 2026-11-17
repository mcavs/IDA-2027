"""
Yalnizca kalan son senaryoyu (S3_nv9) tamamlayip ana sonuclarla birlestirir.
"""
import warnings; warnings.filterwarnings("ignore")
import h2o
import json, os
from run_experiments_2880s_160m import run_scenario, OUT

def main():
    print("H2O baslatiliyor (S3_nv9 icin)...")
    h2o.init(nthreads=4, max_mem_size="4G", verbose=False)
    h2o.no_progress()
    
    MASTER_PATH = os.path.join(OUT, "master_analysis_results_2880s_160m.json")
    with open(MASTER_PATH, "r") as f:
        data = json.load(f)
        
    print(f"Mevcut senaryolar: {list(data.keys())}")
    
    # S3_nv9 calistir
    res_s3_nv9 = run_scenario(3, 9, max_models=160, max_runtime=1800)
    data["S3_nv9"] = res_s3_nv9
    
    with open(MASTER_PATH, "w") as f:
        json.dump(data, f, indent=2)
        
    h2o.cluster().shutdown()
    print("✓ S3_nv9 basariyla tamamlandi ve master json guncellendi!")

if __name__ == "__main__":
    main()
