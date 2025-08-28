import ot
import sys
sys.path += ["../src"]
import pandas as pd
import numpy as np
import pickle
from glob import glob





if __name__ == "__main__":
    n_projections = 1000

    all_distances = []

    for file in sorted(glob(f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_*.pickle')):
        prov = file.split(".")[-2][-2:]
        print(prov)

        with open(file, 'rb') as f:
            data = pickle.load(f)

        wsd = {}
        wsd["prov"] = prov

        for k in data.keys():
            if "95" in k:
                wsd[k] = ot.sliced_wasserstein_distance(data["df_real95"][['x', 'y']].to_numpy(), data[k][['x', 'y']].to_numpy(), n_projections=n_projections)
            else:
                wsd[k] = ot.sliced_wasserstein_distance(data["df_real"][['x', 'y']].to_numpy(), data[k][['x', 'y']].to_numpy(), n_projections=n_projections)
        
        all_distances.append(wsd)
        pd.DataFrame(all_distances).set_index("prov").to_csv("/data/housing/data/intermediate/jl_pop_synth/wasserstein_isp_250827.csv")
        #pd.DataFrame(all_distances).set_index("prov").to_csv("/data/housing/data/intermediate/jl_pop_synth/wasserstein_isp_250825.csv")

        



