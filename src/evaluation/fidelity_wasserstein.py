# compute sliced wasserstein distance between the xy distribution of the real and synthetic populations

import ot
import sys
sys.path += [".."]
sys.path += ["../src/generation"]
import pandas as pd
import numpy as np
import pickle
from glob import glob
import config




if __name__ == "__main__":
    n_projections = 1000

    all_distances = []

    for file in sorted(glob(config.baselines_path + f'all_baselines_*.pickle')):
        city = file.split(".")[-2].splut("_")[-1]
        print(city)

        with open(file, 'rb') as f:
            data = pickle.load(f)

        wsd = {}
        wsd["city"] = city

        for k in data.keys():
            if "95" in k:
                wsd[k] = ot.sliced_wasserstein_distance(data["df_real95"][['x', 'y']].to_numpy(), data[k][['x', 'y']].to_numpy(), n_projections=n_projections)
            else:
                wsd[k] = ot.sliced_wasserstein_distance(data["df_real"][['x', 'y']].to_numpy(), data[k][['x', 'y']].to_numpy(), n_projections=n_projections)
        
        all_distances.append(wsd)
    pd.DataFrame(all_distances).set_index("city").to_csv(config.path_airbnb + "wasserstein_distances.csv")
        

        



