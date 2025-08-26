
import pandas as pd
import numpy as np
import pandas as pd
from glob import glob

import geopandas as gpd
import pickle
import jl_vae






def get_cap_from_xy(df, geo_dict):
    df_geo = gpd.points_from_xy(df['x'], df['y'], z=None, crs="EPSG:4326")
    df_geo = df_geo.to_crs('EPSG:3035') #4326 3035
    if "CAP" in df.columns:
        df_ = df.drop(columns = ["CAP"]).copy()
    else:
        df_ = df.copy()
    df_gpd = gpd.GeoDataFrame(df_, geometry= df_geo)
    
    df_join = (gpd.tools.sjoin(df_gpd, geo_dict["cap"], 
                             predicate="within", how='left'))
    CAP = df_join["CAP"]
    
    return CAP


def proportion_obs_houses(data, geo_dict):
    df_real = data["df_real"].copy()
    bins_mq = pd.qcut(df_real["log_mq"], q = 5, labels = False, duplicates = "drop", retbins = True)[1]
    bins_price = pd.qcut(df_real["log_price"], q = 5, labels = False, duplicates = "drop", retbins = True)[1]
    bins_mq[0] = -np.inf
    bins_mq[-1] = np.inf
    bins_price[0] = -np.inf
    bins_price[-1] = np.inf

    data_bins = {}
    for k in data:
        df_ = data[k].copy()
        df_["bin_mq"] = pd.cut(df_["log_mq"], bins = bins_mq)
        df_["bin_price"] = pd.cut(df_["log_price"], bins = bins_price)
        cap = get_cap_from_xy(df_, geo_dict)

        cap = cap.groupby(cap.index).first()

        df_["CAP"] = cap
        df_.drop(columns = ["x", "y", "log_price", "log_mq"], inplace = True)
        data_bins[k] = df_
    
    drop_cols = []
    df_real_list = [list(u) for _,u in data_bins["df_real"].drop(columns = drop_cols).iterrows()]
    in_real = {k: np.mean([list(data_bins[k].drop(columns = drop_cols).iloc[u,:]) in df_real_list for u in range(len(data_bins[k]))]) for k in data_bins}
    return in_real


if __name__ == "__main__":
    geo_dict = jl_vae.load_geo_data()            

    all_zero_cell = {}

    for file in sorted(glob(f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_*.pickle')):
        prov = file.split(".")[-2].split("_")[-1]
        print(prov)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        data = {k: data[k] for k in data if "95" not in k}

        prop_obs_prov = proportion_obs_houses(data, geo_dict)
        all_zero_cell[prov] = prop_obs_prov
        zero_cell_df = pd.DataFrame(all_zero_cell).T
        zero_cell_df.to_csv(f'/data/housing/data/intermediate/jl_pop_synth/zero_cell_ips.csv')

