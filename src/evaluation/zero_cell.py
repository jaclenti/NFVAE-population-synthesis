# zero cell problem
# compute the proportion of houses of df_synthetic that is present in df_real
# to do so, we bin the continous variables in df_real and we assign df_synthetic to the same bins
# we associate xy coordinates with neighbourhood
# then we remove the continous variables
# and we compare the two dataframes

import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
import sys
sys.path += [".."]
sys.path += ["../src/generation"]
import config
from dataframe_from_insideairbnb import read_gz, read_json, get_df_city_for_nf
import pickle
from generate_all_baselines import get_area_from_xy


def proportion_obs_houses(data, gdf):
    df_real = data["df_real"].copy()
    bins_rscore = pd.qcut(df_real["review_scores_rating"], q = 5, labels = False, duplicates = "drop", retbins = True)[1]
    bins_rmonth = pd.qcut(df_real["reviews_per_month"], q = 5, labels = False, duplicates = "drop", retbins = True)[1]
    bins_price = pd.qcut(df_real["log_price"], q = 5, labels = False, duplicates = "drop", retbins = True)[1]
    bins_rscore[0] = -np.inf
    bins_rscore[-1] = np.inf
    bins_rmonth[0] = -np.inf
    bins_rmonth[-1] = np.inf
    bins_price[0] = -np.inf
    bins_price[-1] = np.inf

    data_bins = {}
    for k in data:
        df_ = data[k].copy()
        df_["bin_rscore"] = pd.cut(df_["review_scores_rating"], bins = bins_rscore)
        df_["bin_rmonth"] = pd.cut(df_["reviews_per_month"], bins = bins_rmonth)
        df_["bin_price"] = pd.cut(df_["log_price"], bins = bins_price)
        neighbourhood = get_area_from_xy(df_, gdf, var_area = "neighbourhood", epsg = 4326)

        df_["neighbourhood"] = neighbourhood
        df_.drop(columns = ["x", "y", "review_scores_rating", "reviews_per_month",  "log_price"], inplace = True)
        data_bins[k] = df_
    
    drop_cols = []
    df_real_list = [list(u) for _,u in data_bins["df_real"].drop(columns = drop_cols).iterrows()]
    in_real = {k: np.mean([list(data_bins[k].drop(columns = drop_cols).iloc[u,:]) in df_real_list for u in range(len(data_bins[k]))]) for k in data_bins}
    return in_real


if __name__ == "__main__":
    all_zero_cell = {}


    for file in sorted(glob(config.baselines_path + f'all_baselines_*.pickle')):
        city = file.split(".")[-2].split("_")[-1]
        print(city)
        all_zero_cell[city] = {}

        with open(file, 'rb') as f:
            data = pickle.load(f)

        data = {k: data[k] for k in data if "95" not in k}
        gdf = read_json(city)

        prop_obs_prov = proportion_obs_houses(data, gdf)
        all_zero_cell[city] = prop_obs_prov
    zero_cell_df = pd.DataFrame(all_zero_cell)
    zero_cell_df.to_csv(config.path_airbnb + f'zero_cell_airbnb.csv')

