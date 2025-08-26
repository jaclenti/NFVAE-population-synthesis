import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
import sys
sys.path += ["../src"]
from jl_synthetic_pop_all_vae_provinces_airbnb import read_gz, read_json, get_df_city_for_nf
import pickle


def get_neighbourhood_from_xy(df, gdf):
    df_geo = gpd.points_from_xy(df['x'], df['y'], z=None, crs="EPSG:3035")
    df_geo = df_geo.to_crs('EPSG:3035') #4326 3035
    if "neighbourhood" in df.columns:
        df_ = df.drop(columns = ["neighbourhood"]).copy()
    else:
        df_ = df.copy()
    df_gpd = gpd.GeoDataFrame(df_, geometry= df_geo)
    
    df_join = (gpd.tools.sjoin(df_gpd, gdf, 
                             predicate="within", how='left'))
    neighbourhood = df_join["neighbourhood"]
    
    return neighbourhood

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
        neighbourhood = get_neighbourhood_from_xy(df_, gdf)

        df_["neighbourhood"] = neighbourhood
        df_.drop(columns = ["x", "y", "review_scores_rating", "reviews_per_month",  "log_price"], inplace = True)
        data_bins[k] = df_
    
    drop_cols = []
    df_real_list = [list(u) for _,u in data_bins["df_real"].drop(columns = drop_cols).iterrows()]
    in_real = {k: np.mean([list(data_bins[k].drop(columns = drop_cols).iloc[u,:]) in df_real_list for u in range(len(data_bins[k]))]) for k in data_bins}
    return in_real


if __name__ == "__main__":
    all_zero_cell = {}


    for file in sorted(glob(f'/data/housing/data/intermediate/jl_pop_synth/airbnb_baselines/all_baselines_*.pickle')):
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
        zero_cell_df.to_csv(f'/data/housing/data/intermediate/jl_pop_synth/zero_cell_airbnb.csv')

