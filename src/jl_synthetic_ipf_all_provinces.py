import sys
sys.path += ["../src"]
import utils
from glob import glob
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.pyplot import subplots as sbp 
from importlib import reload
import jl_vae
import jl_nflows_geo_coordinates_2 as nfg
from jl_nflows_geo_coordinates import load_nf as load_dict

from _51_abm_functions import cod_prov_abbrv_df

# Global Spatial Autocorrelation
from spatial_autocorrelation import get_moransI, moransI_scatterplot, hypothesis_testing
# Local Spatial Autocorrelation
from spatial_autocorrelation import get_localMoransI, LISA_scatterplot
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import OneHotEncoder
from jl_synthetic_pop_all_provinces import add_cat_features
from ipfn import ipfn
import config

def transform_bins(df, vars):
    df1 = df.copy()
    for var in vars:
        df1[var + "_bin"] =  pd.qcut(df1[var], q = 5, labels = False).astype(int)
        df1.drop(columns = [var], inplace = True)
    return df1

def transform_bins_to_real(sample_df, df_real, var):
    bins_values = pd.qcut(df_real[var], q = 5, labels = False, retbins = True)[1]
    sample_df[var] = [np.random.uniform(low = bins_values[u], high = bins_values[u + 1]) for u in sample_df[var + "_bin"]]
    sample_df.drop(columns = [var + "_bin"], inplace = True)
    return sample_df

    

if __name__ == "__main__":
    geo_dict = jl_vae.load_geo_data()
    date = "250717"

    for _, (prov, cod_prov) in cod_prov_abbrv_df.sort_values("prov_abbrv").iterrows():
        print(prov)
        real_pops = jl_vae.path_pop_synth + f"pop_samples/pop_real_with_hedonic_price"
        df_real_ = pd.read_csv(real_pops + f"/pop_real_full_250110{prov}.csv", index_col = 0)
        df_real_ = add_cat_features(df_real_)
        df_real = df_real_.assign(log_price = lambda x: x["price_corrected"], CAP = lambda x: x["CAP"].astype(str))

        np.random.seed(2)
        ran_xy = np.random.uniform(low = df_real["x"].min(), high = df_real["x"].max(), size = 1000000), np.random.uniform(low = df_real["y"].min(), high = df_real["y"].max(), size = 1000000)

        df_locations = utils.spatial_matching_ABM(pd.DataFrame({"x": ran_xy[0], "y": ran_xy[1]}), geo_dict["hydro_risk"], geo_dict["census"], 
                                                  geo_dict["omi_og"], geo_dict["cap"]).query("prov_abbrv == @prov")
        
        df_real = df_real_.assign(log_price = lambda x: x["price_corrected"], CAP = lambda x: x["CAP"].astype(str))
        df_real = df_real[['flag_garage', 'flag_pertinenza', 'flag_air_conditioning',
            'flag_multi_floor', 
            'log_mq', 'log_price',
            'flag_air_conditioning_Missing', 
            # 'x', 'y', 
            'CAP',
            'energy_class', 
            'COD_CAT',
            'anno_costruzione']]
        
        df_sample_list = []

        for cap in df_real["CAP"].unique():
            df_cap = df_real.query("CAP == @cap").dropna()
            df_cap = transform_bins(df_cap, ["log_mq", "log_price"])
            N_cap = len(df_cap)
            df_counts = df_cap.value_counts().reset_index()
            cap_locs = df_locations.query("CAP == @cap").sample(n = N_cap)
            sample_df_cap = df_counts.sample(n = 10 * N_cap, weights = df_counts["count"],
                                         random_state = 2, replace = True).drop(columns = ["count"])

            sample_df_cap[["x", "y"]] = cap_locs[["GEO_LONGITUDINE_BENE_ROUNDED", "GEO_LATITUDINE_BENE_ROUNDED"]]
            sample_df_cap["CAP"] = cap
            sample_df_cap = transform_bins_to_real(sample_df_cap, df_real, "log_mq")
            sample_df_cap = transform_bins_to_real(sample_df_cap, df_real, "log_price")

            df_sample_list.append(sample_df_cap)
        sample_df = pd.concat(df_sample_list)
        sample_df.to_csv(jl_vae.path_pop_synth + f"ipf_samples/pop_sample_{prov}_{date}.csv")


        

            




