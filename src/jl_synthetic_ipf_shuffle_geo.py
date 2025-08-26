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


features_synpop = ['flag_garage', 'flag_pertinenza', 'flag_air_conditioning',
            'flag_multi_floor', 'log_mq', 'ANNO_COSTRUZIONE_1500_1965',
            'ANNO_COSTRUZIONE_1965_1985', 'ANNO_COSTRUZIONE_1985_2005',
            'ANNO_COSTRUZIONE_2005_2025', 'ANNO_COSTRUZIONE_Missing',
            'High_energy_class', 'Low_energy_class', 'Medium_energy_class',
            'Missing_energy_class', 'COD_CAT_A02', 'COD_CAT_A03',
            'COD_CAT_A_01_07_08', 'COD_CAT_A_04_05', 'floor_0.0', 'floor_1.0',
            'floor_2.0', 'floor_3.0', 'floor_Missing', 'floor_plus_4',
            'flag_air_conditioning_Missing', 'flag_multi_floor_Missing', 'y', 'x','log_price']


def transform_bins(df, vars):
    df1 = df.copy()
    for var in vars:
        df1[var + "_bin"] =  pd.qcut(df1[var], q = 5, labels = False, duplicates = "drop").astype(int)
        df1.drop(columns = [var], inplace = True)
    return df1

def transform_bins_to_real(sample_df, df_real, var):
    bins_values = pd.qcut(df_real[var], q = 5, labels = False, duplicates = "drop", retbins = True)[1]
    np.random.seed(2)
    sample_df[var] = [np.random.uniform(low = bins_values[u], high = bins_values[u + 1]) for u in sample_df[var + "_bin"]]
    sample_df.drop(columns = [var + "_bin"], inplace = True)
    return sample_df

    

if __name__ == "__main__":
    geo_dict = jl_vae.load_geo_data()
    date = "250718"

    for _, (prov, cod_prov) in cod_prov_abbrv_df.sort_values("prov_abbrv").iterrows():
        print(prov)
        try:
            df_real = pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/real_populations/df_real_{prov}.csv", index_col = 0)[features_synpop + ["CAP"]]
            df_real95 = df_real.sample(frac = 0.95, random_state = 1111)

            sample_locs = True
            np.random.seed(2)
            # matched_df = []
            # while sample_locs:
            #     ran_xy = np.random.uniform(low = df_real["x"].min(), high = df_real["x"].max(), size = 1000000), np.random.uniform(low = df_real["y"].min(), high = df_real["y"].max(), size = 1000000)

            #     matched_df.append(utils.spatial_matching_ABM(pd.DataFrame({"x": ran_xy[0], "y": ran_xy[1]}), geo_dict["hydro_risk"], geo_dict["census"], 
            #                                             geo_dict["omi_og"], geo_dict["cap"]).query("prov_abbrv == @prov"))
            #     df_locations = pd.concat(matched_df)
            #     if len(df_locations["CAP"].unique()) >= len(df_real["CAP"].unique()):
            #         sample_locs = False

            ran_xy = np.random.uniform(low = df_real["x"].min(), high = df_real["x"].max(), size = 1000000), np.random.uniform(low = df_real["y"].min(), high = df_real["y"].max(), size = 1000000)

            df_locations = utils.spatial_matching_ABM(pd.DataFrame({"x": ran_xy[0], "y": ran_xy[1]}), geo_dict["hydro_risk"], geo_dict["census"], 
                                                         geo_dict["omi_og"], geo_dict["cap"]).query("prov_abbrv == @prov")
            
            

            sample_dfs = []
            for df_ in [df_real.copy(), df_real95.copy()]:
                df_ = df_.dropna(subset = "CAP").assign(CAP = lambda x: [f"{int(u):05d}" for u in x["CAP"]])
                df_.drop(columns = ["x", "y"], inplace = True)
            
                df_sample_list = []
            
                for cap in df_["CAP"].unique():
                    locations_cap = df_locations.query("CAP == @cap")
                    df_cap = df_.query("CAP == @cap").dropna()
                    N_cap = len(df_cap)
                    if N_cap > 1:
                        df_cap = transform_bins(df_cap, ["log_mq", "log_price"])            
                        df_counts = df_cap.value_counts().reset_index()
                        sample_df_cap = df_counts.sample(n = 10 * N_cap, weights = df_counts["count"],
                                                    random_state = 2, replace = True).drop(columns = ["count"])

                        sample_df_cap = transform_bins_to_real(sample_df_cap, df_, "log_mq")
                        sample_df_cap = transform_bins_to_real(sample_df_cap, df_, "log_price")
                    else:
                        sample_df_cap = pd.concat([df_cap] * 10)
                    
                    if len(locations_cap) > 0:
                        cap_locs = locations_cap.sample(n = 10 * N_cap, replace = True, random_state = 2)
                        sample_df_cap["x"] = list(cap_locs["GEO_LONGITUDINE_BENE_ROUNDED"])
                        sample_df_cap["y"] = list(cap_locs["GEO_LATITUDINE_BENE_ROUNDED"])
                        sample_df_cap["CAP"] = cap
                    
                    df_sample_list.append(sample_df_cap)
                sample_df = pd.concat(df_sample_list)
                sample_dfs.append(sample_df)
            sample_dfs[0].to_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample_{prov}_{date}.csv")
            sample_dfs[1].to_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample95_{prov}_{date}.csv")
        except Exception as e:
            print(e)


            

                




