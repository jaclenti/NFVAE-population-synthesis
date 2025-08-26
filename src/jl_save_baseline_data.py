import sys
sys.path += ["../src"]
import utils
from glob import glob
import jl_vae
import jl_nflows_geo_coordinates_2 as nfg
from jl_nflows_geo_coordinates import load_nf as load_dict

from _51_abm_functions import cod_prov_abbrv_df

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import OneHotEncoder
from jl_synthetic_pop_all_provinces import add_cat_features
from ipfn import ipfn
import config

import geopandas as gpd

cat_synpop = ['flag_garage', 'flag_pertinenza', 'flag_air_conditioning',
            'flag_multi_floor', 'log_mq', 'anno_costruzione',
            'energy_class',
            'COD_CAT', 'floor', 
            'flag_air_conditioning_Missing', 'flag_multi_floor_Missing', 'y', 'x','log_price']


def filter_houses_prov(df, df_real, geo_dict, prov):
    df_out = (utils.spatial_matching_ABM(df,
                                hydro_risk = geo_dict["hydro_risk"],
                                census = geo_dict["census"],
                                omi_og = geo_dict["omi_og"],
                                cap = geo_dict["cap"])
                                .rename(columns = {"GEO_LONGITUDINE_BENE_ROUNDED":"x", "GEO_LATITUDINE_BENE_ROUNDED":"y"})
                                .sample(n = len(df_real), random_state = 1807, replace = True)
                                .query("prov_abbrv == @prov")
                                )
    return df_out
    
baseline_path = "/data/housing/data/intermediate/jl_pop_synth/pop_samples/baselines"

if __name__ == "__main__":
    census = gpd.read_file(config.census_hydro_risk_path).to_crs('EPSG:3035') 
    census['PRO_COM_formated'] = census['PRO_COM'].apply(lambda x: '{:06d}'.format(int(x)))
    census=census.drop(columns = 'PRO_COM').rename(columns={'PRO_COM_formated':'PRO_COM'})
    census = census.loc[:,['SEZ2011','PRO_COM','geometry']] 
    census['SEZ2011'] =census['SEZ2011'].astype(str)

    geo_dict = jl_vae.load_geo_data()


    for _, (prov, cod_prov) in cod_prov_abbrv_df.sort_values("prov_abbrv").iterrows():
        print(prov)

        try:
            df_real = pd.read_csv(jl_vae.path_pop_synth + f"real_populations/df_real_{prov}.csv", index_col = 0)
            df_real = add_cat_features(df_real)[cat_synpop]
            df_real95 = df_real.sample(frac = 0.95, random_state = 1111)
            df_excluded = df_real[~df_real.index.isin(df_real95.index)]


            df_nfvae = pd.read_csv(jl_vae.path_pop_synth + f"pop_samples/synthetic_pop_full_250709price_{prov}.csv", index_col = 0)
            df_nfvae = filter_houses_prov(df_nfvae, df_real, geo_dict, prov)
            df_nfvae.to_csv(baseline_path + f"/df_nfvae_{prov}.csv")
            
            df_vae = pd.read_csv(jl_vae.path_pop_synth + f"pop_samples_ablation/synthetic_pop_full_250710{prov}.csv", index_col = 0)
            df_vae = filter_houses_prov(df_vae, df_real, geo_dict, prov)
            df_vae.to_csv(baseline_path + f"/df_vae_{prov}.csv")

            df_nfvae95 = pd.read_csv(jl_vae.path_pop_synth + f"95sample/pop_samples/synthetic_pop_full_250703{prov}.csv", index_col = 0)
            df_nfvae95 = filter_houses_prov(df_nfvae95, df_real, geo_dict, prov)
            df_nfvae95.to_csv(baseline_path + f"/df_nfvae95_{prov}.csv")

            df_copula_ablation = pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_ablation_{prov}.csv", index_col = 0)
            df_copula_ablation = filter_houses_prov(df_copula_ablation, df_real, geo_dict, prov)
            df_copula_ablation.to_csv(baseline_path + f"/df_copula_ablation_{prov}.csv")
            
            df_copula_nf = pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_nf_{prov}.csv", index_col = 0)
            df_copula_nf = filter_houses_prov(df_copula_nf, df_real, geo_dict, prov)
            df_copula_nf.to_csv(baseline_path + f"/df_copula_nf_{prov}.csv")
            
            df_copula_ablation95 = pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_ablation95_{prov}.csv", index_col = 0)
            df_copula_ablation95 = filter_houses_prov(df_copula_ablation95, df_real, geo_dict, prov)
            df_copula_ablation95.to_csv(baseline_path + f"/df_copula_ablation95_{prov}.csv")
            
            df_copula_nf95 = pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_nf95_{prov}.csv", index_col = 0)
            df_copula_nf95 = filter_houses_prov(df_copula_nf95, df_real, geo_dict, prov)
            df_copula_nf95.to_csv(baseline_path + f"/df_copula_nf95_{prov}.csv")

            df_ipf = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample_{prov}_250718.csv", index_col = 0)
            df_ipf = filter_houses_prov(df_ipf, df_real, geo_dict, prov)
            df_ipf.to_csv(baseline_path + f"/df_ipf_{prov}.csv")
            
            df_ipf95 = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample95_{prov}_250718.csv", index_col = 0)
            df_ipf95 = filter_houses_prov(df_ipf95, df_real, geo_dict, prov)
            df_ipf95.to_csv(baseline_path + f"/df_ipf95_{prov}.csv")

            df_ipf_shuffle = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample95_{prov}_250718.csv", index_col = 0)
            df_ipf_shuffle = filter_houses_prov(df_ipf95, df_real, geo_dict, prov)
            df_ipf_shuffle.to_csv(baseline_path + f"/df_ipf95_{prov}.csv")
            df_ipf_shuffle95 = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample95_{prov}_250718.csv", index_col = 0)
            df_ipf_shuffle95 = filter_houses_prov(df_ipf95, df_real, geo_dict, prov)
            df_ipf_shuffle95.to_csv(baseline_path + f"/df_ipf95_{prov}.csv")

            
            print("done")
        except Exception as e:
            print(e)


            


                