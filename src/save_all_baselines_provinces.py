import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots as sbp
import sys
sys.path += ["../src"]
import jl_vae
from glob import glob
from jl_synthetic_pop_all_provinces import add_cat_features
from tqdm import tqdm
from _51_abm_functions import cod_prov_abbrv_df
import utils
import geopandas as gpd
import config

import pickle 

def select_n_in_border(df_sample, n, gdf, seed = 5):
    cols = df_sample.columns
    df_sample = df_sample.reset_index(drop = True).dropna(subset = ["log_mq", "log_price", "x", "y"])
    df_sample_points = gpd.GeoDataFrame(df_sample, geometry = gpd.points_from_xy(df_sample["x"], df_sample["y"]))
    df_sample = (gpd.tools.sjoin(df_sample_points, gdf, predicate="within", how='left')
                 .dropna(subset = ["prov_cod"])
                 .sample(n, random_state = seed, replace = True)
                 .reset_index(drop = True)
                 [cols])
    return df_sample


def save_province_baselines(prov):
    baseline_path = "/data/housing/data/intermediate/jl_pop_synth/pop_samples/baselines"
    cod_prov = cod_prov_abbrv_df.query("prov_abbrv == @prov")["COD_PROV"].iloc[0]

    df_real = pd.read_csv(jl_vae.path_pop_synth + f"real_populations/df_real_{prov}.csv", index_col = 0)
    df_real95 = df_real.sample(frac = 0.95, random_state = 1111)
    df_excluded = df_real[~df_real.index.isin(df_real95.index)]

    census_filter = census.assign(prov_cod = lambda x: [u[:3] for u in x["PRO_COM"]]).query("prov_cod == @cod_prov")
    polyg = census_filter[["prov_cod", "geometry"]].dissolve().to_crs('EPSG:4326')
    df_real =  pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/real_populations/df_real_{prov}.csv", index_col = 0)
    cols = [v for v in df_real.columns if v not in ['x_latent', 'x_norm', 'y_latent', 'y_norm', 'CAP', 'OMI_id', 'PRO_COM', 'SEZ2011', 'prov_abbrv']]

    df_real =  pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/real_populations/df_real_{prov}.csv", index_col = 0)
    df_real95 =  pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/real_populations/df_real_{prov}.csv", index_col = 0).sample(frac = 0.95, random_state = 1111)
    df_nfvae =  pd.read_csv(jl_vae.path_pop_synth + f"pop_samples/synthetic_pop_full_250811price_{prov}.csv", index_col = 0)
    df_nfvae95 =  pd.read_csv(jl_vae.path_pop_synth + f"95sample/pop_samples/synthetic_pop_full_250811price{prov}.csv", index_col = 0)
    df_ablation =  pd.read_csv(jl_vae.path_pop_synth + f"pop_samples_ablation/synthetic_pop_full_250811{prov}.csv", index_col = 0)
    df_ablation95 =  pd.read_csv(jl_vae.path_pop_synth + f"pop_samples_ablation/synthetic_pop_full_250811{prov}_95.csv", index_col = 0)
    #df_ipf =  pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample_{prov}_250718.csv", index_col = 0)
    #df_ipf95 =  pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample95_{prov}_250718.csv", index_col = 0)
    df_copula_nf =  pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_nf_{prov}.csv", index_col = 0)
    df_copula_nf95 =  pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_nf95_{prov}.csv", index_col = 0)
    df_copula_ablation =  pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_ablation_{prov}.csv", index_col = 0)
    df_copula_ablation95 =  pd.read_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_ablation95_{prov}.csv", index_col = 0)
    #df_ipf_shuffle = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample95_{prov}.csv")
    #df_ipf_shuffle95 = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_sample95_{prov}.csv")
    df_ipfs = {}
    df_ipfs95 = {}
    for province_shuffle in [0,1]:
        for with_bins in [0,1]:
            df_ipf = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_shuffle_{prov}_250822_provshuffle{province_shuffle}_bins_{with_bins}.csv", index_col = 0)
            df_ipfs[(province_shuffle, with_bins)] = df_ipf
                
            df_ipf95 = pd.read_csv(jl_vae.path_pop_synth + f"ipf_samples/df_shuffle95_{prov}_250822_provshuffle{province_shuffle}_bins_{with_bins}.csv", index_col = 0)
            df_ipfs95[(province_shuffle, with_bins)] = df_ipf95



    all_baselines = {
        "df_real": df_real,
        "df_real95": df_real95,
        "df_nfvae": df_nfvae,
        "df_nfvae95": df_nfvae95,
        "df_ablation": df_ablation,
        "df_ablation95": df_ablation95,
        #"df_ipf": df_ipf,
        #"df_ipf95": df_ipf95,
        "df_copula_nf": df_copula_nf,
        "df_copula_nf95": df_copula_nf95,
        "df_copula_ablation": df_copula_ablation,
        "df_copula_ablation95": df_copula_ablation95,
        #"df_ipf_shuffle": df_ipf_shuffle,
        #"df_ipf_shuffle95": df_ipf_shuffle95
        "df_shuffle_province_bins": df_ipfs[(1,1)],
        "df_shuffle_province_bins95": df_ipfs95[(1,1)],
        "df_shuffle_cap_bins": df_ipfs[(0,1)],
        "df_shuffle_cap_bins95": df_ipfs95[(0,1)],
        "df_shuffle_province_num": df_ipfs[(1,0)],
        "df_shuffle_province_num95": df_ipfs95[(1,0)],
        "df_shuffle_cap_num": df_ipfs[(0,0)],
        "df_shuffle_cap_num95": df_ipfs95[(0,0)],
    }

    for u in all_baselines:
        if ("95" in u)&("real" not in u):
            all_baselines[u] = select_n_in_border(all_baselines[u], n = len(df_real95), gdf = polyg, seed = 1996)
        if ("95" not in u)&("real" not in u):
            all_baselines[u] = select_n_in_border(all_baselines[u], n = len(df_real), gdf = polyg, seed = 1996)
        all_baselines[u] = all_baselines[u][cols]
        all_baselines[u][['flag_garage','flag_pertinenza','flag_air_conditioning','flag_multi_floor',
                        'ANNO_COSTRUZIONE_1500_1965','ANNO_COSTRUZIONE_1965_1985','ANNO_COSTRUZIONE_1985_2005','ANNO_COSTRUZIONE_2005_2025','ANNO_COSTRUZIONE_Missing',
                        'High_energy_class','Low_energy_class','Medium_energy_class','Missing_energy_class','COD_CAT_A02','COD_CAT_A03','COD_CAT_A_01_07_08','COD_CAT_A_04_05',
                        'floor_0.0','floor_1.0','floor_2.0','floor_3.0','floor_Missing','floor_plus_4','flag_air_conditioning_Missing','flag_multi_floor_Missing']].dtype = bool
        
    with open(f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_{prov}.pickle', 'wb') as handle:
        pickle.dump(all_baselines, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


if __name__ == "__main__":
    census = gpd.read_file(config.census_hydro_risk_path).to_crs('EPSG:3035') 
    census['PRO_COM_formated'] = census['PRO_COM'].apply(lambda x: '{:06d}'.format(int(x)))
    census=census.drop(columns = 'PRO_COM').rename(columns={'PRO_COM_formated':'PRO_COM'})
    census = census.loc[:,['SEZ2011','PRO_COM','geometry']] 
    census['SEZ2011'] =census['SEZ2011'].astype(str)

    geo_dict = jl_vae.load_geo_data()

    abm_path = "ISP_data_for_ABM/ISP_ABM_up_to_2024_08_0001.csv"
    data = pd.read_csv(jl_vae.path_intermediate + abm_path, index_col = 0)

    path_pop_synth95 = jl_vae.path_pop_synth + "95sample/"

    for prov in data["prov_abbrv"].sort_values().unique():
        try:
            print(prov)
            save_province_baselines(prov)
        except Exception as e:
            print(e)

    
