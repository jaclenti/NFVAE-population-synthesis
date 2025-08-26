
import sys
sys.path += ["../src"]
import utils
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.pyplot import subplots as sbp 
from importlib import reload
import jl_vae
import shapely
import scipy.stats
import geopandas as gpd
import jl_nflows_geo_coordinates_2 as nfg
from jl_nflows_geo_coordinates import load_nf as load_dict
import ot
from _51_abm_functions import cod_prov_abbrv_df

# Global Spatial Autocorrelation
from spatial_autocorrelation import get_moransI, moransI_scatterplot, hypothesis_testing
# Local Spatial Autocorrelation
from spatial_autocorrelation import get_localMoransI, LISA_scatterplot
import pickle

import ot
from joblib import Parallel, delayed
import os
#%% FUNCTIONS

def Find_province_shapefile(prov_code,reg_code):

    # fill reg_code 0 

    reg_code = str(reg_code).zfill(2)  # ensure reg_code is two digits

    region_shapefile = gpd.read_file(f'/data/housing/data/input/census/shapefiles_2011/R{reg_code}_11_WGS84/R{reg_code}_11_WGS84.shp')
    region_shapefile['PRO_COM'] = region_shapefile['PRO_COM'].apply(lambda x: '{:06d}'.format(int(x)))
    #province 
    prov_code =str(prov_code).zfill(3)  # ensure prov_code is three digits
    #keep 3 first digits of PRO_COM as COD_PROV
    region_shapefile['COD_PROV'] = region_shapefile['PRO_COM'].str[0:3]
    # filter prov
    province_shapefile= region_shapefile.loc[region_shapefile.COD_PROV == prov_code, "geometry"].reset_index().dissolve()
    province_shapefile.to_crs(epsg=4326, inplace=True)
    province_shapefile['index']=1
    province_shapefile.rename(columns={'index': 'in_province'}, inplace=True)  #needed for the next steps

    return province_shapefile

def add_cat_features(df):
    df["energy_class"] = df[[u for u in df.columns if "_energy" in u]].stack().rename("col").reset_index().query("col == 1")["level_1"]
    df["COD_CAT"] = [u[8:] for u in df[[u for u in df.columns if "COD_CAT_" in u]].stack().rename("col").reset_index().query("col == 1")["level_1"]]
    df["anno_costruzione"] = [u[17:] for u in df[[u for u in df.columns if "ANNO_COSTRUZIONE" in u]].stack().rename("col").reset_index().query("col == 1")["level_1"]]
    return df

def Shuffle_attributes(data, attributes_shuffle, rand_geo, rand_attributes, province_shapefile=None):
    if rand_attributes: 
        for attribute in attributes_shuffle: 
            if attribute !='num_homes': 
                data[attribute] = np.random.permutation(data[attribute].values)
        
    if rand_geo: 
        # Randomly shuffle the geometry of the homes
        geo =  province_shapefile.sample_points(len(data)).explode('geometry').reset_index(drop=True)
        data['x'] = geo.x
        data['y'] = geo.y
    return data


#%%

# LOAD DATA
def SWD_for_province(prov,cod_prov_abbrv_df):
    """
    Load the spatial weights data for a given province.
    """
    print('province:', prov)
    with open(f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_{prov}.pickle', 'rb') as f:
        all_baselines = pickle.load(f)

    
    df_real = all_baselines['df_real']
    df_nfvae= all_baselines['df_nfvae']
    df_ablation= all_baselines['df_ablation']
    df_ipf = all_baselines['df_ipf']
    df_copula_nf = all_baselines['df_copula_nf']
    df_copula_ablation = all_baselines['df_copula_ablation']

    # Variables
    continuous_var = ['log_mq','log_price']
    ignore = ['x', 'y', 'SEZ2011', 'PRO_COM', 'CAP', 'OMI_id', 'prov_abbrv', 'energy_class', 'COD_CAT',
        'anno_costruzione', 'COD_REG', 'COD_PROV']

    categoricals = list(set(df_real.columns).difference(set(continuous_var + ignore)))


    # transform categoricals to numeric
    for col in categoricals:
        if col in df_real.columns:
            df_real[col] = df_real[col].astype(float)
        if col in df_nfvae.columns:
            df_nfvae[col] = df_nfvae[col].astype(float)
        if col in df_copula_ablation.columns:
            df_copula_ablation[col] = df_copula_ablation[col].astype(float)
        if col in df_copula_nf.columns:
            df_copula_nf[col] = df_copula_nf[col].astype(float)
        if col in df_ablation.columns:
            df_ablation[col] = df_ablation[col].astype(float)
        if col in df_ipf.columns:
            df_ipf[col] = df_ipf[col].astype(float)
    
    # Add cod_reg to data 
    metadata  = pd.read_csv('/data/housing/data/intermediate/master_table_locations.csv')
    cod_prov = int(cod_prov_abbrv_df.query("prov_abbrv == @prov")["COD_PROV"].iloc[0])

    metadata_reg = pd.read_csv('/data/housing/data/intermediate/master_table_regressions.csv')
    cod_reg = metadata_reg.loc[metadata_reg['COD_PROV'] == cod_prov, 'COD_REG'].unique()[0]

    # Add categorical features
    df_real = add_cat_features(df_real)
    df_nfvae = add_cat_features(df_nfvae)
    df_ablation = add_cat_features(df_ablation)
    df_copula_ablation = add_cat_features(df_copula_ablation)
    df_copula_nf = add_cat_features(df_copula_nf)
    df_ipf = add_cat_features(df_ipf)


    # Cut province shapefile
    province_shapefile = Find_province_shapefile(cod_prov,cod_reg)

    # Add categorical features
    df_real = add_cat_features(df_real)
    df_nfvae = add_cat_features(df_nfvae)
    df_ablation = add_cat_features(df_ablation)
    df_copula_ablation = add_cat_features(df_copula_ablation)
    df_copula_nf = add_cat_features(df_copula_nf)
    df_ipf = add_cat_features(df_ipf)



    agg_rules = {
        'num_homes': 'sum',
        'flag_garage': 'mean',
        'flag_air_conditioning': 'mean',
        'flag_multi_floor': 'mean',
        'flag_pertinenza': 'mean',
        'log_mq': 'mean',
        'COD_CAT': 'first',
        'anno_costruzione': 'first',
        'energy_class': 'first',
    }

    # Generate 2 null models: 
    # 1. Randomly shuffle the geographical coordinates of the homes
    df_null = Shuffle_attributes(df_real.copy(deep=True).reset_index(drop=True), agg_rules.keys(), rand_geo = True, rand_attributes =False, province_shapefile=province_shapefile)


    # Estimate SWD
    SWD = {
        'real_vs_nfvae' : ot.sliced_wasserstein_distance(df_real[['x', 'y']].to_numpy(),df_nfvae[['x', 'y']].to_numpy(), n_projections=n_projections),
        'real_vs_ablation': ot.sliced_wasserstein_distance(df_real[['x', 'y']].to_numpy(),df_ablation[['x', 'y']].to_numpy(), n_projections=n_projections),
        'real_vs_copula': ot.sliced_wasserstein_distance(df_real[['x', 'y']].to_numpy(),df_copula_ablation[['x', 'y']].to_numpy(), n_projections=n_projections),
        'real_vs_copula_nf': ot.sliced_wasserstein_distance(df_real[['x', 'y']].to_numpy(),df_copula_nf[['x', 'y']].to_numpy(), n_projections=n_projections),
        'real_vs_ipf': ot.sliced_wasserstein_distance(df_real[['x', 'y']].to_numpy(),df_ipf[['x', 'y']].to_numpy(), n_projections=n_projections),
        'real_vs_null' : ot.sliced_wasserstein_distance(df_real[['x', 'y']].to_numpy(),df_null[['x', 'y']].to_numpy(), n_projections=n_projections),
        'nfvae_vs_null' : ot.sliced_wasserstein_distance(df_nfvae[['x', 'y']].to_numpy(),df_null[['x', 'y']].to_numpy(), n_projections=n_projections)}#,
        #'real_vs_shuffle' :  ot.sliced_wasserstein_distance(df_real[['x', 'y']].to_numpy(),df_feat_shuffle[['x', 'y']].to_numpy(), n_projections=n_projections),
        #'sample_vs_shuffle' : ot.sliced_wasserstein_distance(df_nfvae[['x', 'y']].to_numpy(),df_feat_shuffle[['x', 'y']].to_numpy(), n_projections=n_projections)}
    SWD_df = pd.DataFrame.from_dict(SWD, orient='index', columns=['Wasserstein distance']).reset_index()
    SWD_df['province'] = prov
    return SWD_df


# list all files folder

list_files = os.listdir('/data/housing/data/intermediate/jl_pop_synth/isp_baselines/')
# keep only province name from files    
list_files = [u.split('_')[2] for u in list_files if u.endswith('.pickle')]
prov_list = [u.split('.')[0] for u in list_files]

# loop for prov list

n_cores =3
n_projections = 1000
SWD_results = pd.concat(Parallel(n_jobs=n_cores)(delayed(SWD_for_province)(prov,cod_prov_abbrv_df) for prov in prov_list))

# save to txt
SWD_results.to_csv('/data/housing/data/intermediate/jl_pop_synth/Sliced_WD_all_prov.txt', index=False)

#%%
# append all dataframes into one
#SWD_results = [SWD_for_province(prov, cod_prov_abbrv_df) for prov in prov_list[0:3]]

# %%
