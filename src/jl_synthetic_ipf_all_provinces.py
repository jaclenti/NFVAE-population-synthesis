import sys
sys.path += ["../src"]
import utils
from glob import glob
import jl_vae
from _51_abm_functions import cod_prov_abbrv_df
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from jl_synthetic_pop_all_provinces import add_cat_features
import geopandas as gpd

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
        _, bins_values =  pd.qcut(df1[var], q = 5, labels = False, duplicates = "drop", retbins = True)
        bins_values[0] = -np.inf
        bins_values[-1] = np.inf
        df1[var + "_bin"] = pd.cut(df1[var], bins = bins_values, labels = np.arange(len(bins_values) - 1)).astype(int)
        df1.drop(columns = [var], inplace = True)
    return df1

def transform_bins_to_real(sample_df_, df_real, vars):
    sample_df = sample_df_.copy()

    for var in vars:
        _, bins_values = pd.qcut(df_real[var], q = 5, labels = False, duplicates = "drop", retbins = True)
        np.random.seed(2)
        sample_df[var] = [np.random.uniform(low = bins_values[u], high = bins_values[u + 1]) for u in sample_df[var + "_bin"]]
        sample_df.drop(columns = [var + "_bin"], inplace = True)
    return sample_df    


def get_area_from_xy(df, gdf_area, var_area = "CAP", epsg = 3035):
    df_geo = gpd.points_from_xy(df['x'], df['y'], z=None, crs="EPSG:4326")
    df_geo = df_geo.to_crs(f'EPSG:{epsg}') #4326 3035
    if var_area in df.columns:
        df_ = df.drop(columns = [var_area]).copy()
    else:
        df_ = df.copy()
    df_gpd = gpd.GeoDataFrame(df_, geometry= df_geo)
    
    df_join = (gpd.tools.sjoin(df_gpd, gdf_area, 
                             predicate="within", how='left'))
    area = df_join[var_area]


    
    area = area.groupby(area.index).first()
    
    return area

def create_pop(df_real, with_bins, province_shuffle, 
               gdf_area, 
               var_bins = ["log_price", "log_mq"], 
               var_area = "CAP",
               ratio_syn_real = 10,
               epsg = 3035):
    df_bin = df_real.copy()
    if with_bins:
        df_bin = transform_bins(df_real, vars = var_bins)
    CAP_real = get_area_from_xy(df_real, gdf_area, var_area = var_area, epsg = epsg)
    df_bin[var_area] = CAP_real
    df_sample = df_bin.dropna(subset = [var_area]).sample(n = ratio_syn_real * len(df_bin), replace = True).sort_values(var_area).reset_index(drop = True)
    

    np.random.seed(2)
    #x_min, y_min, x_max, y_max = gdf_area.to_crs("EPSG:4326").total_bounds
    x_min, y_min, x_max, y_max = df_real["x"].min() - 1, df_real["y"].min() - 1, df_real["x"].max() + 1, df_real["y"].max() + 1
    ran_xy = np.random.uniform(low = x_min, high = x_max, size = 1000000), np.random.uniform(low = y_min, high = y_max, size = 1000000)
    df_locations = pd.DataFrame({"x": ran_xy[0], "y": ran_xy[1]})
    df_locations[var_area] = get_area_from_xy(df_locations, gdf_area, var_area = var_area, epsg = epsg)
    df_locations = df_locations.dropna(subset = [var_area])
            
    ran_coords = []

    if province_shuffle:
        ran_coords.append(df_locations.sample(n = len(df_sample), replace = True))
    else:
        for area in df_sample[var_area].unique():
            ran_coords.append(df_locations.loc[df_locations[var_area] == area].sample(n = len(df_sample.loc[df_sample[var_area] == area]), replace = True))
    
    df_shuffle = (pd.concat([df_sample.drop(columns = ["x", "y", var_area]),
                            pd.concat(ran_coords).reset_index(drop = True)], axis = 1)
                            .drop(columns  = [var_area]))
    if with_bins:
        df_shuffle = transform_bins_to_real(df_shuffle, df_real, var_bins)
    return df_shuffle


if __name__ == "__main__":
    geo_dict = jl_vae.load_geo_data()
    date = "250822"

    for _, (prov, cod_prov) in cod_prov_abbrv_df.sort_values("prov_abbrv").iterrows():
        print(prov)
        try:
            #df_real = pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/real_populations/df_real_{prov}.csv", index_col = 0)[features_synpop + ["CAP"]]
            #df_real95 = df_real.sample(frac = 0.95, random_state = 1111)

            file = f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_{prov}.pickle'

            with open(file, 'rb') as f:
                data = pickle.load(f)


            for province_shuffle in [0,1]:
                for with_bins in [0,1]:
                    for df_name in ["df_real", "df_real95"]:
                        df_shuffle = create_pop(data[df_name], 
                                                with_bins = with_bins, 
                                                province_shuffle = province_shuffle, 
                                                gdf_area = geo_dict["cap"])
                        df_shuffle.to_csv(jl_vae.path_pop_synth + f"ipf_samples/df_shuffle{df_name.split('real')[-1]}_{prov}_{date}_provshuffle{province_shuffle}_bins_{with_bins}.csv")
        except Exception as e:
            print(e)


            

                




