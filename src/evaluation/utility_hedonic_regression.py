
# compute utility - train on synthetic test on real
# train a hedonic regression with spatial fixed effects on synthetic data
# and predict home prices on real data
# compute R^2 between home price real and predicted by the model
# the absolute difference between R^2 of model trained on real and on synthetic data measures the utility
# if this is close to 0, it means that synthetic data are as useful as real data in tackling the task of
# predicting home prices with hedonic regression with spatial fixed effects


import sys
sys.path += [".."]
sys.path += ["../src/generation"]
import config
import pandas as pd
import numpy as np
from glob import glob
import pickle
from tqdm import tqdm
import geopandas as gpd
import pyfixest as pf
from sklearn.metrics import r2_score
from generate_all_baselines import get_area_from_xy
from dataframe_from_insideairbnb import read_gz, read_json


def process_data_for_hedonic(df):
    df = df.fillna(0) # probably here there is the problem
    df = df.loc[:,df.sum()!=0]
    df = df.drop(columns= [[u for u in df.columns if var in u][0] for var in cat_variables])
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("-", "_")
    df["neighbourhood"] = get_area_from_xy(df, gdf_full, "neighbourhood")
    return df
                

if __name__ == "__main__":
    # this cell takes 16 min
    key2keep = ['df_real','df_nfvae','df_vae','df_nf_copula','df_copula','df_shuffle_global','df_shuffle_local']

    R2_dict = pd.DataFrame()

    fix_eff = "neighbourhood"
    cat_variables = ["roomtype"]
    

    base_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'Air conditioning',
        'Elevator', 'Self check-in', 'Pets allowed', 'Private living room',
        'Backyard', 'Pool', #'roomtype_Entire home/apt', 
        'roomtype_Hotel room',
        'roomtype_Private room', 'roomtype_Shared room', 'review_scores_rating',
        'reviews_per_month',fix_eff]

    base_features =' + '.join(base_features)
    regr_formula = "log_price ~ "+base_features #+" | " + fix_eff


    for file in tqdm(sorted(glob(config.baselines_path + f'all_baselines_*.pickle'))):
        city = file.split('-')[-2].split('_')[-1]
        print(city)

        # data loading
        with open(file, 'rb') as f:
            all_baselines = pickle.load(f)

        gdf_full = read_json(city) # load neighbourhood shapefile

        
        for k in key2keep:
            
            real_df = process_data_for_hedonic(all_baselines["df_real"])            
            cols = [col for col in list(real_df.columns) if col not in ['log_price','x', 'y']]
            # doing the regression equation
            base_features =' + '.join(cols)
            regr_formula = "log_price ~ " + base_features
                
            # regressions
            regr = pf.feols(fml=regr_formula, data=real_df, drop_intercept=False,vcov = {"CRV1":fix_eff})
            R2_dict.loc[city,k] = regr._r2
            
            for k in all_baselines:
                syn_df = process_data_for_hedonic(all_baselines[k])
                cols = [col for col in list(syn_df.columns) if col not in ['log_price','x', 'y']]
                base_features =' + '.join(cols)
                regr_formula = "log_price ~ "+base_features

                real_df_ = real_df.loc[:,real_df.columns.isin(syn_df.columns)]
                syn_df_ = syn_df.loc[syn_df.neighbourhood.isin(real_df.neighbourhood.unique()),:]
                regr = pf.feols(fml=regr_formula, data=syn_df_, drop_intercept=False,vcov = {"CRV1":fix_eff})
                pred_price = regr.predict(real_df_)
                mask = np.isnan(pred_price)
                if np.sum(mask)/len(mask) <= 0.05:
                    #print("fraction of Nans: ",np.sum(mask)/len(mask))
                    R2_dict.loc[city,k] = r2_score(real_df_.log_price[~mask], pred_price[~mask])
                else:
                    print(city + ' too many nans')
                
    R2_dict.to_csv(config.path_airbnb + 'airbnb_utility.csv')