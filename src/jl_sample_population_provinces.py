import pandas as pd
import numpy as np
from glob import glob
import sys
sys.path += ["../src"]
import jl_vae

import jl_nflows_geo_coordinates_2 as nfg
import utils
from jl_nflows_geo_coordinates import load_nf as load_dict
import config
import pickle
#import dcor

from pyfixest.estimation import feols
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def HedonicRegression_PriceTrain (train_data,sp_fix_eff,save):
    """
    Function to compute the hedonic price of the houses in the synthetic population
    Inputs:
     - train_data: ISP data
     - sp_fix_eff: spatial fixed effects (we start using PRO_COM)
     - save: flag that is True if you want to save the regression parameters in this file f'/data/housing/data/intermediate/HedonicRegressionABM/reg_risk_{sp_fix_eff}.pkl'. 

    Outputs:
     - reg_risk: regression objetc
    """
    # Drop colinearity variables
    drop_coliniarity= ['ANNO_COSTRUZIONE_1500_1965','Medium_energy_class',
                        'COD_CAT_A_04_05','floor_0.0','year_erogaz_2016']
    train_data = train_data.drop(columns=drop_coliniarity)

    # Add dummies to formula: 
    floor_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('floor_')]))
    year_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('year_erogaz_20')]))
    energy_class_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('_energy_class')]))
    ANNO_COSTRUZIONE_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('ANNO_COSTRUZIONE')]))
    COD_CAT_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('COD_CAT_')]))

    dummies_vec = COD_CAT_dummies + '+' + ANNO_COSTRUZIONE_dummies + '+' + energy_class_dummies + ' + ' + floor_dummies + '+' + year_dummies

    # Regression specification
    base_features = 'flag_garage + flag_pertinenza + flag_air_conditioning + flag_multi_floor +' + dummies_vec
    mq_feature = 'log_mq' 

    eq_regr_risk = 'log_price ~ scenario_Risk +' + mq_feature + '+' + base_features + '|' +  sp_fix_eff  # + C(month_erogaz) year_erogaz_prov +
    
    # Train regression
    reg_risk = feols(fml=eq_regr_risk, data=train_data, drop_intercept=False ,vcov = {"CRV1":sp_fix_eff})
    if save==True:
        model_state = {"_beta_hat": reg_risk._beta_hat,
                    "_coefnames": reg_risk._coefnames,
                    "_se": reg_risk._se,
                    "_tstat": reg_risk._tstat,
                    "_pvalue": reg_risk._pvalue,
                    "_conf_int": reg_risk._conf_int,
                    "_vcov": reg_risk._vcov,
                    "_fml": reg_risk._fml,
                    "_sumFE": reg_risk._sumFE,
                    "_fixef": reg_risk._fixef,
                    "_fixef_dict": reg_risk._fixef_dict,
                    "_data": reg_risk._data}
        with open(f'/data/housing/data/intermediate/HedonicRegressionABM/reg_risk_{sp_fix_eff}.pkl', 'wb') as handle:
            pickle.dump(model_state, handle)

    return reg_risk

def Predict_function(isp,train_path,test_data):
    """
    Function to estimate the hedonic price of a house. 

    Input variables:
        -isp: data from ISP data (used to initialize the regression model)
        - train_path: path to coefficients af the trained model
        - test_data: data for which you want to estimate the home hedonic price

    Output:
        - test_data: same Dataframe as the input with the estimated hedonic price corrected by inflation
    and flood_risk effects (``price_corrected``). 
    """

    # Initialise dummy regresion
    reg_risk = HedonicRegression_PriceTrain(isp.sample(1000),'PRO_COM',save=False)

    # Update reg_risk with good params
    with open(train_path, 'rb') as handle:
        model_state = pickle.load(handle)

    reg_risk._beta_hat = model_state["_beta_hat"]
    reg_risk._coefnames = model_state["_coefnames"]
    reg_risk._se = model_state["_se"]
    reg_risk._tstat = model_state["_tstat"]
    reg_risk._pvalue = model_state["_pvalue"]
    reg_risk._conf_int = model_state["_conf_int"]
    reg_risk._vcov = model_state["_vcov"]
    reg_risk._fml = model_state["_fml"]
    reg_risk._sumFE = model_state["_sumFE"]
    reg_risk._fixef = model_state["_fixef"]
    reg_risk._fixef_dict = model_state["_fixef_dict"]
    reg_risk._data = model_state["_data"]

    test_data['log_price_estimation'] = reg_risk.predict(test_data)

    
    # Correct price estimations with Risk coefficient and average Inflation
    _, df_coeff = utils.ImportantCoefficients(reg_risk,['Risk'])

    # Risk correction
    test_data['price_corrected'] = test_data['log_price_estimation'] - (df_coeff.loc['scenario_Risk','Coefficient']*test_data['scenario_Risk'])

    # Correct inflation
    df_year,_ = utils.ImportantCoefficients(reg_risk,['year']) # av_inflation
    avg_inflation = df_year.Coefficient.mean()
    
    for i in df_year.index:
        test_data['price_corrected'] = test_data['price_corrected'] - (df_year.loc[i,'Coefficient']*test_data[i])
    test_data['price_corrected'] = test_data['price_corrected'] + avg_inflation
    
    # Remove log and drop unnecessary columns
    test_data['price_corrected'] = np.exp(test_data['price_corrected'])
    test_data.drop(columns='log_price_estimation',inplace=True)
    
        
    return test_data
    


if __name__ == "__main__":

    # 40 seconds
    abm_path = "ISP_data_for_ABM/ISP_ABM_up_to_2024_08_0001.csv"
    data = pd.read_csv(jl_vae.path_intermediate + abm_path, index_col = 0)
    geo_dict = jl_vae.load_geo_data()

    # these need to be added to the generated data for predicting with hedonic regression
    year_erogaz = [f'year_erogaz_{u}' for u in [u for u in range(2016, 2025)]] 
    # Trained Hedonic Regression
    train_path = '/data/housing/data/intermediate/HedonicRegressionABM/reg_risk_OMI_id.pkl'
    # Load ISP data for the ABM and clean it 
    data_path = '/data/housing/data/intermediate/ISP_data_for_ABM/ISP_ABM_up_to_2024_08.csv'
    isp = pd.read_csv(data_path,dtype=str,encoding='latin1',na_values=config.default_missing)

    drop_col = ['floor_numeric','NUM_MQ_SUPERF_COMM','COD_CAT_CATASTALE','year_erogaz']
    isp = isp.drop(columns=drop_col)
    isp = utils.CorrectTypes_ABM(isp)



    date_nf = "241203"
    date = "240107"
    vae_data = "full"

    for prov in [u.split(".")[-2][-2:] for u in sorted(glob(jl_vae.path_pop_synth + f"vae_models/*_2401*.pkl"))]:
        print(prov)
        # load VAE

        vae_loaded = jl_vae.load_vae_province(prov, 
                                            jl_vae.path_pop_synth + f"vae_models/settings_{vae_data}_{date}{prov}.pkl", 
                                            jl_vae.path_pop_synth + f"vae_models/vae_{vae_data}_{date}{prov}.pkl")

        # load NFs
        nf_dict = load_dict(jl_vae.path_pop_synth + f"nf_models/nf_{date_nf}{prov}.pkl")
        len(vae_loaded.full_df)
        # real dataframe
        # print("real")
        # df_real = utils.spatial_matching_ABM(vae_loaded.full_df, 
        #                                     hydro_risk = geo_dict["hydro_risk"], 
        #                                     census = geo_dict["census"], 
        #                                     omi_og = geo_dict["omi_og"], 
        #                                     cap = geo_dict["cap"]).dropna().rename(columns = {"GEO_LONGITUDINE_BENE_ROUNDED":"x", "GEO_LATITUDINE_BENE_ROUNDED":"y"})
        # # I set year erogazione as the last year we have 
        # df_real[year_erogaz] = False
        # df_real[year_erogaz[-1]] = True
        # # assign the hedonic price to all samples
        # df_real_with_price = Predict_function(isp, train_path, df_real).rename(columns = {"GEO_LONGITUDINE_BENE_ROUNDED":"x", "GEO_LATITUDINE_BENE_ROUNDED":"y"})
        # # 53 seconds
        # # generate a sample

        # df_real_with_price.to_csv(jl_vae.path_pop_synth + f"pop_samples/pop_real_with_hedonic_price/pop_real_full_250110{prov}.csv")

        print("sample")
        for h in range(10):
            print(h)
            df_sample = vae_loaded.get_sample_from_vae(nf_dict, n_samples = 100000, seed = 800 + h)
            df_sample = utils.spatial_matching_ABM(df_sample,
                                                hydro_risk = geo_dict["hydro_risk"],
                                                census = geo_dict["census"],
                                                omi_og = geo_dict["omi_og"],
                                                cap = geo_dict["cap"]).rename({"GEO_LONGITUDINE_BENE_ROUNDED":"x", "GEO_LATITUDINE_BENE_ROUNDED":"y"})

            # I set year erogazione as the last year we have 
            df_sample[year_erogaz] = False
            df_sample[year_erogaz[-1]] = True
            # df_sample = df_sample.sample(n = len(df_real), replace = False)

            # assign the hedonic price to all samples
            df_sample_with_price = Predict_function(isp, train_path, df_sample).rename(columns = {"GEO_LONGITUDINE_BENE_ROUNDED":"x", "GEO_LATITUDINE_BENE_ROUNDED":"y"})
            df_sample_with_price.to_csv(jl_vae.path_pop_synth + f"pop_samples/pop_synth_with_hedonic_price_250210/pop_synth_full_250210{prov}_{h}.csv")
        