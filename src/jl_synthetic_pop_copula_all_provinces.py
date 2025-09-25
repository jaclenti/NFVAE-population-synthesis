import sys
sys.path += ["../src"]
import utils
import pandas as pd
import numpy as np
from glob import glob
import jl_vae
import jl_nflows_geo_coordinates_2 as nfg
from jl_nflows_geo_coordinates import load_nf as load_dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from jl_synthetic_pop_all_provinces import add_cat_features
from _51_abm_functions import cod_prov_abbrv_df
import torch
import pickle

sys.path += ['../src/'] 
import utils

from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata

features_synpop = ['flag_garage', 'flag_pertinenza', 'flag_air_conditioning',
            'flag_multi_floor', 'log_mq', 'ANNO_COSTRUZIONE_1500_1965',
            'ANNO_COSTRUZIONE_1965_1985', 'ANNO_COSTRUZIONE_1985_2005',
            'ANNO_COSTRUZIONE_2005_2025', 'ANNO_COSTRUZIONE_Missing',
            'High_energy_class', 'Low_energy_class', 'Medium_energy_class',
            'Missing_energy_class', 'COD_CAT_A02', 'COD_CAT_A03',
            'COD_CAT_A_01_07_08', 'COD_CAT_A_04_05', 'floor_0.0', 'floor_1.0',
            'floor_2.0', 'floor_3.0', 'floor_Missing', 'floor_plus_4',
            'flag_air_conditioning_Missing', 'flag_multi_floor_Missing', 'y', 'x','log_price']

def get_copula_data(df, N):
    metadata = Metadata.detect_from_dataframe(df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    synthesizer.reset_sampling()
    df_copula = synthesizer.sample(num_rows = 10 * N)
    return df_copula

def latent_to_real_coordinates(df_real, df_sample, path_nf):
    nf_dict = load_dict(path_nf)

    transf_xy = nfg.nf_latent_to_real(nf_dict, np.array(df_sample[["y_latent", "x_latent"]].astype(np.float32)))
    df_sample["y_trans"] = transf_xy[:,0]
    df_sample["x_trans"] = transf_xy[:,1]
    
    scaler = MinMaxScaler((-1,1))
    scaler.fit(np.array(df_real[["x", "y"]]))
    df_sample[["x", "y"]] = scaler.inverse_transform(np.array(df_sample[["x_trans", "y_trans"]]))
    df_sample.drop(columns = ["x_trans", "y_trans", "x_latent", "y_latent"], inplace = True)
    return df_sample



if __name__ == "__main__":
    prov_len = []
    for _, (prov, cod_prov) in cod_prov_abbrv_df.sort_values("prov_abbrv").iterrows():
        if prov != "SU":
            df_real = pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/real_populations/df_real_{prov}.csv")
            prov_len.append([prov, len(df_real)])
    prov_from_small = list(pd.DataFrame(prov_len, columns = ["prov", "count"]).sort_values("count")["prov"])
    df_prov = cod_prov_abbrv_df.set_index("prov_abbrv").loc[prov_from_small].reset_index()

    
    
    # for _, (prov, cod_prov) in cod_prov_abbrv_df.sort_values("prov_abbrv").iterrows():
    for _, (prov, cod_prov) in df_prov.iterrows():
        print(prov)
        try:
            df_real_latent = pd.read_csv(jl_vae.path_pop_synth + f"pop_samples/pop_real_with_hedonic_price/pop_real_full_250110{prov}.csv", index_col = 0).dropna()[["x_latent", "y_latent"]]
            df_real = pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/real_populations/df_real_{prov}.csv")
            df_real = pd.concat([df_real, df_real_latent], axis = 1)
            df_real95 = df_real.sample(frac = 0.95, random_state = 1111).dropna()

            df_real_ablation = df_real[features_synpop]
            df_copula_ablation  = get_copula_data(df_real_ablation, N = 10 * len(df_real))
            df_copula_ablation.to_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_ablation_{prov}.csv")

            path_nf = jl_vae.path_pop_synth + f"nf_models/nf_241203{prov}.pkl"
            df_real_nf = df_real[[u for u in features_synpop if u not in ["x", "y"]] + ["x_latent", "y_latent"]].dropna()
            df_copula_nf  = get_copula_data(df_real_nf, N = 10 * len(df_real))
            df_copula_nf = latent_to_real_coordinates(df_real, df_copula_nf, path_nf)
            df_copula_nf.to_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_nf_{prov}.csv")
            
            df_real_ablation95 = df_real95[features_synpop]
            df_copula_ablation95  = get_copula_data(df_real_ablation95, N = 10 * len(df_real95))
            df_copula_ablation95.to_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_ablation95_{prov}.csv")

            try:
                path_nf95 = jl_vae.path_pop_synth + f"95sample/nf_models/nf_250703{prov}.pkl"            
                nf_dict95 = load_dict(path_nf95)
                scaler = MinMaxScaler((-1,1))
                scaler.fit(df_real95[["x", "y"]])
                df_real95[["x_norm", "y_norm"]] = scaler.transform(df_real95[["x", "y"]])
                inv_coord = nf_dict95["flow"].flow.forward(torch.tensor(np.array(df_real95[["x_norm", "y_norm"]]), dtype = torch.float32))
                df_real95[["y_latent", "x_latent"]] = torch.sigmoid(inv_coord[0][-1]).detach().numpy()
                df_real_nf95 = df_real95[[u for u in features_synpop if u not in ["x", "y", "x_norm", "y_norm"]] + ["x_latent", "y_latent"]]
                df_copula_nf95  = get_copula_data(df_real_nf95, N = 10 * len(df_real95))

                df_copula_nf95 = latent_to_real_coordinates(df_real, df_copula_nf95, path_nf95)
                df_copula_nf95.to_csv(jl_vae.path_pop_synth + f"copula_samples/df_copula_nf95_{prov}.csv")
            except Exception as e:
                print(e)

                
        except Exception as e:
            print(e)