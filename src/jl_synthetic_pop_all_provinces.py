# ********** VAE training for all provinces ************
# in this script we loop over all the italian provinces to train (i) a Normalizing Flows (NFs) able to map geographical coordinates into a latent space
# (ii) a Variational Autoencoder (VAE) able to generate new samples of realistic houses, both for the geographical coordinates and the other
# categorical and numeric features.
# 
# We save (i) the trained NFs, (ii) the trained VAE, (iii) a sample of the synthetic population
# 
# In this implementation, VAE maps the data in a high-dimensional space, allowing for overfitting.
# In this scenario, we want to generate data that are very close to the observed ones.
# 
# For a single province, NFs takes approximately 4 hours to train, and VAE takes 3 minutes.
# 
# we consider two strategies for handling missing data
# (i) generate samples, allowing fot the "Missing" category
# (ii) drop all the missing data
# given the high number of missing, we apply (i)
# 


import pandas as pd
import numpy as np
import torch
import sys
sys.path += ["../src"]
import jl_vae
import jl_nflows_geo_coordinates_2 as nfg
from jl_nflows_geo_coordinates import load_nf as load_dict
import utils
import config
import pickle
from tqdm import tqdm
from time import time

if __name__ == "__main__":
    num_layers_nf = 32
    hidden_features_nf = 32
    num_epochs_nf = 20000
    lr_nf_ = 10
    lr_nf = lr_nf_ / 1000000
    opt_nf = "Adam" 
    flow_name = "NSF_CL"
    batch_dim_nf = 100
    _ = "nf"

    latent_dims_vae = 20
    epochs_vae = 100
    middle_hidden_dims_vae = [64, 32]
    lr_vae = 0.001
    opt_vae = "Adam"

    # geo_data_dict = jl_vae.load_geo_data()
    abm_path = "ISP_data_for_ABM/ISP_ABM_up_to_2024_08_0001.csv"
    data = pd.read_csv(jl_vae.path_intermediate + abm_path, index_col = 0)

    for prov in data["prov_abbrv"].sort_values().unique():
        print(prov)
        try:
            date = "241203" + prov
            # id = "_".join([str(u) for u in [num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf, flow_name, batch_dim_nf, date]])

            path_nf = jl_vae.path_pop_synth + f"nf_models/nf_{date}.pkl"
            
            df_prov_full = jl_vae.get_df_prov(prov, dropna = False)

            df_prov_full = df_prov_full[[u for u in jl_vae.cols if u not in ["prov_abbrv"]] + ["x_norm", "y_norm"]].assign(flag_air_conditioning_Missing = lambda x: x["flag_air_conditioning"] == "Missing",
                                                            flag_multi_floor_Missing = lambda x: x["flag_multi_floor"] == "Missing").replace("Missing",0).astype(float)
            df_prov_dropna = jl_vae.get_df_prov(prov, dropna = True, drop_cols = ["flag_air_conditioning", "flag_multi_floor", 'floor_0.0', 'floor_1.0',
                                                                                  'floor_2.0', 'floor_3.0', 'floor_Missing', 'floor_plus_4'])

            model_settings = {"num_layers_nf": num_layers_nf, "hidden_features_nf": hidden_features_nf, 
                            "num_epochs_nf": num_epochs_nf, "lr_nf": lr_nf, "opt_nf": opt_nf, "flow_name": flow_name,
                            "batch_dim": batch_dim_nf, "date": date, "prov": prov}


            print(prov, "NF")
            t0 = time()
            
            # nf_dict = nfg.train_model(df_prov_full[["x_norm", "y_norm"]], 
            #                         num_layers = num_layers_nf, 
            #                         hidden_features = hidden_features_nf,
            #                         num_epochs = num_epochs_nf, 
            #                         opt_name = opt_nf, 
            #                         batch_dim = batch_dim_nf, 
            #                         flow_name = flow_name,
            #                         hide_progress = True, 
            #                         path = path_nf,
            #                         model_settings = model_settings)
            nf_dict = load_dict(path_nf)

            t1 = time()
            print(prov, "NF", round(t1 - t0, 2))
            

            for j,df_vae in enumerate([df_prov_full.copy(), df_prov_dropna.copy()]):
                if j:
                    continue
                print(prov, j, "VAE")
                date = "240107" + prov
                
                t0 = time()
                inv_coord = nf_dict["flow"].flow.forward(torch.tensor(np.array(df_vae[["x_norm", "y_norm"]]), dtype = torch.float32))
                df_vae[["y_latent", "x_latent"]] = torch.sigmoid(inv_coord[0][-1]).detach().numpy()

                if j:
                    df_vae.drop(columns = [u for u in df_vae.columns if "Missing" in u], inplace = True)

                # df_vae.drop(columns = [u for u in df_vae.columns if "floor" in u], inplace = True)
                # df_vae.drop(columns = [u for u in df_vae.columns if "ANNO" in u], inplace = True)
                # df_vae.drop(columns = [u for u in df_vae.columns if "COD" in u], inplace = True)
                # df_vae.drop(columns = [u for u in df_vae.columns if "latent" in u], inplace = True)
                # df_vae.drop(columns = [u for u in ["flag_geo_valid", "log_mq"] if u in df_vae.columns], inplace = True)
                df_vae.drop(columns = [u for u in ["flag_geo_valid"] if u in df_vae.columns], inplace = True)
                # df_vae.drop(columns = ["x", "y"], inplace = True)
                
                hidden_dims_vae = [df_vae.drop(columns = ["x", "y", "x_norm", "y_norm"]).shape[1]] + middle_hidden_dims_vae 

                vae = jl_vae.VariationalAutoencoder(full_df = df_vae.astype(np.float32),
                                                    latent_dims = latent_dims_vae,
                                                    hidden_dims = hidden_dims_vae,
                                                    )
                
                vae.train(epochs = epochs_vae,
                        lr = lr_vae,
                        hide_tqdm = True,
                        verbose = False, 
                        opt_name = opt_vae,
                        batch_size = 100,
                        weight_reconstruction = 10, #6,
                        weight_kl = 0.1,
                        weights_geo = 30, # 20,
                        kl_annealing = False
                        )
                

                path_vae = jl_vae.path_pop_synth + f"vae_models/vae_{['full', 'dropna'][j]}_{date}.pkl"
                path_settings = jl_vae.path_pop_synth + f"vae_models/settings_{['full', 'dropna'][j]}_{date}.pkl"
                
                vae.save(path_vae)
                vae.save_settings(path_settings)
                
                df_sample = vae.get_sample_from_vae(nf_dict = nf_dict)

                # df_sample = utils.spatial_matching_ABM(df_sample.rename(columns = {"x":"GEO_LONGITUDINE_BENE_ROUNDED", 
                #                                                                 "y":"GEO_LATITUDINE_BENE_ROUNDED"}), 
                #                                                                 geo_data_dict["hydro_risk"], 
                #                                                                 geo_data_dict["census"], 
                #                                                                 geo_data_dict["omi_og"], 
                #                                                                 geo_data_dict["cap"])
                
                df_sample.to_csv(jl_vae.path_pop_synth + f"pop_samples/synthetic_pop_{['full', 'dropna'][j]}_{date}.csv")

                t1 = time()
                print(prov, "VAE", j, round(t1 - t0, 2))
        except:
            print("Error", prov)
