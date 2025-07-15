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
import geopandas as gpd
import gzip
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler

amenities_vars = ["Air conditioning", "Elevator", "Self check-in", "Pets allowed", "Private living room", "Backyard", "Pool"]
int_vars = ["accomodates", "bathrooms", "bedrooms", "beds"]

def read_gz(city, fn):
    return pd.read_csv(gzip.open(config.INTERMEDIATE_DATA_PATH + "airbnb/" + city + "/" + fn, mode='rb'))

def read_json(city):
    path_city = config.INTERMEDIATE_DATA_PATH + "airbnb/" + city
    gdf = gpd.read_file(path_city + "/neighbourhoods.geojson")
    return gdf

def get_df_city_for_nf(city):
    df_full = read_gz(city, fn = "/listings.csv.gz")
    
    df_full["amenities"] = [json.loads(u) for u in df_full["amenities"]]
    for var in amenities_vars:
        df_full[var] = [var in u for u in df_full["amenities"]]
    df = df_full[['id', #'neighbourhood_cleansed', 
                'latitude', 'longitude', #'property_type',
                'room_type', 'accommodates', 'bathrooms',
                'bedrooms', 'beds', 'price',
                'review_scores_rating', 'reviews_per_month'] + amenities_vars].rename(columns = {"latitude": "y", "longitude": "x"}).dropna()
    df["bathrooms"] = df["bathrooms"].astype(int)
    df["price"] = [float(u[1:].split(".")[0].replace(",","")) for u in df["price"]]
    df["log_price"] = np.log(df["price"])

    df.rename(columns = {u: "flag_" + u for u in amenities_vars})
    df.rename(columns = {u: "int_" + u for u in int_vars})    
    df = pd.concat([df.drop(columns = ["room_type"]), 
                    pd.get_dummies(df["room_type"], prefix = "roomtype")], axis = 1).set_index("id")
    df = df.drop(columns = ["price"])

    scaler = MinMaxScaler((-1,1))
    scaler.fit(np.array(df[["x", "y"]]))
    df[["x_norm", "y_norm"]] = scaler.transform(np.array(df[["x", "y"]]))
    return df



if __name__ == "__main__":
    airbnb_path = "/data/housing/data/intermediate/jl_pop_synth/airbnb/"

    num_layers_nf = 32
    hidden_features_nf = 32
    num_epochs_nf = 20000
    lr_nf_ = 10
    lr_nf = lr_nf_ / 1000000
    opt_nf = "Adam" 
    flow_name = "NSF_CL"
    batch_dim_nf = 100
    _ = "nf"

    latent_dims_vae = 15
    epochs_vae = 100
    middle_hidden_dims_vae = [64, 32]
    lr_vae = 0.001
    opt_vae = "Adam"

    date = "250703"

    for city in ["copenhagen", "brisbane", "hawaii", "naples", "paris", "barcelona"]:
        print(city, "NF")
        t0 = time()
        
        df = get_df_city_for_nf(city)
        model_settings = {"num_layers_nf": num_layers_nf, "hidden_features_nf": hidden_features_nf, 
                        "num_epochs_nf": num_epochs_nf, "lr_nf": lr_nf, "opt_nf": opt_nf, "flow_name": flow_name,
                        "batch_dim": batch_dim_nf}

        path_nf = airbnb_path + f"nf_models/{city}_{date}.pkl"
        
        nf_dict = nfg.train_model(df[["x_norm", "y_norm"]], 
                                num_layers = num_layers_nf, 
                                hidden_features = hidden_features_nf,
                                num_epochs = num_epochs_nf, 
                                opt_name = opt_nf, 
                                batch_dim = batch_dim_nf, 
                                flow_name = flow_name,
                                hide_progress = True, 
                                path = path_nf,
                                model_settings = model_settings)

        nf_dict = load_dict(path_nf)
        t1 = time()
        print(city, "NF", round(t1 - t0, 2))
            
        
        print(city, "VAE")
                        
        t0 = time()
        inv_coord = nf_dict["flow"].flow.forward(torch.tensor(np.array(df[["x_norm", "y_norm"]]), dtype = torch.float32))
        df[["y_latent", "x_latent"]] = torch.sigmoid(inv_coord[0][-1]).detach().numpy()
                                                
        hidden_dims_vae = [df.drop(columns = ["x", "y", "x_norm", "y_norm"]).shape[1]] + middle_hidden_dims_vae 

        vae = jl_vae.VariationalAutoencoder(full_df = df.astype(np.float32),
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
                        

        path_vae = airbnb_path + f"vae_models/vae_{city}_{date}.pkl"
        path_settings = airbnb_path + f"vae_models/settings_{city}_{date}.pkl"
                        
        vae.save(path_vae)
        vae.save_settings(path_settings)
                        
        df_sample = vae.get_sample_from_vae(nf_dict = nf_dict, n_samples = 100000)

                        
        df_sample.to_csv(airbnb_path + f"pop_samples/synthetic_pop_{city}_{date}.csv")

        t1 = time()
        print(city, "VAE", round(t1 - t0, 2))
        
