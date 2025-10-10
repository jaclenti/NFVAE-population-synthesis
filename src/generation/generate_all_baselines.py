# script to generate the synthetic populations with all the benchmarks:
# - NF+VAE
# - VAE
# - copula
# - NF+copula
# - global shuffle
# - local shuffle

import sys
sys.path += [".."]
sys.path += ["../src/evaluation"]
import geopandas as gpd
import pandas as pd
import numpy as np
import json
from time import time
from dataframe_from_insideairbnb import read_gz, read_json, get_df_city_for_nf
import nf_utils as nfg
import torch
import vae
import config
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


# NF + VAE

def train_nf_vae(df,  # contain ["x_norm, y_norm"]
                 num_layers_nf = 32,
                 hidden_features_nf = 32,
                 num_epochs_nf = 20000,
                 lr_nf = 1 / 100000,
                 opt_nf = "Adam" ,
                 flow_name = "NSF_CL",
                 batch_dim_nf = 100,
                 latent_dims_vae = 15,
                 epochs_vae = 100,
                 middle_hidden_dims_vae = [64, 32],
                 lr_vae = 0.001,
                 batch_size_vae = 100,
                 weight_reconstruction_vae = 10,
                 weight_kl_vae = 0.1,
                 weights_geo_vae = 30,
                 opt_vae = "Adam",
                 date = "250703",
                 path_nf = "",
                 path_vae = "",
                 path_settings = "",
                 path_samples = "",
                 n_samples = 100000,
                 train_nf = False):
    
        model_settings = {"num_layers_nf": num_layers_nf, "hidden_features_nf": hidden_features_nf, 
                        "num_epochs_nf": num_epochs_nf, "lr_nf": lr_nf, "opt_nf": opt_nf, "flow_name": flow_name,
                        "batch_dim": batch_dim_nf}

        
        if train_nf:
            # train normalizing flows 
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
        else:
             # load trained normalizing flows
             nf_dict = nfg.load_nf(path_nf)

        # transform normalized geographic coordinates into latent
        # with trained normalizing flows
        inv_coord = nf_dict["flow"].flow.forward(torch.tensor(np.array(df[["x_norm", "y_norm"]]), dtype = torch.float32))
        df[["y_latent", "x_latent"]] = torch.sigmoid(inv_coord[0][-1]).detach().numpy()
                                                
        # list with the number of neurons per layer in the encoder 
        # the decoder will be symmetric
        hidden_dims_vae = [df.drop(columns = ["x", "y", "x_norm", "y_norm"]).shape[1]] + middle_hidden_dims_vae 

        # define and train the vae
        vae = vae.VariationalAutoencoder(full_df = df.astype(np.float32),
                                            latent_dims = latent_dims_vae,
                                            hidden_dims = hidden_dims_vae,
                                                            )                        
        vae.train(epochs = epochs_vae,
                lr = lr_vae,
                hide_tqdm = True,
                verbose = False, 
                opt_name = opt_vae,
                batch_size = batch_size_vae,
                weight_reconstruction = weight_reconstruction_vae,
                weight_kl = weight_kl_vae,
                weights_geo = weights_geo_vae,
                kl_annealing = False
            )

                        
        vae.save(path_vae)
        vae.save_settings(path_settings)
                        
        df_sample = vae.get_sample_from_vae(nf_dict = nf_dict, n_samples = n_samples)
                        
        #df_sample.to_csv(path_samples)
        return df_sample


# VAE

def train_only_vae(df,  # contain ["x_norm, y_norm"]
                 latent_dims_vae = 15,
                 epochs_vae = 100,
                 middle_hidden_dims_vae = [64, 32],
                 lr_vae = 0.001,
                 batch_size_vae = 100,
                 weight_reconstruction_vae = 10,
                 weight_kl_vae = 0.1,
                 weights_geo_vae = 30,
                 opt_vae = "Adam",
                 date = "250703",
                 path_vae = "",
                 path_settings = "",
                 path_samples = "",
                 n_samples = 100000,
                 ):
        # only change the domain of geographic coordinates from [-1,1] to [0,1]
        df[["y_latent", "x_latent"]] = df[["x_norm", "y_norm"]] / 2 + 0.5
    
        # list with the number of neurons per layer in the encoder 
        # the decoder will be symmetric
        hidden_dims_vae = [df.drop(columns = ["x", "y", "x_norm", "y_norm"]).shape[1]] + middle_hidden_dims_vae 

        # define and train the vae
        vae = vae.VariationalAutoencoder(full_df = df.astype(np.float32),
                                            latent_dims = latent_dims_vae,
                                            hidden_dims = hidden_dims_vae,
                                                            )
                        
        vae.train(epochs = epochs_vae,
                lr = lr_vae,
                hide_tqdm = True,
                verbose = False, 
                opt_name = opt_vae,
                batch_size = batch_size_vae,
                weight_reconstruction = weight_reconstruction_vae,
                weight_kl = weight_kl_vae,
                weights_geo = weights_geo_vae,
                kl_annealing = False
            )

                        
        vae.save(path_vae)
        vae.save_settings(path_settings)
                        
        df_sample = vae.get_sample_from_vae(nf_dict = None, n_samples = n_samples)

                        
        #df_sample.to_csv(path_samples)
        return df_sample


# copula

def train_only_copula(df, # with categorical features, not one-hot
                    num_samples, 
                    path_samples = "",
                    numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"]
                    ):
    df_mean, df_std = df[numeric_variables].mean(), df[numeric_variables].std()
    df_ = df.copy()
    df_[numeric_variables] = (df_[numeric_variables] - df_mean) / df_std

    metadata = Metadata.detect_from_dataframe(df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_)
    synthesizer.reset_sampling()
    df_copula = synthesizer.sample(num_rows = num_samples)
    df_copula[numeric_variables] = df_copula[numeric_variables] * df_std + df_mean
    # df_copula.to_csv(path_samples)
    return df_copula

# NF + copula

def train_nf_copula(df, # with categorical features, not one-hot
                    num_samples, 
                    path_samples = "",
                    path_nf = "",
                    numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"]
                    ):
    df_mean, df_std = df[numeric_variables].mean(), df[numeric_variables].std()
    df_ = df.copy()
    df_[numeric_variables] = (df_[numeric_variables] - df_mean) / df_std

    nf_dict = nfg.load_nf(path_nf)
    inv_coord = nf_dict["flow"].flow.forward(torch.tensor(np.array(df_[["x_norm", "y_norm"]]), dtype = torch.float32))
    df_[["x_latent", "y_latent"]] = torch.sigmoid(inv_coord[0][-1]).detach().numpy()
    

    df_.dropna(inplace = True)
    df_real = df_.copy()
    df_.drop(columns = ["x", "y", "x_norm", "y_norm"], inplace = True)

    metadata = Metadata.detect_from_dataframe(df_)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_)
    synthesizer.reset_sampling()
    df_copula = synthesizer.sample(num_rows = num_samples)
    df_copula = latent_to_real_coordinates_copula(df_real, df_copula, path_nf)
    df_copula[numeric_variables] = df_copula[numeric_variables] * df_std + df_mean
    # df_copula.to_csv(path_samples)
    return df_copula


# forward transformation normalizing flows

def latent_to_real_coordinates_copula(df_real, df_sample, path_nf):
    nf_dict = load_dict(path_nf)

    transf_xy = nfg.nf_latent_to_real(nf_dict, np.array(df_sample[["y_latent", "x_latent"]].astype(np.float32)))
    df_sample["x_trans"] = transf_xy[:,0]
    df_sample["y_trans"] = transf_xy[:,1]
    
    scaler = MinMaxScaler((-1,1))
    scaler.fit(np.array(df_real[["x", "y"]]))
    df_sample[["x", "y"]] = scaler.inverse_transform(np.array(df_sample[["x_trans", "y_trans"]]))
    df_sample.drop(columns = ["x_trans", "y_trans", "x_latent", "y_latent"], inplace = True)
    return df_sample


def from_cat_to_dummies_airbnb(df, features):
    df = pd.concat([
        pd.concat([pd.get_dummies(df["roomtype"], dtype = float), 
                   pd.DataFrame(columns = ['roomtype_Entire home/apt', 'roomtype_Hotel room', 'roomtype_Private room', 'roomtype_Shared room'])]),
        df.drop(columns = ["roomtype"])
        ], axis = 1)[features]
    
    return df

def get_area_from_xy(df, gdf_area, var_area = "neighbourhood", epsg = 4326):
    df_geo = gpd.points_from_xy(df['x'], df['y'], z=None, crs = "EPSG:4326")
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


# global shuffle and local shuffle
# draw samples with replacement and assign geographic coordinates random in the region (global shuffle) or in the subregion (local shuffle)
def train_shuffle(df_real,
                  global_shuffle, 
                  gdf_area, 
                  var_bins = ["log_price", "log_mq"], 
                  var_area = "neighbourhood",
                  ratio_syn_real = 10,
                  epsg = 3035,
                  seed = 2):
    subregion_real = get_area_from_xy(df_real, gdf_area, var_area = var_area, epsg = epsg)
    df_real[var_area] = subregion_real
    df_sample = df_real.dropna(subset = [var_area]).sample(n = ratio_syn_real * len(df_real), replace = True).sort_values(var_area).reset_index(drop = True)
    
    np.random.seed(seed)
    # random uniform coordinates within region borders
    x_min, y_min, x_max, y_max = df_real["x"].min() - 1, df_real["y"].min() - 1, df_real["x"].max() + 1, df_real["y"].max() + 1
    ran_xy = np.random.uniform(low = x_min, high = x_max, size = 1000000), np.random.uniform(low = y_min, high = y_max, size = 1000000)
    df_locations = pd.DataFrame({"x": ran_xy[0], "y": ran_xy[1]})
    # match xy coordinates with subregion
    df_locations[var_area] = get_area_from_xy(df_locations, gdf_area, var_area = var_area, epsg = epsg)
    # remove xy coordinates that are not matched with any subregion
    df_locations = df_locations.dropna(subset = [var_area])
            
    ran_coords = []

    
    if global_shuffle:
        ran_coords.append(df_locations.sample(n = len(df_sample), replace = True))
    else:
        for area in df_sample[var_area].unique():
            ran_coords.append(df_locations.loc[df_locations[var_area] == area].sample(n = len(df_sample.loc[df_sample[var_area] == area]), replace = True))
    
    df_shuffle = (pd.concat([df_sample.drop(columns = ["x", "y", var_area]),
                            pd.concat(ran_coords).reset_index(drop = True)], axis = 1)
                            .drop(columns  = [var_area]))
    
    return df_shuffle





def select_n_in_border(df_sample, n, gdf, seed = 5):
    df_sample_points = gpd.GeoDataFrame(df_sample, geometry = gpd.points_from_xy(df_sample["x"], df_sample["y"]))
    df_sample = (gpd.tools.sjoin(df_sample_points, gdf, predicate="within", how='left')
                 .query("(review_scores_rating >= 0)&(review_scores_rating <= 5)&(reviews_per_month >= 0)")
                 .dropna(subset = "neighbourhood")[df_sample.columns]
                 )
    try:
        df_sample = df_sample.sample(n, random_state = seed, replace = False)
    except:
        df_sample = df_sample.sample(n, random_state = seed, replace = True)
    return df_sample



def baselines_dict(city, date = "250811"):
    df_real = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_df_real_{date}.csv")
    df_real95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_df_real95_{date}.csv")
            
    df_vae = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_vae.csv")
    df_vae95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_vae_95.csv")
    df_nfvae = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nfvae.csv")
    df_nfvae95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nfvae_95.csv")
            
    df_shuffle_global = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_global.csv")
    df_shuffle_global95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_global95.csv")
    df_shuffle_local = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_local.csv")
    df_shuffle_local95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_local95.csv")

    df_copula = pd.read_csv(config.airbnb_path +f"pop_samples/{city}_{date}_copula.csv")
    df_copula95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_copula95.csv")
    df_nf_copula = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nf_copula.csv")
    df_nf_copula95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nf_copula95.csv")

    baselines_pop = {
    "df_real": df_real,
    "df_real95": df_real95,
    "df_copula": select_n_in_border(df_copula, n = len(df_real), gdf = gdf_full),
    "df_copula95": select_n_in_border(df_nf_copula, n = len(df_real), gdf = gdf_full),
    "df_nf_copula": select_n_in_border(df_copula95, n = len(df_real), gdf = gdf_full),
    "df_nf_copula95": select_n_in_border(df_nf_copula95, n = len(df_real), gdf = gdf_full),
    "df_shuffle_global":df_shuffle_global,
    "df_shuffle_global95":df_shuffle_global95,
    "df_shuffle_local":df_shuffle_local,
    "df_shuffle_local95":df_shuffle_local95,
    "df_vae": select_n_in_border(df_vae, n = len(df_real), gdf = gdf_full),
    "df_vae95": select_n_in_border(df_vae95, n = len(df_real), gdf = gdf_full),
    "df_nfvae": select_n_in_border(df_nfvae, n = len(df_real), gdf = gdf_full),
    "df_nfvae95": select_n_in_border(df_nfvae95, n = len(df_real), gdf = gdf_full),
    }


    for u in baselines_pop:
        if "copula" in u:
            baselines_pop[u] = from_cat_to_dummies_airbnb(baselines_pop[u], features = config.cols).fillna(False)
        baselines_pop[u] = baselines_pop[u][config.cols] + 0.


    for k in ["df_nfvae", "df_vae", "df_nfvae95", "df_vae95"]:
        for var in [u for u in amenities_vars]:
            baselines_pop[k][var] = (baselines_pop[k][var] > baselines_pop[k][var].quantile(1 - baselines_pop[k.replace("nf", "").replace("vae", "real")][var].astype(np.float32).mean())) + 0.

        baselines_pop[k][["accommodates","bathrooms","bedrooms","beds"]] = baselines_pop[k][["accommodates","bathrooms","bedrooms","beds"]].astype(int) + 0.


    with open(config.baselines_path + f'/all_baselines_{city}.pickle', 'wb') as f:
        pickle.dump(baselines_pop, f)


def assign_discrete_variables(df, df_real):
    amenities_vars = ["Air conditioning", "Elevator", "Self check-in", "Pets allowed", "Private living room", "Backyard", "Pool"]

    for var in [u for u in amenities_vars]:
        df[var] = (df[var] > df[var].quantile(1 - df_real[var].astype(np.float32).mean())) + 0.

    df[["accommodates","bathrooms","bedrooms","beds"]] = df[["accommodates","bathrooms","bedrooms","beds"]].astype(int) + 0.

    return df





if __name__ == "__main__":
    date = "250924"

    date_nf = "250703"
    date_nf95 = "250908"
    

    for city in config.cities:
        
        print(">>>", city, "<<<")
        try:
            t0 = time()
            df_full = read_gz(city, fn = "/listings.csv.gz")
            df_full["amenities"] = [json.loads(u) for u in df_full["amenities"]]
            amenities_vars = ["Air conditioning", "Elevator", "Self check-in", "Pets allowed", "Private living room", "Backyard", "Pool"]
            for var in amenities_vars:
                df_full[var] = [var in u for u in df_full["amenities"]]

            df = df_full[['id', 'neighbourhood_cleansed', 'latitude',
                'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
                'bedrooms', 'beds', 'price',
                'review_scores_rating', 'reviews_per_month'] + amenities_vars].rename(columns = {"latitude": "y", "longitude": "x"})


            gdf_full = read_json(city)
            gdf = gdf_full[["neighbourhood","geometry"]].dissolve().to_crs('EPSG:4326')


            df_real = get_df_city_for_nf(city).rename(columns = {"neighbourhood_cleansed": "neighbourhood"})
            df_real95 = df_real.sample(frac = 0.95, random_state = 1234)
            #df_real = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_df_real_{date}.csv", index_col = 0)
            #df_real95 = pd.read_csv(config.airbnb_path + f"pop_samples/{city}_df_real95_{date}.csv", index_col = 0)

            print(city, "shuffle")
            df_shuffle_global = train_shuffle(df_real, global_shuffle = True, gdf_area = gdf_full, var_area = "neighbourhood",
                                        var_bins = ["review_scores_rating", "reviews_per_month", "log_price"],
                                        epsg = 4326)
            df_shuffle_global95 = train_shuffle(df_real95, global_shuffle = True, gdf_area = gdf_full, var_area = "neighbourhood",
                                        var_bins = ["review_scores_rating", "reviews_per_month", "log_price"],
                                        epsg = 4326)
            df_shuffle_local = train_shuffle(df_real, global_shuffle = False, gdf_area = gdf_full, var_area = "neighbourhood",
                                        var_bins = ["review_scores_rating", "reviews_per_month", "log_price"],
                                        epsg = 4326)
            df_shuffle_local95 = train_shuffle(df_real95, global_shuffle = False, gdf_area = gdf_full, var_area = "neighbourhood",
                                        var_bins = ["review_scores_rating", "reviews_per_month", "log_price"],
                                        epsg = 4326)
                
            
            print(city, "vae")
            df_vae = train_only_vae(df = df_real, n_samples = 5 * len(df_real), latent_dims_vae = 10, middle_hidden_dims_vae = [64,32,32])
            df_vae95 = train_only_vae(df = df_real95, n_samples = 5 * len(df_real95), latent_dims_vae = 10, middle_hidden_dims_vae = [64,32,32])
            
            print(city, "nfvae")
            df_nfvae = train_nf_vae(df = df_real, 
                                    path_nf = config.airbnb_path + f"nf_models/{city}_{date_nf}.pkl",
                                    train_nf = False,
                                    n_samples = 5 * len(df_real), 
                                    latent_dims_vae = 10, 
                                    middle_hidden_dims_vae = [64,32,32]
                                    )

            print(city, "nfvae95")
            df_nfvae95 = train_nf_vae(df = df_real95, 
                                      path_nf = config.airbnb_path + f"nf_models/{city}_{date_nf95}.pkl",
                                      train_nf = False,
                                      n_samples = 5 * len(df_real95), 
                                      latent_dims_vae = 10, middle_hidden_dims_vae = [64,32,32])

            df_vae = assign_discrete_variables(df_vae, df_real)
            df_nfvae = assign_discrete_variables(df_nfvae, df_real)
            df_vae95 = assign_discrete_variables(df_vae95, df_real95)
            df_nfvae95 = assign_discrete_variables(df_nfvae95, df_real95)

            print(city, "copula")
            df_real_cat = df_real.copy()

            drop_cols = ['roomtype_Entire home/apt', 'roomtype_Hotel room',
                'roomtype_Private room', 'roomtype_Shared room']
            df_real_cat["roomtype"] = pd.from_dummies(df_real_cat[['roomtype_Entire home/apt', 'roomtype_Hotel room',
                'roomtype_Private room', 'roomtype_Shared room']]).iloc[:,0].to_list()
            df_real_cat.drop(columns = drop_cols, inplace = True)
            df_real_cat95 = df_real95.copy()
            df_real_cat95["roomtype"] = pd.from_dummies(df_real_cat95[['roomtype_Entire home/apt', 'roomtype_Hotel room',
                'roomtype_Private room', 'roomtype_Shared room']]).iloc[:,0].to_list()
            df_real_cat95.drop(columns = drop_cols, inplace = True)

            df_copula = train_only_copula(df = df_real_cat,
                                        num_samples = 5 * len(df_real),
                                        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])
            df_nf_copula = train_nf_copula(df = df_real_cat, num_samples = 5 * len(df_real),
                                        path_nf = config.airbnb_path + f"nf_models/{city}_{date_nf}.pkl",
                                        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])

            df_copula95 = train_only_copula(df = df_real_cat95, 
                                        num_samples = 5 * len(df_real),
                                        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])
            df_nf_copula95 = train_nf_copula(df = df_real_cat95, num_samples = 5 * len(df_real),
                                        path_nf = config.airbnb_path + f"nf_models/{city}_{date_nf95}.pkl",
                                        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])
            
            

            
            
        

            df_real.to_csv(config.airbnb_path + f"pop_samples/{city}_df_real_{date}.csv")
            df_real95.to_csv(config.airbnb_path + f"pop_samples/{city}_df_real95_{date}.csv")
            
            df_vae.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_vae.csv")
            df_vae95.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_vae_95.csv")
            df_nfvae.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nfvae.csv")
            df_nfvae95.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nfvae_95.csv")
            
            df_shuffle_global.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_global.csv")
            df_shuffle_global95.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_global95.csv")
            df_shuffle_local.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_local.csv")
            df_shuffle_local95.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_shuffle_local95.csv")

            df_copula.to_csv(config.airbnb_path +f"pop_samples/{city}_{date}_copula.csv")
            df_copula95.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_copula95.csv")
            df_nf_copula.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nf_copula.csv")
            df_nf_copula95.to_csv(config.airbnb_path + f"pop_samples/{city}_{date}_nf_copula95.csv")
            t1 = time()
            print(round(t1 - t0), "s")
        except Exception as e:
            print(city, e)
