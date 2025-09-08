import sys
sys.path += ["../src"]
import geopandas as gpd
import pandas as pd
import numpy as np
import json
from time import time
from jl_synthetic_pop_all_vae_provinces_airbnb import read_gz, read_json, get_df_city_for_nf
import jl_nflows_geo_coordinates_2 as nfg
from jl_nflows_geo_coordinates import load_nf as load_dict
import torch
import jl_vae
from jl_synthetic_ipf_all_provinces import create_pop # transform_bins, transform_bins_to_real, get_area_from_xy
from jl_synthetic_pop_copula_all_provinces import latent_to_real_coordinates
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata

airbnb_path = "/data/housing/data/intermediate/jl_pop_synth/airbnb/"

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
             nf_dict = load_dict(path_nf)
        
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
                batch_size = batch_size_vae,
                weight_reconstruction = weight_reconstruction_vae,
                weight_kl = weight_kl_vae,
                weights_geo = weights_geo_vae,
                kl_annealing = False
            )

                        
        #vae.save(path_vae)
        #vae.save_settings(path_settings)
                        
        df_sample = vae.get_sample_from_vae(nf_dict = nf_dict, n_samples = n_samples)

                        
        #df_sample.to_csv(path_samples)
        return df_sample

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
    
        df[["y_latent", "x_latent"]] = df[["x_norm", "y_norm"]] / 2 + 0.5
                                                
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
                batch_size = batch_size_vae,
                weight_reconstruction = weight_reconstruction_vae,
                weight_kl = weight_kl_vae,
                weights_geo = weights_geo_vae,
                kl_annealing = False
            )

                        
        #vae.save(path_vae)
        #vae.save_settings(path_settings)
                        
        df_sample = vae.get_sample_from_vae(nf_dict = None, n_samples = n_samples)

                        
        #df_sample.to_csv(path_samples)
        return df_sample

def train_ipf(df, # must contain column var_area
              gdf_area,
              with_bins, 
              province_shuffle,
              var_area = "CAP",
              numeric_variables = ["log_price", "log_mq"],
              num_bins = 5,
              seed = 2,
              ratio_syn_real = 10,
              path_pop_sample = "",
              ):
    df.dropna(inplace = True)
    ran_x = np.random.uniform(low = df["x"].min(), high = df["x"].max(), size = 1000000)
    ran_y = np.random.uniform(low = df["y"].min(), high = df["y"].max(), size = 1000000)
    ids = np.arange(1000000)

    df_locations = gpd.points_from_xy(ran_x, ran_y, z = None, crs = "EPSG:3035")
    df_locations = gpd.GeoDataFrame(pd.Series(ids, name = "id"), geometry= df_locations)
    df_locations = gpd.tools.sjoin(df_locations, gdf_area, predicate = "within", how = 'left')
    df_locations.drop(columns = 'index_right', inplace = True)
    df_locations["x"] = df_locations.geometry.x
    df_locations["y"] = df_locations.geometry.y

    df = df.dropna(subset = var_area)
    df_sample_list = []
            
    for area in df[var_area].unique():
        locations_area = df_locations.loc[df_locations[var_area] == area]
        df_area = df.drop(columns = ["x", "y"]).loc[df[var_area] == area].dropna()
        N_obs_area = len(df_area)
        if N_obs_area > 1:
            df_area = transform_bins(df_area, numeric_variables, q = num_bins)
            df_counts = df_area.value_counts().reset_index()
            sample_df_area = df_counts.sample(n = ratio_syn_real * N_obs_area, 
                                             weights = df_counts["count"],
                                             random_state = seed, 
                                             replace = True).drop(columns = ["count"])
            for var in numeric_variables:
                sample_df_area = transform_bins_to_real(sample_df_area, df, var, q = num_bins, seed = seed)
            
        else:
            sample_df_area = pd.concat([df_area] * ratio_syn_real)
                    
        if len(locations_area) > 0:
            sample_locations = locations_area.sample(n = ratio_syn_real * N_obs_area, replace = True, random_state = seed)
            sample_df_area["x"] = list(sample_locations["x"])
            sample_df_area["y"] = list(sample_locations["y"])
            sample_df_area[var_area] = area
                    
        df_sample_list.append(sample_df_area)
    sample_df = pd.concat(df_sample_list).reset_index().drop(columns = "index")

    # sample_df.to_csv(path_pop_sample)
    return sample_df

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

def train_nf_copula(df, # with categorical features, not one-hot
                    num_samples, 
                    path_samples = "",
                    path_nf = "",
                    numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"]
                    ):
    df_mean, df_std = df[numeric_variables].mean(), df[numeric_variables].std()
    df_ = df.copy()
    df_[numeric_variables] = (df_[numeric_variables] - df_mean) / df_std

    nf_dict = load_dict(path_nf)
    inv_coord = nf_dict["flow"].flow.forward(torch.tensor(np.array(df_[["x_norm", "y_norm"]]), dtype = torch.float32))
    df_[["y_latent", "x_latent"]] = torch.sigmoid(inv_coord[0][-1]).detach().numpy()
    

    df_.dropna(inplace = True)
    df_real = df_.copy()
    df_.drop(columns = ["x", "y", "x_norm", "y_norm"], inplace = True)

    metadata = Metadata.detect_from_dataframe(df_)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_)
    synthesizer.reset_sampling()
    df_copula = synthesizer.sample(num_rows = num_samples)
    df_copula = latent_to_real_coordinates(df_real, df_copula, path_nf)
    df_copula[numeric_variables] = df_copula[numeric_variables] * df_std + df_mean
    # df_copula.to_csv(path_samples)
    return df_copula

def from_cat_to_dummies_airbnb(df, features):
    df = pd.concat([
        pd.concat([pd.get_dummies(df["roomtype"], dtype = float), 
                   pd.DataFrame(columns = ['roomtype_Entire home/apt', 'roomtype_Hotel room', 'roomtype_Private room', 'roomtype_Shared room'])]),
        df.drop(columns = ["roomtype"])
        ], axis = 1)[features]
    
    return df


def select_n_in_border(df_sample, n, gdf, seed = 5):
    df_sample_points = gpd.GeoDataFrame(df_sample, geometry = gpd.points_from_xy(df_sample["x"], df_sample["y"]))
    df_sample = (gpd.tools.sjoin(df_sample_points, gdf, predicate="within", how='left')
                 .query("(review_scores_rating >= 0)&(review_scores_rating <= 5)&(reviews_per_month >= 0)")
                 .dropna(subset = "neighbourhood")[df_sample.columns]
                 .sample(n, random_state = seed))
    return df_sample


def baselines_dict(city, date = "250811"):
    df_nfvae = pd.read_csv(airbnb_path + f"pop_samples/synthetic_pop_{city}_{date}.csv", index_col = 0)
    df_nfvae95 = pd.read_csv(airbnb_path + f"pop_samples/synthetic_pop_{city}_{date}_95.csv", index_col = 0)
    #df_real.to_csv(airbnb_path + f"pop_samples/{city}_df_real_{date}.csv")
    df_real = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_real_{date}.csv", index_col = 0)
    df_real95 = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_real95_{date}.csv", index_col = 0)
    dfs_shuffle = {}
    dfs_shuffle95 = {}
    for with_bins in [0,1]:
        for province_shuffle in [0,1]:
            dfs_shuffle[(with_bins, province_shuffle)] = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_shuffle_with_bins{with_bins}_shuffle_province{shuffle_province}.csv", index_col = 0)
            dfs_shuffle95[(with_bins, province_shuffle)] = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_shuffle95_with_bins{with_bins}_shuffle_province{shuffle_province}.csv", index_col = 0)
    df_copula = pd.read_csv(airbnb_path +f"pop_samples/{city}_df_copula_{date}.csv", index_col = 0)
    df_nf_copula = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_nf_copula_{date}.csv", index_col = 0)
    df_copula95 = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_copula95_{date}.csv", index_col = 0)
    df_nf_copula95 = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_nf_copula95_{date}.csv", index_col = 0)
    
    baselines_pop = {
    "df_real": df_real,
    "df_real95": df_real95,
    "df_copula": select_n_in_border(df_copula, n = len(df_real), gdf = gdf_full),
    "df_copula95": select_n_in_border(df_nf_copula, n = len(df_real), gdf = gdf_full),
    "df_nf_copula": select_n_in_border(df_copula95, n = len(df_real), gdf = gdf_full),
    "df_nf_copula95": select_n_in_border(df_nf_copula95, n = len(df_real), gdf = gdf_full),
    #"df_ipf": select_n_in_border(df_ipf.drop(columns = ["neighbourhood"]), n = len(df_real), gdf = gdf_full),
    #"df_ipf95": select_n_in_border(df_ipf95.drop(columns = ["neighbourhood"]), n = len(df_real), gdf = gdf_full),
    "df_shuffle_neighbourhood_num": dfs_shuffle[(0,0)],
    "df_shuffle_neighbourhood_num95": dfs_shuffle95[(0,0)],
    "df_shuffle_city_num": dfs_shuffle[(0,1)],
    "df_shuffle_city_num95": dfs_shuffle95[(0,1)],
    "df_shuffle_neighbourhood_bins": dfs_shuffle[(1,0)],
    "df_shuffle_neighbourhood_bins95": dfs_shuffle95[(1,0)],
    "df_shuffle_city_bins": dfs_shuffle[(1,1)],
    "df_shuffle_city_bins95": dfs_shuffle95[(1,1)],
    "df_nfvae": select_n_in_border(df_nfvae, n = len(df_real), gdf = gdf_full),
    "df_nfvae95": select_n_in_border(df_nfvae95, n = len(df_real), gdf = gdf_full),
    # df_vae,
    # df_vae95
    }






if __name__ == "__main__":
    #citynum = sys.argv[1]
    #date = "250811"
    date = "250905"

    #date_nf = "250703"
    date_nf = "250908"
    #date_nf95 = "250806"
    date_nf95 = "250908_95"

    #for city in ["brisbane", "hawaii", "naples", "paris", "barcelona", "copenhagen"]:
    for city in ["washington dc" ,"rhode island","oslo","oakland","montreal","cape town" ,"lyon","mexico city","santiago","hong-kong","seattle", "austin","singapore"]:
    #city = ["washington dc" ,"rhode island","oslo","oakland","montreal","cape town" ,"lyon","mexico city","santiago","hong-kong","seattle", "austin","singapore"][int(citynum)]
    
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
            #df_real = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_real_{date}.csv", index_col = 0)
            #df_real95 = pd.read_csv(airbnb_path + f"pop_samples/{city}_df_real95_{date}.csv", index_col = 0)
            

            print(city, "ipf")
            #df_ipf = train_ipf(df = df_real, gdf_area = gdf_full, var_area = "neighbourhood",
            #        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])
            #df_ipf95 = train_ipf(df = df_real95, gdf_area = gdf_full, var_area = "neighbourhood",
            #        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])
            dfs_shuffle = {}
            dfs_shuffle95 = {}
            for with_bins in [0,1]:
                for province_shuffle in [0,1]:
                    dfs_shuffle[(with_bins, province_shuffle)] = create_pop(df_real, with_bins = with_bins, province_shuffle = province_shuffle, 
                                        gdf_area = gdf_full, var_area = "neighbourhood",
                                        var_bins = ["review_scores_rating", "reviews_per_month", "log_price"],
                                        epsg = 4326)
                    dfs_shuffle95[(with_bins, province_shuffle)] = create_pop(df_real95, with_bins = with_bins, province_shuffle = province_shuffle, 
                                        gdf_area = gdf_full, var_area = "neighbourhood",
                                        var_bins = ["review_scores_rating", "reviews_per_month", "log_price"],
                                        epsg = 4326)

            print(city, "vae")
            df_vae = train_only_vae(df = df_real, n_samples = 5 * len(df_real))
            df_vae95 = train_only_vae(df = df_real95, n_samples = 5 * len(df_real95))
            print(city, "nfvae")
            df_nfvae = train_nf_vae(df = df_real, 
                                      path_nf = airbnb_path + f"nf_models/{city}_{date_nf}.pkl",
                                      train_nf = True,
                                      n_samples = 5 * len(df_real))

            print(city, "nfvae95")
            df_nfvae95 = train_nf_vae(df = df_real95, 
                                      path_nf = airbnb_path + f"nf_models/{city}_{date_nf95}.pkl",
                                      train_nf = True,
                                      n_samples = 5 * len(df_real95))

            
            print(city, "copula")
            df_real_cat = df_real.copy()

            drop_cols = ['roomtype_Entire home/apt', 'roomtype_Hotel room',
                'roomtype_Private room', 'roomtype_Shared room']#, 'neighbourhood']
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
                                        path_nf = airbnb_path + f"nf_models/{city}_{date_nf}.pkl",
                                        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])

            df_copula95 = train_only_copula(df = df_real_cat95, 
                                        num_samples = 5 * len(df_real),
                                        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])
            df_nf_copula95 = train_nf_copula(df = df_real_cat95, num_samples = 5 * len(df_real),
                                        path_nf = airbnb_path + f"nf_models/{city}_{date_nf}_95.pkl",
                                        numeric_variables = ["review_scores_rating", "reviews_per_month", "log_price"])
            
            

            
            
        
        

            df_real.to_csv(airbnb_path + f"pop_samples/{city}_df_real_{date}.csv")
            df_real95.to_csv(airbnb_path + f"pop_samples/{city}_df_real95_{date}.csv")
            
            df_vae.to_csv(airbnb_path + f"pop_samples/synhtetic_pop_{city}_{date}_ablation.csv")
            df_vae95.to_csv(airbnb_path + f"pop_samples/synhtetic_pop_{city}_{date}_ablation_95_ablation.csv")
            #df_nfvae.to_csv(airbnb_path + f"pop_samples/synhtetic_pop_{city}_{date}.csv")
            df_nfvae95.to_csv(airbnb_path + f"pop_samples/synhtetic_pop_{city}_{date}_95.csv")
            
            #df_ipf.to_csv(airbnb_path + f"pop_samples/{city}_df_ipf_{date}.csv")
            #df_ipf95.to_csv(airbnb_path + f"pop_samples/{city}_df_ipf95_{date}.csv")
            for with_bins, shuffle_province in dfs_shuffle.keys():
                dfs_shuffle[(with_bins, shuffle_province)].to_csv(airbnb_path + f"pop_samples/{city}_df_shuffle_with_bins{with_bins}_shuffle_province{shuffle_province}.csv")
                dfs_shuffle95[(with_bins, shuffle_province)].to_csv(airbnb_path + f"pop_samples/{city}_df_shuffle95_with_bins{with_bins}_shuffle_province{shuffle_province}.csv")
            df_copula.to_csv(airbnb_path +f"pop_samples/{city}_df_copula_{date}.csv")
            df_nf_copula.to_csv(airbnb_path + f"pop_samples/{city}_df_nf_copula_{date}.csv")
            df_copula95.to_csv(airbnb_path + f"pop_samples/{city}_df_copula95_{date}.csv")
            df_nf_copula95.to_csv(airbnb_path + f"pop_samples/{city}_df_nf_copula95_{date}.csv")
            t1 = time()
            print(round(t1 - t0), "s")
        except Exception as e:
            print(city, e)
