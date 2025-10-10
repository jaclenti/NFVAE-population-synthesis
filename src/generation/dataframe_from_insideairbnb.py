# the functions of this script are used to scrape the data from insideairbnb.com

import pandas as pd
import numpy as np
import torch
import sys
sys.path += [".."]
sys.path += ["../src/evaluation"]
import vae
import nf_utils as nfg
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
import requests
import os




amenities_vars = ["Air conditioning", "Elevator", "Self check-in", "Pets allowed", "Private living room", "Backyard", "Pool"]
int_vars = ["accommodates", "bathrooms", "bedrooms", "beds"]


# example
# city = "copenhagen"
# url = "https://data.insideairbnb.com/denmark/hovedstaden/copenhagen/2025-03-23/"
# all urls in ../data/airbnb_urls
def save_gz(url, fn, city):
    resp = requests.get(url + "data/" + fn)
    resp.raise_for_status()
    path_city = config.path_insideairbnb + f"/{city}"
    if not os.path.exists(path_city):
        os.mkdir(path_city)

    with open(path_city + "/" + fn, "wb") as f:
        f.write(resp.content)


def read_gz(city, fn):
    return pd.read_csv(gzip.open(config.path_insideairbnb + f"/{city}/{fn}", mode = "rb"))

def read_json(city):
    gdf = gpd.read_file(config.path_insideairbnb + f"/{city}/neighbourhoods.geojson")
    return gdf


# process data from insideairbnb to have input dataframe for NF+VAE

def get_df_city_for_nf(city):
    df_full = read_gz(city, fn = "/listings.csv.gz")
    
    df_full["amenities"] = [json.loads(u) for u in df_full["amenities"]]
    # select variables in "amenities"
    for var in amenities_vars:
        df_full[var] = [var in u for u in df_full["amenities"]]
    # select other variables
    df = df_full[['id', 
                'latitude', 'longitude', 
                'room_type', 'accommodates', 'bathrooms',
                'bedrooms', 'beds', 'price',
                'review_scores_rating', 'reviews_per_month'] + amenities_vars].rename(columns = {"latitude": "y", "longitude": "x"}).dropna()
    # trasform and rename variables
    df["bathrooms"] = df["bathrooms"].astype(int)
    df["price"] = [float(u[1:].split(".")[0].replace(",","")) for u in df["price"]]
    df["log_price"] = np.log(df["price"])
    df = df.drop(columns = ["price"])
    df.rename(columns = {u: "flag_" + u for u in amenities_vars})
    df.rename(columns = {u: "int_" + u for u in int_vars})    
    
    # categorical in dummies
    df = pd.concat([df.drop(columns = ["room_type"]), 
                    pd.get_dummies(df["room_type"], prefix = "roomtype")], axis = 1).set_index("id")

    # scale geographic variables in [-1,1]
    scaler = MinMaxScaler((-1,1))
    scaler.fit(np.array(df[["x", "y"]]))
    df[["x_norm", "y_norm"]] = scaler.transform(np.array(df[["x", "y"]]))
    return df
