# OBSOLETE VERSION
# INITIAL IMPLEMENTATION OF NORMALIZING FLOWS
# THE UPDATED VERSION IS
# jl_nflows_geo_coordinates_2.py
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path += ['../src/'] 
import utils
import config
from _10_functions_and_lists import *

from time import time

import seaborn as sns
from matplotlib.pyplot import subplots as sbp
from itertools import combinations

import jl_vae
import torch
from glob import glob
from importlib import reload

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim

#from nflows.flows.base import Flow
#from nflows.distributions.normal import StandardNormal
#from nflows.distributions.base import Distribution
#from nflows.utils import torchutils
#from nflows.transforms.base import CompositeTransform
#from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
#from nflows.transforms.permutations import ReversePermutation, RandomPermutation

import pickle

torch.autograd.set_detect_anomaly(True)

def train_model(df, num_layers = 12, hidden_features  = 16, lr = 0.001, num_epochs = 1000, opt_name = "Adam",
                normalize_df = True, hide_progress = True, batch_dim = 1000, path = None, output = True, model_settings = {}):
    
    base_dist = StandardNormal(shape=[2])

    transforms = []
    for _ in range(num_layers):
        # transforms.append(ReversePermutation(features=2))
        transforms.append(RandomPermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2, 
                                                              hidden_features = hidden_features))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    if opt_name == "Adam":
        opt = torch.optim.Adam(flow.parameters(), lr = lr)
    elif opt_name == "SGD":
        opt = torch.optim.SGD(flow.parameters(), lr = lr)
    elif opt_name == "RMSprop":
        opt = torch.optim.RMSprop(flow.parameters(), lr = lr)
    else:
        print(opt_name)
        print("No optimizer")

    
    if normalize_df:
        scaler = StandardScaler()
        scaler.fit(df)
        df = scaler.transform(df)
    
    # x = torch.tensor(df)
    
    losses = []
    loss0 = 1e9
    t0 = time()
    n_obs = len(df)

    for i in tqdm(range(num_epochs), disable = hide_progress):
        
        df_batch = df[np.random.randint(low = 0, high = n_obs, size = [batch_dim]), :]
        x_batch = torch.tensor(df_batch)
        # x_batch = x[torch.randint(low = 0, high =  len(x), size = (batch_dim,)),:]
        opt.zero_grad()
        loss = -flow.log_prob(inputs = x_batch).mean()
        if loss / loss0 > 10:
            print("continue")
            continue
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm = 1.)
        opt.step()
        losses.append(loss.item())
        # loss0 = loss.item()

        if (i > 0)&(i % 1000 == 0):        
            t1 = time()
            # print(round(t1 - t0))
            out = {"flow": flow, "scaler": scaler, "losses": losses, "time": t1 - t0}
            
            if path:
                with open(path, "wb") as handle:
                    pickle.dump(out, handle)

    out.update(model_settings)
    if output:
        return out
    

def sample_from_flow(flow, df, n = None, remove_outliers = True):
    if not n:
        n = len(df)
    new_samples = flow.sample_and_log_prob(n)[0].detach().numpy()
    
    if remove_outliers:
        x_lims, y_lims = np.quantile(new_samples[:,0], q = [0.001, 0.999]), np.quantile(new_samples[:,1], q = [0.001, 0.999])
        new_samples = new_samples[(x_lims[0] < new_samples[:,0])&(new_samples[:,0] < x_lims[1])&((y_lims[0] < new_samples[:,1])&(new_samples[:,1] < y_lims[1]))]

    return pd.DataFrame(new_samples, columns = df.columns)


def load_nf(path):
    with open(path, "rb") as handle:
        nf_dict = pickle.load(handle)
    return nf_dict

def transform_geo_coordinates_with_nf(df, nf_dict):
    coords = df[["GEO_LONGITUDINE_BENE_ROUNDED","GEO_LATITUDINE_BENE_ROUNDED"]]
    
    scaled_coords = nf_dict["scaler"].transform(coords)
    gaussian_coords = nf_dict["flow"].transform_to_noise(torch.tensor(scaled_coords)).detach().numpy()
    
    df_out = df.copy()#.drop(columns = ["GEO_LONGITUDINE_LATENT", "GEO_LATITUDINE_LATENT"])
    df_out[["GEO_LONGITUDINE_BENE_ROUNDED","GEO_LATITUDINE_BENE_ROUNDED"]] =  gaussian_coords
    df_out.rename(columns = {"GEO_LONGITUDINE_BENE_ROUNDED": "GEO_LONGITUDINE_LATENT","GEO_LATITUDINE_BENE_ROUNDED": "GEO_LATITUDINE_LATENT"}, inplace = True)
    return df_out




if __name__ == "__main__":
    num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf, date =  sys.argv[1:]
    batch_dim = 100
    id = "_".join([num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf, str(batch_dim), date])
    
    num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_ = int(num_layers_nf), int(hidden_features_nf), int(num_epochs_nf), int(lr_nf_)
    print("start", num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf)
    lr_nf = lr_nf_ / 1000000
    
    
    model_settings = {"num_layers_nf": num_layers_nf, "hidden_features_nf": hidden_features_nf, 
                      "num_epochs_nf": num_epochs_nf, "lr_nf": lr_nf, "opt_nf": opt_nf,
                       "batch_dim": batch_dim, "date": date}

    df_vae = jl_vae.get_df_vae()
    
    nf_dict = train_model(df_vae[["GEO_LONGITUDINE_BENE_ROUNDED", "GEO_LATITUDINE_BENE_ROUNDED"]],
                                 num_layers = num_layers_nf, hidden_features = hidden_features_nf,
                                 num_epochs = num_epochs_nf, opt_name = opt_nf, batch_dim = batch_dim,
                                 hide_progress = True, path = jl_vae.path_pop_synth + id + ".pkl", model_settings = model_settings)
    
    print("done", num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf)
    
    
