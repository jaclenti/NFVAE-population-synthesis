# *********** Normalizing Flows for mapping geographical coordinates into a uniform latent space ****************
# In the model we train a Normalizing Flows that takes as input a dataset of the geographical coordinates
# and transform them into a Uniform latent space.
# To generate new samples, of houses with realistic coordinates it is sufficient to sample from a uniform distribution
# and apply the inverse transformation to go back to the real coordinates.
# 
# the hyperparameter tuning is then done with jl_nflows_tuning_2.sh 
# a GNU parallel script to run a set of experiments in parallel with different configuration of hyperparameters.
# 
# See https://github.com/karpathy/pytorch-normalizing-flows as reference
# 
# 




import pandas as pd
import numpy as np
import sys
sys.path += ['../src/'] 
import utils
import config
from _10_functions_and_lists import *
from time import time
import jl_vae
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim
sys.path += ["../src/pytorch-normalizing-flows/"]
import nflib
from nflib.flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow, 
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
)
from nflib.spline_flows import NSF_AR, NSF_CL
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import itertools
import pickle


import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)

def train_model(df, num_layers = 12, hidden_features  = 16, 
                lr = 0.001, num_epochs = 1000, opt_name = "Adam", 
                flow_name = "MAF",
                normalize_df = True, hide_progress = True, 
                batch_dim = 1000, path = None, output = True, model_settings = {}):
    
    df = df.astype(np.float32)
    prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) # Logistic distribution
    # several Flows are available in the package nflib 
    # in this step, we build the model, by creating the sequence of Flows (invertible transformation)
    if flow_name == "RealNVP":
        flows = [AffineHalfFlow(dim=2, parity=i%2, nh = hidden_features) for i in range(num_layers)]
    elif flow_name == "NICE":
        flows = [AffineHalfFlow(dim=2, parity=i%2, nh = hidden_features, scale=False) for i in range(num_layers)]
        flows.append(AffineConstantFlow(dim=2, shift=False))
    elif flow_name == "MAF":
        flows = [MAF(dim=2, parity=i%2, nh = hidden_features) for i in range(num_layers)]    
    elif flow_name == "Glow":
        flows = [Invertible1x1Conv(dim=2) for i in range(num_layers)]
        norms = [ActNorm(dim=2) for _ in flows]
        couplings = [AffineHalfFlow(dim=2, parity=i%2, nh = hidden_features) for i in range(len(flows))]
        flows = list(itertools.chain(*zip(norms, flows, couplings))) # append a coupling layer after each 1x1
    elif flow_name == "NSF_CL":
        nfs_flow = NSF_CL if True else NSF_AR
        flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim = hidden_features) for _ in range(num_layers)]
        convs = [Invertible1x1Conv(dim=2) for _ in flows]
        norms = [ActNorm(dim=2) for _ in flows]
        flows = list(itertools.chain(*zip(norms, convs, flows)))
    else:
        print("No flow")
    # initialize the NFs with the prior and the sequence of flows
    model = NormalizingFlowModel(prior, flows)

    # initialize the optimizer
    if opt_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr = lr)
    elif opt_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr = lr)
    elif opt_name == "RMSprop":
        opt = torch.optim.RMSprop(model.parameters(), lr = lr)
    else:
        print(opt_name)
        print("No optimizer")

    out = model_settings
    n_obs = len(df)

    # normalize the geographical coordinates
    # we use MinMaxScaler, because we use a Uniform prior, so we want the observation in [-1,1]
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range = (-1,1))
    scaler.fit(df)
    df = scaler.transform(df)
    
    t0 = time()
    # x = torch.tensor(df)
    losses = []
    model.train()
    for i in tqdm(range(num_epochs + 1), disable = hide_progress):
        df_batch = df[np.random.randint(low = 0, high = n_obs, size = [batch_dim]), :]
        x_batch = torch.tensor(df_batch)
        
        zs, prior_logprob, log_det = model(x_batch)
        logprob = prior_logprob + log_det
        loss = -torch.sum(logprob) # NLL

        model.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        
        if (i > 0)&(i % 1000 == 0):        
            t1 = time()
            # print(round(t1 - t0))
            out.update({"flow": model, "scaler": scaler, 
                        "losses": losses, "time": t1 - t0})
            
            if path:
                with open(path, "wb") as handle:
                    pickle.dump(out, handle)
    t1 = time()
    out.update({"flow": model, "scaler": scaler, 
                "losses": losses, "time": t1 - t0})
            
    if path:
        with open(path, "wb") as handle:
            pickle.dump(out, handle)

    if output:
        return out
    


def sample_from_flow(flow, df, n = None, remove_outliers = True):
    # sample from the prior (uniform distribution, with logistic transformation)
    # and aplpy the inverse tranformation
    prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) # Logistic distribution
    if not n:
        n = len(df)
    prior_samples = prior.sample((n,))
    new_samples = flow.backward(prior_samples)[0][-1].detach().numpy()

    if remove_outliers:
        x_lims, y_lims = np.quantile(new_samples[:,0], q = [0.001, 0.999]), np.quantile(new_samples[:,1], q = [0.001, 0.999])
        new_samples = new_samples[(x_lims[0] < new_samples[:,0])&(new_samples[:,0] < x_lims[1])&((y_lims[0] < new_samples[:,1])&(new_samples[:,1] < y_lims[1]))]

    return pd.DataFrame(new_samples, columns = df.columns)

def df_from_samples(new_samples, df, remove_outliers = True):
    if remove_outliers:
        x_lims, y_lims = np.quantile(new_samples[:,0], q = [0.001, 0.999]), np.quantile(new_samples[:,1], q = [0.001, 0.999])
        new_samples = new_samples[(x_lims[0] < new_samples[:,0])&(new_samples[:,0] < x_lims[1])&((y_lims[0] < new_samples[:,1])&(new_samples[:,1] < y_lims[1]))]
        return pd.DataFrame(new_samples, columns = df.columns)

# to load the trained NFs
# initialize the model with the correct layers and flows, and then load the parameters of the trained model
def load_nf(path, flow_name, hidden_features, num_layers):
    with open(path, "rb") as handle:
        nf_dict = pickle.load(handle)
    
    prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) # Logistic distribution
    
    if flow_name == "RealNVP":
        flows = [AffineHalfFlow(dim=2, parity=i%2, nh = hidden_features) for i in range(num_layers)]
    elif flow_name == "NICE":
        flows = [AffineHalfFlow(dim=2, parity=i%2, nh = hidden_features, scale=False) for i in range(num_layers)]
        flows.append(AffineConstantFlow(dim=2, shift=False))
    elif flow_name == "MAF":
        flows = [MAF(dim=2, parity=i%2, nh = hidden_features) for i in range(num_layers)]    
    elif flow_name == "Glow":
        flows = [Invertible1x1Conv(dim=2) for i in range(num_layers)]
        norms = [ActNorm(dim=2) for _ in flows]
        couplings = [AffineHalfFlow(dim=2, parity=i%2, nh = hidden_features) for i in range(len(flows))]
        flows = list(itertools.chain(*zip(norms, flows, couplings))) # append a coupling layer after each 1x1
    elif flow_name == "NSF_CL":
        nfs_flow = NSF_CL if True else NSF_AR
        flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim = hidden_features) for _ in range(num_layers)]
        convs = [Invertible1x1Conv(dim=2) for _ in flows]
        norms = [ActNorm(dim=2) for _ in flows]
        flows = list(itertools.chain(*zip(norms, convs, flows)))
    else:
        print("No flow")

    model = NormalizingFlowModel(prior, flows)
    model.state_dict = nf_dict["flow"].state_dict
    nf_dict["model"] = model

    return nf_dict

def transform_geo_coordinates_with_nf(df, nf_dict):
    coords = df[["GEO_LONGITUDINE_BENE_ROUNDED","GEO_LATITUDINE_BENE_ROUNDED"]]
    
    scaled_coords = nf_dict["scaler"].transform(coords)
    gaussian_coords = nf_dict["flow"].transform_to_noise(torch.tensor(scaled_coords)).detach().numpy()
    
    df_out = df.copy()#.drop(columns = ["GEO_LONGITUDINE_LATENT", "GEO_LATITUDINE_LATENT"])
    df_out[["GEO_LONGITUDINE_BENE_ROUNDED","GEO_LATITUDINE_BENE_ROUNDED"]] =  gaussian_coords
    df_out.rename(columns = {"GEO_LONGITUDINE_BENE_ROUNDED": "GEO_LONGITUDINE_LATENT","GEO_LATITUDINE_BENE_ROUNDED": "GEO_LATITUDINE_LATENT"}, inplace = True)
    return df_out

def nf_latent_to_real(nf_dict, xy_latent):
    return nf_dict["flow"].flow.backward(torch.logit(torch.clamp(torch.tensor(xy_latent), min = 1e-4, max = 1- 1e-4)))[0][-1].detach().numpy()


# run the script for training and saving the model
# for the province of Bologna (jl_vae.get_old_df_vae())
if __name__ == "__main__":
    num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf, flow_name, date =  sys.argv[1:]
    batch_dim = 100
    id = "_".join([num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf, flow_name, str(batch_dim), date])
    
    num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_ = int(num_layers_nf), int(hidden_features_nf), int(num_epochs_nf), int(lr_nf_)
    print("start", num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf, flow_name)
    lr_nf = lr_nf_ / 1000000
    
    
    model_settings = {"num_layers_nf": num_layers_nf, "hidden_features_nf": hidden_features_nf, 
                      "num_epochs_nf": num_epochs_nf, "lr_nf": lr_nf, "opt_nf": opt_nf, "flow_name": flow_name,
                       "batch_dim": batch_dim, "date": date}

    df_vae = jl_vae.get_old_df_vae()
    # df_vae = jl_vae.get_df_prov("BO")
    
    nf_dict = train_model(df_vae.rename(columns = {"x": "GEO_LONGITUDINE_BENE_ROUNDED", "y": "GEO_LATITUDINE_BENE_ROUNDED"})
                          [["GEO_LONGITUDINE_BENE_ROUNDED", "GEO_LATITUDINE_BENE_ROUNDED"]],
                                 num_layers = num_layers_nf, hidden_features = hidden_features_nf,
                                 num_epochs = num_epochs_nf, opt_name = opt_nf, batch_dim = batch_dim, flow_name = flow_name,
                                 hide_progress = False, path = jl_vae.path_pop_synth + id + ".pkl", model_settings = model_settings)
    
    print("done", num_layers_nf, hidden_features_nf, num_epochs_nf, lr_nf_, opt_nf)
    
    
