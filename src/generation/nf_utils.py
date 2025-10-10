# *********** Normalizing Flows for mapping geographic coordinates into a uniform latent space ****************
# In the model we train a Normalizing Flows that takes as input a dataset of the geographical coordinates
# and transform them into a Uniform latent space.
# To generate new samples, of houses with realistic coordinates it is sufficient to sample from a uniform distribution
# and apply the inverse transformation to go back to the real coordinates.
# 
# 
# See https://github.com/karpathy/pytorch-normalizing-flows as reference
# 
# 


import pandas as pd
import numpy as np
import sys
sys.path += ['../src/'] 
from time import time
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

sys.path += [".."]
sys.path += ["../src/evaluation"]

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

    # normalize the geographic coordinates
    # we use MinMaxScaler, because we use a Uniform prior, so we want the observation in [-1,1]
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range = (-1,1))
    scaler.fit(df)
    df = scaler.transform(df)
    
    t0 = time()

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
    # and apply the inverse tranformation
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


def load_nf(path):
    with open(path, "rb") as handle:
        nf_dict = pickle.load(handle)
    return nf_dict


def nf_latent_to_real(nf_dict, xy_latent):
    return nf_dict["flow"].flow.backward(torch.logit(torch.clamp(torch.tensor(xy_latent), min = 1e-4, max = 1- 1e-4)))[0][-1].detach().numpy()
