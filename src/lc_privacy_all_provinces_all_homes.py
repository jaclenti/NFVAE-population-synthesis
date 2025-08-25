"""
In this script, I want to try having the variables of distances and ratio for 
each home, assembly them in a unique dataframe. This data structure will allow 
me to to do boxblots at the home scale (and not aggregating results for provinces 
as done in the script: lc_privacy_all_provinces.py)
"""


# import libraries
import sys
sys.path += ["../src"]
import utils
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.pyplot import subplots as sbp 
from importlib import reload
import jl_vae
import pickle
from tqdm import tqdm
# import jl_nflows_geo_coordinates_2 as nfg
# from jl_nflows_geo_coordinates import load_nf as load_dict

from _51_abm_functions import cod_prov_abbrv_df

# Global Spatial Autocorrelation
from spatial_autocorrelation import get_moransI, moransI_scatterplot, hypothesis_testing
# Local Spatial Autocorrelation
from spatial_autocorrelation import get_localMoransI, LISA_scatterplot
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import pairwise_distances
import gower


def DataPreparation_Privacy1(data):
    """Function to prepare the data for the privacy analyses"""
    

    for key in data.keys():
        data[key] = ConvertBool2number(data[key])

    df = pd.concat(data, ignore_index=False)
    df = df.reset_index(level=0).rename(columns={"level_0": "origine"})

    num_cols = df.select_dtypes(include=["number"]).columns
    #scaler = MinMaxScaler()
    df[num_cols] = (df[num_cols]-df[num_cols].min())/(df[num_cols].max()-df[num_cols].min())
    df = df.fillna(0)

    data_out = dict()
    for i in data.keys():
        a = df.loc[df.origine == i,:].drop(columns='origine').reset_index(drop=True)
        data_out[i] = a

    return data_out

def ConvertBool2number(df):
    """Function to convert boolean columns to numeric"""
    bool_cols = df.select_dtypes(include=bool).columns
    df[bool_cols] = df[bool_cols].astype(float)
    df = df.reset_index(drop=True)
    return df

def Distance_x1(x1,metric='euclidean'):
    """Function to compute the distance between all point in a set, exception made for the point at hand"""
    # Distance matrix
    if(metric=='euclidean'):
        dists = pairwise_distances(x1.values, metric='euclidean') 
    elif(metric=='norm1'):
        dists = pairwise_distances(x1.values, metric='minkowski', p=1)
    elif(metric == 'gower'):
        dists = gower.gower_matrix(x1)

    np.fill_diagonal(dists, np.inf) 
    #del df2
    # take the minimum distances
    #min_dists = dists.min(axis=1)

    return dists



def TableDistance_all_Homes(data, control_data,prov, test_data = 'df_real95',metric='euclidean'):
    """Function that computes the distance between the real records to the synthetic ones"""
    
    res = pd.DataFrame(columns= control_data + test_data)
    for i in (control_data+test_data):
        # prov_res = pd.DataFrame(columns=['pop_name','min_dist'])
        if i not in test_data:
            # Distance matrix
            if(metric=='euclidean'):
                dists = pairwise_distances(data[test_data[0]].values,data[i].values, metric='euclidean') 
            elif(metric=='norm1'):
                dists = pairwise_distances(data[test_data[0]].values,data[i].values, metric='minkowski', p=1)
            elif(metric == 'gower'):
                dists = gower.gower_matrix(data[test_data[0]].values,data[i].values)
            
            

        else:
            dists = Distance_x1(data[i],metric=metric)

        # take the minimum distances
        min_dists = dists.min(axis=1)
        res[i] = min_dists
        
    res['prov'] = prov
    return res 


def TableNNDR_allHomes(data, control_data,prov,test_data = 'df_real95',metric='euclidean'):
    """Function that compuets the ratio between the minimum and the second minimum distance between a synthetic and real record"""

    res = pd.DataFrame(columns=control_data+test_data) #,'mean_test','std_test'
    for i in control_data+test_data:
        if i not in test_data:
            # Distance matrix
            if(metric=='euclidean'):
                dists_train = pairwise_distances(data[i].values,data[test_data[0]].values, metric='euclidean') 
                #dists_test = pairwise_distances(data[i].values,data['real_excluded'].values, metric='euclidean') 
            elif(metric=='norm1'):
                dists_train = pairwise_distances(data[i].values,data[test_data[0]].values, metric='minkowski', p=1)
                #dists_test = pairwise_distances(data[i].values,data['real_excluded'].values, metric='minkowski', p=1)
            elif(metric == 'gower'):
                dists_train = gower.gower_matrix(data[i].values,data[test_data[0]].values,)
                #dists_test = gower.gower_matrix(data[i].values,data['real_excluded'].values)
            
        else:
            dists_train = Distance_x1(data[i],metric=metric)

        # take the minimum distances
        sorted_rows_train = np.sort(dists_train,axis=1)
        
        min_train = sorted_rows_train[:,0]
        second_min_train = sorted_rows_train[:,1]
        ratio_train = min_train/second_min_train
        
        res[i] = ratio_train

    res['prov'] = prov
        
    return res 


metrics = ['euclidean','norm1', 'gower'] #'euclidean',
folder_path = '/data/housing/data/intermediate/lc_privacyStats/AllHomes_distances/'

for metric in metrics:
    print(metric)

    res_dist = pd.DataFrame()
    res_ratio = pd.DataFrame()

    for file in tqdm(sorted(glob(f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_*.pickle'))):
        prov = file.split(".")[-2][-2:]
        # data loading
        with open(file, 'rb') as f:
            all_baselines = pickle.load(f)

        all_baselines['df_excluded'] = all_baselines['df_real'][~all_baselines['df_real'].index.isin(all_baselines['df_real95'].index)]

        del all_baselines['df_real']

        # data preparation
        data = DataPreparation_Privacy1(data=all_baselines)

        control_data = [i for i in data.keys() if '95' in i and 'real' not in i]

        # distances
        dist = TableDistance_all_Homes(data = data, control_data=control_data, test_data = ['df_real95'],metric=metric,prov=prov)

        res_dist = pd.concat([res_dist,dist])

        # ratios
        ratio = TableNNDR_allHomes(data=data, control_data=control_data,test_data = ['df_real95'],metric=metric,prov=prov)
        ratio['prov'] = prov

        res_ratio = pd.concat([res_ratio,ratio])

    # saving
    res_dist.to_csv(folder_path+f'distances_all_homes_{metric}.csv',index=False)
    res_ratio.to_csv(folder_path+f'ratio_all_homes_{metric}.csv',index=False)