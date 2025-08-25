"""
In this script, I will do the MIA analyses, considering the distance for all homes
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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score


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
    min_dists = dists.min(axis=1)

    return min_dists


def min_distance_to_synth(x_real, x_synth, metric="euclidean"):
    """This function returns the minimum distance between real and synthetic data"""
    if(metric=='euclidean'):
        dists = pairwise_distances(x_real, x_synth, metric=metric)
    elif(metric=='norm1'):
        dists = pairwise_distances(x_real, x_synth, metric='minkowski', p=1)
    elif(metric=='gower'):
        dists = gower.gower_matrix(x_real, x_synth)
    
    return dists.min(axis=1) 





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


def Get_Train_Test_set(df_train,df_test,f=0.80,seed=42):
    df_train_1 = df_train.sample(frac=f,random_state=seed)
    df_train_2 = df_train.drop(index=df_train_1.index)

    df_test_1 = df_test.sample(frac=f,random_state=seed)
    df_test_2 = df_test.drop(index=df_test_1.index)

    df_train_out = pd.concat([df_train_1,df_test_1]).reset_index(drop=True)
    df_test_out = pd.concat([df_train_2,df_test_2]).reset_index(drop=True)

    """Xtrain = df_train.data.values.reshape(-1,1)
    ytrain = df_train.label.values

    Xtest = df_test.data.values.reshape(-1,1)
    ytest = df_test.label.values"""

    return df_train_out, df_test_out

def MIA_Table_Test(df_train,df_test,control_data,seed = 42):
    """
    This function returns several dataframe, each dataframe reports the performance according to a measure for the classification problem.
    The output dataframes have along the rows the synthetic populations nd along the columns the classificators.
    
    """
    res_rocauc=pd.DataFrame(columns=['Logistic Regression','GaussianNB','KNeighbors','DecisionTree',
                                     'Random Forest', 'SVC', 'MLP'])
    res_aucpr=pd.DataFrame(columns=['Logistic Regression','GaussianNB','KNeighbors','DecisionTree',
                                     'Random Forest', 'SVC', 'MLP'])
    res_precision=pd.DataFrame(columns=['Logistic Regression','GaussianNB','KNeighbors','DecisionTree',
                                     'Random Forest', 'SVC', 'MLP'])
    res_recall=pd.DataFrame(columns=['Logistic Regression','GaussianNB','KNeighbors','DecisionTree',
                                     'Random Forest', 'SVC', 'MLP'])
    res_f1=pd.DataFrame(columns=['Logistic Regression','GaussianNB','KNeighbors','DecisionTree',
                                     'Random Forest', 'SVC', 'MLP'])

    for i in control_data:
        roc_auc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        aucpr_list = []

        X = df_train[i].values.reshape(-1,1)
        y = df_train['label_'+i].values

        Xtest = df_test[i].values.reshape(-1,1)
        ytest = df_test['label_'+i].values

        # Logistic Regression
        model = LogisticRegression(random_state=seed)
        model.fit(X, y)
        y_pred_proba = model.predict_proba(Xtest)[:,1]
        roc_auc_list.append(roc_auc_score(ytest, y_pred_proba))
        aucpr_list.append(average_precision_score(ytest,y_pred_proba))
        y_pred = model.predict(Xtest)
        precision_list.append(precision_score(ytest,y_pred))
        recall_list.append(recall_score(ytest,y_pred))
        f1_list.append(f1_score(ytest,y_pred))
        del model
        del y_pred_proba
        del y_pred

        # GaussianNB
        model = GaussianNB()
        model.fit(X, y)
        y_pred_proba = model.predict_proba(Xtest)[:,1]
        roc_auc_list.append(roc_auc_score(ytest, y_pred_proba))
        aucpr_list.append(average_precision_score(ytest,y_pred_proba))
        y_pred = model.predict(Xtest)
        precision_list.append(precision_score(ytest,y_pred))
        recall_list.append(recall_score(ytest,y_pred))
        f1_list.append(f1_score(ytest,y_pred))
        del model
        del y_pred_proba
        del y_pred

        # KNeighbors
        model = KNeighborsClassifier()
        model.fit(X, y)
        y_pred_proba = model.predict_proba(Xtest)[:,1]
        roc_auc_list.append(roc_auc_score(ytest, y_pred_proba))
        aucpr_list.append(average_precision_score(ytest,y_pred_proba))
        y_pred = model.predict(Xtest)
        precision_list.append(precision_score(ytest,y_pred))
        recall_list.append(recall_score(ytest,y_pred))
        f1_list.append(f1_score(ytest,y_pred))
        del model
        del y_pred_proba
        del y_pred

        # DecisionTree
        model = DecisionTreeClassifier(random_state=seed)
        model.fit(X, y)
        y_pred_proba = model.predict_proba(Xtest)[:,1]
        roc_auc_list.append(roc_auc_score(ytest, y_pred_proba))
        aucpr_list.append(average_precision_score(ytest,y_pred_proba))
        y_pred = model.predict(Xtest)
        precision_list.append(precision_score(ytest,y_pred))
        recall_list.append(recall_score(ytest,y_pred))
        f1_list.append(f1_score(ytest,y_pred))
        del model
        del y_pred_proba
        del y_pred

        # Random Forest
        model = RandomForestClassifier(random_state=seed)
        model.fit(X, y)
        y_pred_proba = model.predict_proba(Xtest)[:,1]
        roc_auc_list.append(roc_auc_score(ytest, y_pred_proba))
        aucpr_list.append(average_precision_score(ytest,y_pred_proba))
        y_pred = model.predict(Xtest)
        precision_list.append(precision_score(ytest,y_pred))
        recall_list.append(recall_score(ytest,y_pred))
        f1_list.append(f1_score(ytest,y_pred))
        del model
        del y_pred_proba
        del y_pred

        # SVC
        model = SVC(probability=True, random_state=seed)
        model.fit(X, y)
        y_pred_proba = model.predict_proba(Xtest)[:,1]
        roc_auc_list.append(roc_auc_score(ytest, y_pred_proba))
        aucpr_list.append(average_precision_score(ytest,y_pred_proba))
        y_pred = model.predict(Xtest)
        precision_list.append(precision_score(ytest,y_pred))
        recall_list.append(recall_score(ytest,y_pred))
        f1_list.append(f1_score(ytest,y_pred))
        del model
        del y_pred_proba
        del y_pred

        # MLP
        model = MLPClassifier(random_state=seed)
        model.fit(X, y)
        y_pred_proba = model.predict_proba(Xtest)[:,1]
        roc_auc_list.append(roc_auc_score(ytest, y_pred_proba))
        aucpr_list.append(average_precision_score(ytest,y_pred_proba))
        y_pred = model.predict(Xtest)
        precision_list.append(precision_score(ytest,y_pred))
        recall_list.append(recall_score(ytest,y_pred))
        f1_list.append(f1_score(ytest,y_pred))
        del model
        del y_pred_proba
        del y_pred

        # wide output dataframe
        res_rocauc.loc[i,:] = roc_auc_list
        res_aucpr.loc[i,:] = aucpr_list
        res_precision.loc[i,:] = precision_list
        res_recall.loc[i,:] = recall_list
        res_f1.loc[i,:] = f1_list

    # long output dataframe
    res_rocauc = res_rocauc.reset_index(drop=False,names='pop_name')
    res_rocauc = pd.melt(res_rocauc,id_vars=["pop_name"],var_name=['classifier'],value_name='score')
    res_aucpr = res_aucpr.reset_index(drop=False,names='pop_name')
    res_aucpr = pd.melt(res_aucpr,id_vars=["pop_name"],var_name=['classifier'],value_name='score')
    res_precision = res_precision.reset_index(drop=False,names='pop_name')
    res_precision = pd.melt(res_precision,id_vars=["pop_name"],var_name=['classifier'],value_name='score')
    res_recall = res_recall.reset_index(drop=False,names='pop_name')
    res_recall = pd.melt(res_recall,id_vars=["pop_name"],var_name=['classifier'],value_name='score')
    res_f1 = res_f1.reset_index(drop=False,names='pop_name')
    res_f1 = pd.melt(res_f1,id_vars=["pop_name"],var_name=['classifier'],value_name='score')

    return res_rocauc, res_aucpr, res_precision, res_recall, res_f1


## MAIN

metrics = ['euclidean','norm1', 'gower'] #'euclidean',
folder_path = '/data/housing/data/intermediate/lc_privacyStats/AllHomes_MIA/'

for metric in metrics:
    MIA_res_auc_roc = pd.DataFrame()
    MIA_res_auc_pr = pd.DataFrame()

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for file in tqdm(sorted(glob(f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_*.pickle'))[0:4]):
        prov = file.split(".")[-2][-2:]
        # data loading
        with open(file, 'rb') as f:
            all_baselines = pickle.load(f)

        all_baselines['df_excluded'] = all_baselines['df_real'][~all_baselines['df_real'].index.isin(all_baselines['df_real95'].index)]

        del all_baselines['df_real']

        # data preparation
        data = DataPreparation_Privacy1(data=all_baselines)

        control_data = [i for i in data.keys() if '95' in i and 'real' not in i]

        # the features are the distances
        
        for i in control_data:
            train_features = min_distance_to_synth(data['df_real95'].values, data[i].values,metric=metric)#.reshape(-1,1)
            y_train = np.array([1]*len(train_features))
            holdout_features = min_distance_to_synth(data['df_excluded'].values, data[i].values,metric=metric)#.reshape(-1,1)
            y_test = np.array([0]*len(holdout_features))

            df_train_one_pop = pd.DataFrame({i:train_features,'label_'+i:y_train})
            df_test_one_pop = pd.DataFrame({i:holdout_features,'label'+i:y_test})

            df_train = pd.concat([df_train,df_train_one_pop],axis=1)
            df_test = pd.concat([df_test,df_test_one_pop],axis=1)


    # defining training and testing set
    DF_train, DF_test = Get_Train_Test_set(df_train,df_test)

    res_auc_roc, res_auc_pr, _, _, _ = MIA_Table_Test(DF_train,DF_test,control_data)

    # saving
    MIA_res_auc_roc.to_csv(folder_path+f'MIA_auc_roc_{metric}.csv',index=False)
    MIA_res_auc_pr.to_csv(folder_path+f'MIA_auc_pr_{metric}.csv',index=False)

    
