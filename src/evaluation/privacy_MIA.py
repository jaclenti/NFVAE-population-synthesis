# privacy preservation, measured as the robustness against membership inference attack
# - train the genrator on 95% of real samples
# - split the real data in train - test
# - train a classifier to classify the samples used to train the generator (95%)
# - the classifier is based only on the distance (comprising all features) between the sample and the closest observation in the synthetic data
# - we train several classifiers


# import libraries
import sys
sys.path += [".."]
sys.path += ["../src/generation"]
import pandas as pd
import numpy as np
from glob import glob
import config
import pickle
from tqdm import tqdm


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


def min_distance_to_synth(x_real, x_synth, metric="euclidean"):
    """This function returns the minimum distance between real and synthetic data"""
    if(metric=='euclidean'):
        dists = pairwise_distances(x_real, x_synth, metric=metric)
    elif(metric=='norm1'):
        dists = pairwise_distances(x_real, x_synth, metric='minkowski', p=1)
    elif(metric=='gower'):
        dists = gower.gower_matrix(x_real, x_synth)
    
    return dists.min(axis=1) 


def MIA_Table_Test(data,control_data,metric='euclidean',f=0.8,seed = 42):
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

        # the features are the distances
        train_features = min_distance_to_synth(data['df_real95'].values, data[i].values,metric=metric)#.reshape(-1,1)
        y_train = np.array([1]*len(train_features))
        holdout_features = min_distance_to_synth(data['df_excluded'].values, data[i].values,metric=metric)#.reshape(-1,1)
        y_test = np.array([0]*len(holdout_features))

        df_train = pd.DataFrame({'data':train_features,'label':y_train})
        df_test = pd.DataFrame({'data':holdout_features,'label':y_test})

        df_train_1 = df_train.sample(frac=f,random_state=seed)
        df_train_2 = df_train.drop(index=df_train_1.index)

        df_test_1 = df_test.sample(frac=f,random_state=seed)
        df_test_2 = df_test.drop(index=df_test_1.index)

        df_train = pd.concat([df_train_1,df_test_1]).reset_index(drop=True)
        df_test = pd.concat([df_train_2,df_test_2]).reset_index(drop=True)

        X = df_train.data.values.reshape(-1,1)
        y = df_train.label.values

        Xtest = df_test.data.values.reshape(-1,1)
        ytest = df_test.label.values

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



def DataPreparation_Privacy(data):
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





#%% MAIN



metrics = ['euclidean','norm1'] #'euclidean',


for metric in metrics:
    print(metric)

    MIA_res_auc_roc = pd.DataFrame()
    
    for file in tqdm(sorted(glob(config.populations_path + f'/all_baselines_*.pickle'))):
        prov = file.split('/')[-1].split('_')[-1].split('.')[0]
        # data loading
        with open(file, 'rb') as f:
            all_baselines = pickle.load(f)

        all_baselines['df_excluded'] = all_baselines['df_real'][~all_baselines['df_real'].index.isin(all_baselines['df_real95'].index)]

        del all_baselines['df_real']

        # data preparation
        data = DataPreparation_Privacy(data=all_baselines)

        control_data = [i for i in data.keys() if '95' in i and 'real' not in i]


        # MIA
        res_auc_roc, res_auc_pr, _, _, _ = MIA_Table_Test(data,control_data,metric=metric,f=0.8,seed = 42)
        res_auc_roc['city'] = prov

        MIA_res_auc_roc = pd.concat([MIA_res_auc_roc,res_auc_roc])
    MIA_res_auc_roc.to_csv(config.privacy_path+f'MIA_auc_roc_{metric}_airbnb_data_22092025.csv',index=False)
