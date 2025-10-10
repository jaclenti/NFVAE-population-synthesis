# fidelity local features
# the script compute the similarity between real and synthetic data as the average distance over j between
# the average real homes in cell c_j and synthetic homes in cell c_j
# - divide the region in grid cells c_j
# - compute the average synthetic and real home in c_j
# - project the homes in the principal components explaining 95% of the variance of real data
# - compute the distance between projected data

import pandas as pd
import numpy as np
import sys
sys.path += [".."]
sys.path += ["../src/generation"]
import config
from glob import glob
import pickle

from sklearn.decomposition import PCA


def get_df_bins(df_, df_real, bin_size = 0.01):
    df = df_.copy()

    bins_x = np.arange(df_real["x"].min() - bin_size / 2, df_real["x"].max() + bin_size / 2, bin_size)
    bins_y = np.arange(df_real["y"].min() - bin_size / 2, df_real["y"].max() + bin_size / 2, bin_size)

    df["bin_x"] = pd.cut(df["x"], bins = bins_x, labels = np.arange(len(bins_x)-1))
    df["bin_y"] = pd.cut(df["y"], bins = bins_y, labels = np.arange(len(bins_y)-1))
    df.drop(columns = ["x", "y"], inplace = True)

    return df


if __name__ == "__main__":
    all_geo_features = {}

    for file in sorted(glob(config.baselines_path + f'all_baselines_*.pickle')):
        city = file.split(".")[-2].split("_")[-1]

        all_geo_features[city] = {}

        with open(file, 'rb') as f:
            data = pickle.load(f)
        data = {k: data[k].reset_index(drop = True) for k in data}

        pca = PCA(n_components = 14)
        pca.fit(data["df_real"].drop(columns = ["x", "y"]))
        n_components95 = (pca.explained_variance_ratio_.cumsum() < .95).sum()


        data_pca = {k: pd.concat([data[k][["x", "y"]], pd.DataFrame(pca.transform(data[k].fillna(0).drop(columns = ["x", "y"]))[:,:n_components95])], axis = 1) for k in data if "95" not in k}
        df_real = get_df_bins(data_pca["df_real"].copy(), data_pca["df_real"])

        for k in data:
            if "95" not in k:
                df_k = get_df_bins(data_pca[k], data_pca["df_real"])
                dist = (np.abs((df_real.groupby(["bin_x", "bin_y"]).mean() - df_k.groupby(["bin_x", "bin_y"]).mean()).dropna()).mean() * pca.explained_variance_ratio_[:n_components95]).sum()
                all_geo_features[city][k] = dist

                df_geo_features = pd.DataFrame(all_geo_features)
    df_geo_features.to_csv(config.path_airbnb + f"similarity_grid_geo_features_pca_airbnb.csv")
    






