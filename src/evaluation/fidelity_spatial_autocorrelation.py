# spatial autocorrelation
# measure the spatial autocorrelation as the weighted average of the moran's index on the principal components (PCs)
# - compute the PCs explaining 95% of variance of the real data
# - for each synthetic population, project the data on the real 95% PCs
# - compute Moran's index with two different weighting function (k nearest neighbours, within radius distance)
# - compute spatial autocorrelation as the weighted average of the Moran's I for each component, weighted by the explained variance of each component

import pandas as pd
import numpy as np
from glob import glob 
import sys
sys.path += [".."]
sys.path += ["../src/generation"]
import config
import pickle
from fidelity_spatial_autocorrelation import get_moransI, get_localMoransI
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import geopandas as gpd
from sklearn.decomposition import PCA
import vae



def get_moransI_sparse(W, y):
    """
    Moran's I using sparse weight matrix W (csr_matrix).
    y is a 1D numpy array.
    """
    n = len(y)
    y = y.astype(float)
    y_mean = y.mean()
    y_diff = y - y_mean

    # denominator
    denom = np.sum(y_diff ** 2)

    # numerator (efficient sparse multiplication)
    num = y_diff @ (W @ y_diff)

    # normalization factor
    W_sum = W.sum()

    I = (n / W_sum) * (num / denom)
    return I


# compute Morans index
# W is the distance weight, can be modified 
def compute_spatial_autocorrelation(df, var, D = None, k = 20, local_moran = False, weighting = "within_CAP"):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:4326")
#    gdf = gdf.to_crs(epsg=3857)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    
    if weighting == "nearest_k":
        coords = np.array([(pt.x, pt.y) for pt in gdf.geometry])
        n = coords.shape[0]
        nn = NearestNeighbors(n_neighbors = k + 1, metric = "euclidean").fit(coords)
        dists, inds = nn.kneighbors(coords)         
        dists, inds = dists[:,1:], inds[:,1:] # remove self
        
        weights = np.exp(-dists / np.median(dists))
        row_idx = np.repeat(np.arange(n), k)
        col_idx = inds.ravel()
        w_ij = weights.ravel()

        W = csr_matrix((w_ij, (row_idx, col_idx)), shape=(n, n))
        W = 0.5 * (W + W.T)
        W = W.todense()

    if weighting == "dist_threshold":
        coords = np.array([(pt.x, pt.y) for pt in gdf.geometry])
        dists = pairwise_distances(coords)
        for k in range(len(dists)):
            dists[k,k] = np.inf
        threshold = np.percentile(dists, 1)
        W = (dists < threshold) + 0.
    
    return get_moransI_sparse(csr_matrix(W),np.array(df[var]))
    


if __name__ == "__main__":

    for file in sorted(glob(config.baselines_path + f'all_baselines_*.pickle')):
        moran_city = {}

        city = file.split(".")[-2].split("_")[-1]
        print(city)

        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)

            # embed real data through principal components achieving 95% explained variance
            pca = PCA(n_components = 13)
            pca.fit(data["df_real"].drop(columns = ["x", "y"]))
            n_components95 = (pca.explained_variance_ratio_.cumsum() < .95).sum()

            # use the same principal components to project all synthetic populations
            data_pca = {k: pd.concat([data[k][["x", "y"]], 
                                      pd.DataFrame(pca.transform(data[k].fillna(0).drop(columns = ["x", "y"]))[:,:n_components95], 
                                                   index = data[k].index)], axis = 1) 
                        for k in data if "95" not in k}
            
            for weighting in ["nearest_k","dist_threshold"]:
                # compute morans i on each principal component
                moran_pca = {k:{comp: compute_spatial_autocorrelation(data_pca[k],  comp, k = 20,
                                local_moran = False, weighting = weighting) for comp in range(n_components95)}
                                for k in data_pca}
                # morans as the weighted average of all single morans
                moran_city[city] = {k:pd.Series(moran_pca[k]) @ pca.explained_variance_ratio_[:n_components95] for k in moran_pca}
           
                with open(config.path_airbnb + f"spatial_autocorrelation/moran_{weighting}_{city}_pca.pkl", "wb") as f:
                    pickle.dump(moran_city, f, protocol = pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)
            



