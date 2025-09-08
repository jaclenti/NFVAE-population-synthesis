import pandas as pd
import numpy as np
from glob import glob 
import sys
sys.path += ["../src"]
import pickle
from spatial_autocorrelation import get_moransI, get_localMoransI
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import geopandas as gpd
from jl_synthetic_ipf_all_provinces import get_area_from_xy
from sklearn.decomposition import PCA
import jl_vae



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

    if weighting == "within_CAP":
        cap = get_area_from_xy(df, geo_dict["cap"]).fillna(0)
        W = (np.array(cap.astype(int))[:,None] - np.array(cap.astype(int))[None,:] == 0) + 0.
    
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

        # make symmetric by averaging (optional but common)
        W = 0.5 * (W + W.T)
        W = W.todense()

    if weighting == "dist_threshold":
        coords = np.array([(pt.x, pt.y) for pt in gdf.geometry])
        dists = pairwise_distances(coords)
        for k in range(len(dists)):
            dists[k,k] = np.inf
        threshold = np.percentile(dists, 1)
        W = (dists < threshold) + 0.
    
    if weighting == "exp":
        coords = np.array([(pt.x, pt.y) for pt in gdf.geometry])
        dists = pairwise_distances(coords)
        for k in range(len(dists)):
            dists[k,k] = np.inf
        
        W = np.exp(-dists / np.percentile(dists,1))
        W = W * (dists < np.percentile(dists,10))

    if local_moran:
        return get_localMoransI(W,np.array(df[var]), list(df.index))
    else:
        return get_moransI_sparse(csr_matrix(W),np.array(df[var]))
    


if __name__ == "__main__":

    geo_dict = jl_vae.load_geo_data()    


    #for file in sorted(glob(f'/data/housing/data/intermediate/jl_pop_synth/isp_baselines/all_baselines_*.pickle')): # kill MI, RM
    for file in sorted(glob(f'/data/housing/data/intermediate/jl_pop_synth/airbnb_baselines/all_baselines_*.pickle'))[5:]: # kill barcelona
        moran_prov = {}

        prov = file.split(".")[-2].split("_")[-1]
        print(prov)

        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            
            pca = PCA(n_components = 14)
            pca.fit(data["df_real"].drop(columns = ["x", "y"]))
            n_components95 = (pca.explained_variance_ratio_.cumsum() < .95).sum()


            data_pca = {k: pd.concat([data[k][["x", "y"]], pd.DataFrame(pca.transform(data[k].fillna(0).drop(columns = ["x", "y"]))[:,:n_components95], index = data[k].index)], axis = 1) for k in data if "95" not in k}

            for weighting in ["nearest_k","dist_threshold","exp"]:
                moran_pca = {k:{comp: compute_spatial_autocorrelation(data_pca[k],  comp, k = 20,
                                local_moran = False, weighting = weighting) for comp in range(n_components95)}
                                for k in data_pca}
                
                moran_prov[prov] = {k:pd.Series(moran_pca[k]) @ pca.explained_variance_ratio_[:n_components95] for k in moran_pca}
           
                with open(f"/data/housing/data/intermediate/jl_pop_synth/spatial_autocorrelation/moran_{weighting}_{prov}_pca.pkl", "wb") as f:
                    pickle.dump(moran_prov, f, protocol = pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)
            



