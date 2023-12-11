from natsort import natsorted
import pandas as pd
from typing import Optional
from anndata import AnnData
import numpy as np
from sklearn.decomposition import PCA
import scanpy as sc

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix


def spatial_reconstruction(
        adata_ori: AnnData,
        alpha: float = 1,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = True,
        normalize_total: bool = True,  # 使用总数归一化
        copy: bool = True,
        n_components: int = 20,
):
    adata = adata_ori.copy() if copy else adata_ori
    adata.layers['counts'] = adata.X  # adata.X是表达矩阵

    sc.pp.normalize_total(adata) if normalize_total else None
    sc.pp.log1p(adata)  # log(1+x)对偏度比较大的数据用log1p函数进行转化，使其更加服从高斯分布。
    adata.layers['log1p-ori'] = adata.X

    hvg = list(adata.var['highly_variable'][adata.var['highly_variable'].values].index)
    exmatrix_ori = adata.to_df(layer='log1p-ori')[hvg].to_numpy()
    pca_ori = PCA(n_components=n_components)
    pca_ori.fit(exmatrix_ori)
    exmatrix_ori = pca_ori.transform(exmatrix_ori)

    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    coord = adata.obsm['spatial']  # 4727*2 (x,y)

    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)  # 4727*4727 邻接矩阵？

    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1

    conns = nbrs.T.toarray() * dists

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X

    stg = conns / np.sum(conns, axis=0, keepdims=True)

    adata.X = csr_matrix(X_rec)
    adata.layers['log1p-aug'] = adata.X

    exmatrix_aug = adata.to_df(layer='log1p-aug')[hvg].to_numpy()
    pca_aug = PCA(n_components=n_components)
    pca_aug.fit(exmatrix_aug)
    exmatrix_aug = pca_ori.transform(exmatrix_aug)

    del adata.obsm['X_pca']

    '''记录重构参数'''
    adata.uns['spatial_reconstruction'] = {}
    rec_dict = adata.uns['spatial_reconstruction']
    rec_dict['params'] = {}
    rec_dict['params']['alpha'] = alpha
    rec_dict['params']['n_neighbors'] = n_neighbors
    rec_dict['params']['n_pcs'] = n_pcs
    rec_dict['params']['use_highly_variable'] = use_highly_variable
    rec_dict['params']['normalize_total'] = normalize_total

    return adata if copy else None, exmatrix_ori, exmatrix_aug, stg


def refine_spatial_domains(y_pred, coord, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coord)
    distances, indices = nbrs.kneighbors(coord)
    indices = indices[:, 1:]

    y_refined = pd.Series(index=y_pred.index, dtype='object')

    for i in range(y_pred.shape[0]):

        y_pred_count = y_pred[indices[i, :]].value_counts()

        if (y_pred_count.loc[y_pred[i]] < n_neighbors / 2) and (y_pred_count.max() > n_neighbors / 2):
            y_refined[i] = y_pred_count.idxmax()
        else:
            y_refined[i] = y_pred[i]

    y_refined = pd.Categorical(
        values=y_refined.astype('U'),
        categories=natsorted(map(str, y_refined.unique())),
    )
    return y_refined


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SpaCAE', random_seed=100):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def purity_func(cluster, label):
    cluster = np.array(cluster)
    label = np.array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0)) / len(cluster)