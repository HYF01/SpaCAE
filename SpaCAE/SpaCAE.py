import numpy as np
import torch
import random
import os
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from model import CGAE
from torch.optim import Adam
import torch.nn as nn
from utils import mclust_R, spatial_reconstruction, refine_spatial_domains
import matplotlib.pyplot as plt
from tqdm import trange
from anndata import AnnData
from typing import Optional

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix

plot_color = ["#6D1A9C", "#D1D1D1", "#F56867", "#59BE86", "#FEB915", "#C798EE", "#7495D3", "#15821E",
              "#3A84E6", "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C",
              "#F9BD3F", "#DAB370", "#877F6C", "#268785"]
allcolour = ["#40E0D0", "#98FB98", "#3A84E6", "#FC8002", "#59BE86", "#FFFF00", "#D1D1D1"]


class SpaCAE:
    def __init__(self,
                 adata,
                 num_cluster,
                 alpha=0.5,
                 input_dim=14,  # 15
                 z_dim=14,
                 output_dim=14,
                 spacae_seed=100,
                 device='cuda',
                 lr=1e-4,
                 n_epochs=40000,
                 copy: bool = True,
                 refine: bool = True,
                 ):
        super(SpaCAE, self).__init__()

        # Set seed
        random.seed(spacae_seed)
        torch.manual_seed(spacae_seed)
        np.random.seed(spacae_seed)
        self.device = device

        self.cgae = CGAE(input_dim, z_dim, output_dim).to(self.device)
        self.lr = lr
        self.epochs = n_epochs
        self.loss_func = nn.MSELoss()
        self.optimizer = Adam(self.cgae.parameters(), lr=self.lr)

        self.adata = adata.copy() if copy else adata
        self.num_cluster = num_cluster
        self.alpha = alpha
        self.indim = input_dim
        self.zdim = z_dim
        self.outdim = output_dim
        self.refine = refine

    def load_data(self):
        print("1/3.Starting loading data...")
        cadata, oridata, augdata, stg = spatial_reconstruction(self.adata,
                                                               alpha=self.alpha,
                                                               n_components=self.zdim,
                                                               )
        print("The data was successfully loaded and the graph constructed")
        return cadata, oridata, augdata, stg

    def train_SpaCAE(self, oridata, augdata, stg):
        print("2/3.Starting training...")
        stg = torch.Tensor(stg).to(self.device)
        oridata = torch.Tensor(oridata).to(self.device)
        augdata = torch.Tensor(augdata).to(self.device)
        pbar = trange(self.epochs)
        for epoch in pbar:
            z_ori, z_aug, xhat_ori, xhat_aug = self.cgae(oridata, augdata, stg)
            loss = (0.5 * (self.loss_func(z_ori, z_aug) + self.loss_func(xhat_ori, xhat_aug)) +
                    1.5 * (self.loss_func(xhat_ori, oridata) + self.loss_func(xhat_aug, augdata))).to(
                self.device)
            pbar.set_description('loss={}'.format(loss))
            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            z_ori, z_aug, xhat_ori, xhat_aug = self.cgae(oridata, augdata, stg)
        print("Successful completion of training")
        return z_ori, z_aug, xhat_ori, xhat_aug

    def cluster(self, c_adata, c_zori):
        print("3/3.Starting clusting...")
        X = pd.DataFrame(c_adata.X.toarray()[:, ], index=c_adata.obs.index, columns=c_adata.var.index)
        cells = np.array(X.index)
        cell_reps = pd.DataFrame(c_zori.cpu().numpy())
        cell_reps.index = cells
        c_adata.obsm['SpaCAE'] = cell_reps.loc[c_adata.obs_names,].values
        '''调用pp.neighbor,和mcluster'''
        sc.pp.neighbors(c_adata, n_neighbors=10, use_rep='SpaCAE')
        c_adata = mclust_R(c_adata, num_cluster=self.num_cluster)

        if self.refine:
            c_adata.obs['spacae_clust'] = refine_spatial_domains(c_adata.obs['mclust'],
                                                                 coord=c_adata.obsm['spatial'],
                                                                 n_neighbors=10)
        else:
            pass
        return c_adata

    def plot_domain(self, p_adata):
        fig, axs = plt.subplots(figsize=(8, 8))
        sc.pl.spatial(
            p_adata,
            img_key='hires',
            color='spacae_clust',
            size=1.4,
            title='SpaCAE',
            palette=plot_color,
            legend_loc='right margin',
            show=False,
            ax=axs
        )
        plt.savefig('pre.png', dpi=600)

    def spatial_denoising(self,
                          adata_ori: AnnData,
                          alpha: float = 8,
                          n_neighbors: int = 10,
                          n_pcs: int = 15,
                          use_highly_variable: Optional[bool] = True,
                          normalize_total: bool = True,
                          copy: bool = True,
                          ):
        adata = adata_ori.copy() if copy else adata_ori
        adata.layers['counts'] = adata.X

        sc.pp.normalize_total(adata) if normalize_total else None
        sc.pp.log1p(adata)
        adata.layers['log1p-ori'] = adata.X

        # hvg = list(adata.var['highly_variable'][adata.var['highly_variable'].values].index)
        exmatrix_ori = adata.to_df(layer='log1p-ori')

        sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)
        coord = adata.obsm['spatial']
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
        nbrs = neigh.kneighbors_graph(coord)
        dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
        conns = nbrs.T.toarray() * dists

        '''X_rec = X + alpha*X*conns '''
        X = adata.X.toarray() if issparse(adata.X) else adata.X
        X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X

        adata.X = csr_matrix(X_rec)
        adata.layers['log1p-aug'] = adata.X

        exmatrix_aug = adata.to_df(layer='log1p-aug')

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

        return adata if copy else None, exmatrix_ori, exmatrix_aug
