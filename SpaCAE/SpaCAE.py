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

os.environ['R_HOME'] = 'E:\\R\\R-4.2.2'
os.environ['R_USER'] = 'E:\\environment2023\\Anaconda3\\envs\\ae38\\Lib\\site-packages\\rpy2'

plot_color = ["#6D1A9C", "#D1D1D1", "#F56867", "#59BE86", "#FEB915", "#C798EE", "#7495D3", "#15821E",
              "#3A84E6", "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C",
              "#F9BD3F", "#DAB370", "#877F6C", "#268785"]


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

        self.cadata, self.oridata, self.augdata, self.stg = self.load_data()
        self.z_ori, self.z_aug = self.train_SpaCAE()
        self.cluster()
        self.plot_domain()

    def load_data(self):
        print("1/3.Starting loading data...")
        cadata, oridata, augdata, stg = spatial_reconstruction(self.adata,
                                                               alpha=self.alpha,
                                                               n_components=self.zdim,
                                                               )
        stg = torch.Tensor(stg).to(self.device)
        oridata = torch.Tensor(oridata).to(self.device)
        augdata = torch.Tensor(augdata).to(self.device)
        print("The data was successfully loaded and the graph constructed")
        return cadata, oridata, augdata, stg

    def train_SpaCAE(self):
        print("2/3.Starting training...")
        pbar = trange(self.epochs)
        for epoch in pbar:
            z_ori, z_aug, xhat_ori, xhat_aug = self.cgae(self.oridata, self.augdata, self.stg)
            loss = (0.5 * (self.loss_func(z_ori, z_aug) + self.loss_func(xhat_ori, xhat_aug)) +
                    1.5 * (self.loss_func(xhat_ori, self.oridata) + self.loss_func(xhat_aug, self.augdata))).to(
                self.device)
            pbar.set_description('loss={}'.format(loss))
            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            z_ori, z_aug, xhat_ori, xhat_aug = self.cgae(self.oridata, self.augdata, self.stg)
        print("Successful completion of training")
        return z_ori, z_aug

    def cluster(self):
        print("3/3.Starting clusting...")
        X = pd.DataFrame(self.cadata.X.toarray()[:, ], index=self.cadata.obs.index, columns=self.cadata.var.index)
        cells = np.array(X.index)
        cell_reps = pd.DataFrame(self.z_ori.cpu().numpy())
        cell_reps.index = cells
        self.cadata.obsm['SpaCAE'] = cell_reps.loc[self.cadata.obs_names,].values
        '''调用pp.neighbor,和mcluster'''
        sc.pp.neighbors(self.cadata, n_neighbors=10, use_rep='SpaCAE')
        self.cadata = mclust_R(self.cadata, num_cluster=self.num_cluster)

        if self.refine:
            self.cadata.obs['mclust'] = refine_spatial_domains(self.cadata.obs['mclust'],
                                                               coord=self.cadata.obsm['spatial'],
                                                               n_neighbors=10)
        else:
            pass

    def plot_domain(self):
        fig, axs = plt.subplots(figsize=(8, 8))
        sc.pl.spatial(
            self.cadata,
            img_key='hires',
            color='mclust',
            size=1.4,
            title='SpaCAE',
            palette=plot_color,
            legend_loc='right margin',
            show=False,
            ax=axs
        )
        plt.savefig('pre.png', dpi=600)
