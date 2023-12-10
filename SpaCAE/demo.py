import scanpy as sc
from SpaCAE import SpaCAE

adata = sc.read_visium('151675')
adata.var_names_make_unique()
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')

num_cluster = 7

SpaCAE(adata=adata, num_cluster=num_cluster)
