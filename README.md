Spatially contrastive variational autoencoder for deciphering tissue heterogeneity from spatially resolved transcriptomics
======
## Overview
![Image text](https://github.com/HYF01/SpaCAE/blob/main/overview.png)
SpaCAE (SPAtially Contrastive variational AutoEncoder) is a spatially contrastive variational autoencoder framework designed for spatial domains identification and highly sparse SRT data denoising.

SpaCAE contrasts transcriptomic signals of each spot and its spatial neighbors to achieve fine-grained tissue structures detection. By employing a graph embedding variational autoencoder and incorporating a deep contrastive strategy, SpaCAE achieves a balance between spatial local information and global information of expression, enabling effective learning of representations with spatial constraints. Particularly, SpaCAE provides a graph deconvolutional decoder to address the smoothing effect of local spatial structure on expression’s self-supervised learning, an aspect often overlooked by current graph neural networks. 

The latent representation and reconstructed expression can be applied to downstream analyses, including identifying spatial domains by Mclust algorithms, denoising the SRT profiles with reconstructed expression and visualization. 
## Requirements
Hardware resources used in this project.<br>
+ GPU NVIDIA GeForce RTX 3060<br>
+ CUDA Version 11.7<br><br>

You'll need to install the following packages in order to run the codes.<br>
+ python==3.8<br>
+ torch==2.0.0<br>
+ numpy==1.21.6<br>
+ pandas==2.0.0<br>
+ scanpy==1.9.3<br>
+ anndata==0.8.0<br>
+ scipy==1.10.1<br>
+ scikit-learn==1.2.2<br>
+ tqdm==4.65.0<br>
+ matplotlib==3.7.1<br>
+ R==4.2.2<br>
+ rpy2==3.5.10<br>

## Tutorial
For the step-by-step tutorial, please refer to: 
<br>
https://github.com/HYF01/SpaCAE/blob/main/SpaCAE/tutorial.md
<br>
A Jupyter Notebook of the tutorial is accessible from : 
<br>
https://github.com/HYF01/SpaCAE/blob/main/SpaCAE/tutorial.ipynb
<br>

