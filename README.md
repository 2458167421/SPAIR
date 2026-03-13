# SPAIR
SPAIR: Spatial Multi-omics Integration and Alignment via Pairwise Contrastive Learning
![image](https://github.com/2458167421/SPAIR/edit/main/overview.png)
## Overview
SPAIR is a computational framework designed for multi-batch spatial multi-omics integration, enabling unified spatial domain identification, 3D tissue reconstruction, and cross-omics inference. As illustrated in Figure~\ref{fig:overview}A, the pipeline begins by processing preprocessed molecular data and spatial graphs through the SOI Module to extract modality-specific embeddings. These embeddings are fused via the WNN module to produce an integrated multi-omics representation for downstream tasks, including spatial domain identification and cross-slice alignment (Figure~\ref{fig:overview}B).

The SOI Module (Figure~\ref{fig:overview}C) employs graph attention networks to capture local microenvironment features, integrating both inner-batch and cross-batch contrastive learning. Feature reconstruction, adjacency reconstruction, and domain distribution tuning further refine the embeddings. The WNN module (Figure~\ref{fig:overview}D) adaptively integrates heterogeneous modal embeddings by computing cross-modality affinity ratios and applying weighted fusion. Finally, for spatial registration across slices, SPAIR uses a TrICP alignment strategy (Figure~\ref{fig:overview}E) that excludes outlier correspondences to improve both accuracy and robustness.

In summary, SPAIR provides a flexible and efficient framework for mosaic integration of multi-batch spatial multi-omics data. It supports core downstream analyses—including multi-modal integration, local and global slice alignment, 3D tissue reconstruction, and cross-omics prediction—offering a robust computational tool for in-depth exploration of spatial biological mechanisms.
## Software dependencies
scanpy==1.9.3  
squidpy==1.3.0  
pytorch==1.13.0(cuda==11.6)   
torch_geometric==2.3.1(cuda==11.6)  
R==3.5.1  
mclust==5.4.10
## Set up
First clone the repository.
```python
git clone https://github.com/Zhenpm/SpatialMOSI.git
cd SpatialMOSI-main
``` 
Then, we suggest creating a new environment：
```python
conda create -n spatialmosi python=3.10 
conda activate spatialmosi
```
Additionally, install the packages required:
```python
pip install -r requiements.txt
```
