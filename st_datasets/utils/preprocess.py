import pandas as pd
import scanpy as sc
import numpy as np
import igraph as ig
from sklearn.neighbors import NearestNeighbors
import time
from scipy.spatial.distance import pdist, squareform

import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


import anndata
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_graph1(adata, radius=20):
    """
    构建单细胞数据（scRNA）的图结构并返回边列表

    参数:
    adata (anndata.AnnData): 包含单细胞数据的AnnData对象
    radius (float): 用于查找邻居的半径

    返回:
    edges (numpy.ndarray): 边列表，形状为 (num_edges, 2)，每行表示连接的两个细胞的索引
    """
    num_cells = adata.n_obs
    print(f"细胞数量: {num_cells}")

    # 使用 NearestNeighbors 以指定半径找到每个细胞的邻居，使用欧几里得距离
    nbrs = NearestNeighbors(radius=radius, metric='euclidean').fit(adata.X)
    _, indices = nbrs.radius_neighbors(adata.X)

    # 生成边列表
    edge_list = []
    for i, sublist in enumerate(indices):
        for j in sublist:
            if i != j:  # 跳过自身连接
                edge_list.append((i, j))

    edge_list = np.array(edge_list)
    print(f">>> INFO: 生成 {edge_list.shape[0]} 条边，每个细胞平均有 {(edge_list.shape[0] / adata.shape[0]):.3f} 条边。")

    return edge_list


def build_graph2(adata, radius=None, knears=None, distance_metrics='l2', use_repo='X_lsi'):
    start = time.time()

    if (isinstance(adata, np.ndarray)):
        coor = pd.DataFrame(adata)
        print(1)
    elif ('X' == use_repo):
        coor = pd.DataFrame(adata.X.todense())
        print(2)
    else:
        coor = pd.DataFrame(adata.obsm[use_repo])
        coor.index = adata.obs.index
        coor.columns = ['row', 'col']
        print(3)

    if (radius):
        nbrs = NearestNeighbors(radius=radius, metric=distance_metrics).fit(coor)
        _, indices = nbrs.radius_neighbors(coor, return_distance=True)
    else:
        nbrs = NearestNeighbors(n_neighbors=knears+1, metric=distance_metrics).fit(coor)
        _, indices = nbrs.kneighbors(coor)

    edge_list = np.array([[i, j] for i, sublist in enumerate(indices) for j in sublist])
    print (edge_list)
    print(f">>> INFO: Generate {edge_list.shape[0]} edges, {(edge_list.shape[0] / adata.shape[0]) - 1:.3f} edges per spot.({time.time() - start:.3f}s)")

    return edge_list




def build_graph(adata, radius=None, knears=None, distance_metrics='l2', use_repo='spatial'):
    start = time.time()

    if (isinstance(adata, np.ndarray)):
        coor = pd.DataFrame(adata)
        print(1)
    elif ('X' == use_repo):
        coor = pd.DataFrame(adata.X.todense())
        print(2)
    else:
        coor = pd.DataFrame(adata.obsm[use_repo])
        coor.index = adata.obs.index
        coor.columns = ['row', 'col']
        print(3)

    if (radius):
        nbrs = NearestNeighbors(radius=radius, metric=distance_metrics).fit(coor)
        _, indices = nbrs.radius_neighbors(coor, return_distance=True)
    else:
        nbrs = NearestNeighbors(n_neighbors=knears+1, metric=distance_metrics).fit(coor)
        _, indices = nbrs.kneighbors(coor)

    edge_list = np.array([[i, j] for i, sublist in enumerate(indices) for j in sublist])
    print (edge_list)
    print(f">>> INFO: Generate {edge_list.shape[0]} edges, {(edge_list.shape[0] / adata.shape[0]) - 1:.3f} edges per spot.({time.time() - start:.3f}s)")

    return edge_list


def convert_edge_to_adj(edge_list, spot_num=None, dense=True):
    g = ig.Graph(n=spot_num, edges=edge_list.T)
    
    if (dense):
        return np.asarray(g.get_adjacency_sparse().todense())
    else:
        return g.get_adjacency_sparse()


def convert_adj_to_edge(adj: np.array):
    return np.array(np.nonzero(adj))


def get_not_adjacency_pair(adj: np.array):
    return np.array(np.nonzero(adj == 0))


def add_self_loop(edge_list, spot_num=None):
    if (spot_num):
        self_loop = np.arange(0, spot_num)
    else:
        self_loop = np.arange(0, np.max(edge_list))
    no_loop_list = delete_self_loop(edge_list)
    result_list = np.array([
        np.concatenate(no_loop_list[0], self_loop),
        np.concatenate(no_loop_list[1], self_loop)
    ])

    return result_list


def delete_self_loop(edge_list):
    non_duplicate_element = edge_list[0, :] != edge_list[1, :]
    filtered_array = np.array([
        edge_list[0][non_duplicate_element], 
        edge_list[1][non_duplicate_element]
    ])

    return filtered_array


def concat_adjacency_matrix(adata_list, edge_list, return_type=None):
    edges = edge_list[0]
    spot_count = adata_list[0].X.shape[0]

    for i in range(1, len(adata_list)):
        edge_list[i] = spot_count + edge_list[i]
        edges = np.vstack((edges, edge_list[i]))
        spot_count += adata_list[i].X.shape[0]

    if ('adj' == return_type):
        return convert_edge_to_adj(edges.T)
    else:
        return edges.T
    

def conv_to_one_hot(X):
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(sparse=False)
    one_hot = enc.fit_transform(X.reshape(-1, 1)) 
    return one_hot


def k_hop_adj(edge_list, adj, k, spot_num=None):
    if (edge_list):
        adj = convert_edge_to_adj(edge_list, spot_num)

    return adj ** k
