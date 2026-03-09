import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import faiss
import random
import anndata
import matplotlib.pyplot as plt
import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 


# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
##

def gen_clust_embed(data, batchs, method, para, seed, device):
    start_time = time.time()
    cluster_embed = []
    
    for i in range(len(set(batchs))):
        x = data[str(i) == batchs]

        if ('kmeans' == method):
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=para, init='k-means++', n_init=20, random_state=seed)
            pred = kmeans.fit_predict(x)
        elif ('louvain' == method):
            import scanpy as sc
            adata = sc.AnnData(x)
            sc.pp.neighbors(adata, random_state=seed, use_rep='X', n_neighbors=15)
            sc.tl.louvain(adata, resolution=para, random_state=seed)
            pred = adata.obs['louvain'].astype(int).to_numpy()

        df = pd.DataFrame(x, index=range(x.shape[0]))
        df.insert(loc=1, column='labels', value=pred)

        cluster_embed.append(nn.Parameter(torch.FloatTensor(np.asarray(df.groupby("labels").mean())).to(device)))

    print(f'>>> INFO: Finish generate precluster embedding({time.time() - start_time:.3f}s)!')
    return cluster_embed


def find_similar_index(source: np.ndarray, target: np.ndarray, top_k: int=1):
    index = faiss.IndexFlatL2(source.shape[1])

    faiss.normalize_L2(np.float32(source))
    faiss.normalize_L2(np.float32(target)) 
    index.add(np.float32(target))

    return index.search(np.float32(source), top_k)


def get_mnn_pairs(data, node_id_map, top_k):
    start_time = time.time()
    edge_list = [[], []]

    def get_node_pairs(node_i, node_j):
        return {
            (node_id_map[i][node[0]], node_id_map[j][node[1]]) 
            for node in np.vstack((node_i, node_j)).T
        }
        
    for i in range(len(node_id_map)):
        for j in range(i+1, len(node_id_map)):

            # get used graph gene expression data
            data_i = data[list(node_id_map[i].values())]
            data_j = data[list(node_id_map[j].values())]

            # find approx. similar node by faiss
            distance_i2j, indices_i2j = find_similar_index(data_i, data_j, top_k)
            distance_j2i, indices_j2i = find_similar_index(data_j, data_i, top_k)

            # convert node id to actual id
            used_i2j_list = distance_i2j.reshape(-1) <= distance_j2i.reshape(-1).max() * 0.75
            i2j_pairs = get_node_pairs(
                np.array([np.arange(len(node_id_map[i]))]*top_k).T.reshape(-1)[used_i2j_list], 
                indices_i2j.reshape(-1)[used_i2j_list]
            )
            used_j2i_list = distance_j2i.reshape(-1) <= distance_j2i.reshape(-1).max() * 0.75
            j2i_pairs = get_node_pairs(
                indices_j2i.reshape(-1)[used_j2i_list],
                np.array([np.arange(len(node_id_map[j]))]*top_k).T.reshape(-1)[used_j2i_list]
            )

            # find mnn paris and concat with other edges
            mnn_pairs = np.array(list(i2j_pairs & j2i_pairs)).T
            edge_list = np.array([
                np.concatenate((edge_list[0], mnn_pairs[0])),
                np.concatenate((edge_list[1], mnn_pairs[1]))
            ])
    
    print(f'>>> INFO: Finish finding mmn pairs, find {edge_list.shape[1]} mnn node pairs({time.time() - start_time:.3f}s)!')
    return edge_list






def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     


# transform spot coordination
def lsi(
        adata: anndata.AnnData, n_components: int = 50,#20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   


def coor_transform(coor, M):
    if coor.shape[0] == 2:
        coor = np.vstack((coor, np.ones((1, coor.shape[1]))))
    elif coor.shape[0] == 3:
        coor = np.vstack((coor, np.ones((1, coor.shape[1]))))
    return np.dot(M, np.hstack((coor, np.array([1] * coor.shape[0]).reshape(-1, 1))).T)


def coor_transform0(coor, M):
    return np.dot(M, np.hstack((coor, np.array([1] * coor.shape[0]).reshape(-1, 1))).T)


# set the palette for each label
def get_palette(label_list, opacity=1.0, use_cmap_func=None):
    palette = {}
    max_label = np.array(list(label_list)).shape[0]
    map_label_id = {id: label_name for id, label_name in enumerate(list(label_list))}
    
    # set cmp function
    import matplotlib
    if (not isinstance(use_cmap_func, matplotlib.colors.Colormap)):
        if (max_label < 10):
            use_cmap_func = plt.cm.tab10
        elif (max_label < 20):
            use_cmap_func = plt.cm.tab20
        else:
            use_cmap_func = plt.cm.gist_ncar
    assert(max_label <= use_cmap_func.N), '>>> ERROR: The cmp function has fewer colors than the label count'

    for label_id in range(max_label):
        color = use_cmap_func(int(label_id) / (max_label + 1))
        palette[map_label_id[label_id]] = (color[0], color[1], color[2], opacity)

    return palette


def plotting1(coor_list, label_list,save_path=None, palette=None, norm_coor=False, spot_size=1, dims='2d', line_list=None, title=None):

    # if input 2d coor, convert to 3d
    if (2 == coor_list[0].shape[0]):
        new_coor_list = [
            np.vstack([coor, np.ones((1, coor.shape[1]))])
            for coor in coor_list
        ]
        coor_list = new_coor_list

    fig = plt.figure()
    if ('3d' == dims):
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    if (None == palette):
        palette = get_palette(np.unique(np.hstack(label_list)))

    if (norm_coor):
        for i, coor in enumerate(coor_list):
            xs, ys, _ = np.array(coor)
            coor_list[i][0] = xs - xs.min()
            coor_list[i][1] = ys - ys.min()

    for i, coor in enumerate(coor_list):
        xs, ys, _ = np.array(coor)

        label_color = [palette[label] for label in label_list[i]]
        if ('3d' == dims):
            ax.scatter(xs=xs, ys=ys, s=spot_size, zs=i, c=label_color, label=f'slice_{i}')
        else:
            ax.scatter(x=xs, y=ys, s=spot_size, c=label_color, label=f'slice_{i}')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plotting(coor_list, label_list, save_path=None, palette=None, norm_coor=False, 
             spot_size=1, dims='2d', line_list=None, title=None):
    
    # if input 2d coor, convert to 3d
    if (2 == coor_list[0].shape[0]):
        new_coor_list = [
            np.vstack([coor, np.ones((1, coor.shape[1]))])
            for coor in coor_list
        ]
        coor_list = new_coor_list

    fig = plt.figure()
    if ('3d' == dims):
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    
    # 使用Tableau颜色替代原来的调色板
    if palette is None:
        colors = list(mcolors.TABLEAU_COLORS.values())
        unique_labels = np.unique(np.hstack(label_list))
        palette = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    if norm_coor:
        for i, coor in enumerate(coor_list):
            xs, ys, _ = np.array(coor)
            coor_list[i][0] = xs - xs.min()
            coor_list[i][1] = ys - ys.min()

    for i, coor in enumerate(coor_list):
        xs, ys, _ = np.array(coor)
        label_color = [palette[label] for label in label_list[i]]
        
        if '3d' == dims:
            ax.scatter(xs=xs, ys=ys, s=spot_size, zs=i, c=label_color, label=f'slice_{i}')
        else:
            ax.scatter(x=xs, y=ys, s=spot_size, c=label_color, label=f'slice_{i}')

    if isinstance(line_list, np.ndarray):
        layer = 0
        for lines in line_list:
            xs_0, ys_0, _ = np.array(coor_list[layer])
            xs_1, ys_1, _ = np.array(coor_list[layer+1])

            for line in lines:
                src, dst = line
                ax.plot([xs_0[src], xs_1[dst]], [ys_0[src], ys_1[dst]], [layer, layer+1], 
                        c='gray', linestyle='-', linewidth=1)
            layer += 1

    ax.set_xticks([])
    ax.set_yticks([])
    if '3d' == dims:
        ax.set_zticks([])
    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)


    if (isinstance(line_list, np.ndarray)):
        layer = 0
        for lines in line_list:
            xs_0, ys_0, _ = np.array(coor_list[layer])
            xs_1, ys_1, _ = np.array(coor_list[layer+1])

            for line in lines:
                src, dst = line
                ax.plot([xs_0[src], xs_1[dst]], [ys_0[src], ys_1[dst]], [layer, layer+1], c='gray', linestyle='-', linewidth=0.1)
            
            layer += 1

    ax.set_xticks([])
    ax.set_yticks([])
    if ('3d' == dims):
        ax.set_zticks([])
    if (title):
        plt.title(title)

    if (save_path):
        plt.savefig(save_path)
