
import scanpy as sc
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix, lil_matrix, diags
import time



def get_nearestneighbor(knn, neighbor=1):
    '''For each row of knn, returns the column with the lowest value
    I.e. the nearest neighbor'''
    indices = knn.indices
    indptr = knn.indptr
    data = knn.data
    nn_idx = []
    for i in range(knn.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        nn_idx.append(cols[idx[neighbor-1]])
    return(np.array(nn_idx))


def compute_bw(knn_adj, embedding, n_neighbors=20):
    intersect = knn_adj.dot(knn_adj.T)
    indices = intersect.indices
    indptr = intersect.indptr
    data = intersect.data
    data = data / ((n_neighbors*2) - data)
    bandwidth = []
    for i in range(intersect.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        valssort = rowvals[idx]
        numinset = len(cols)
        if numinset < n_neighbors:
            sys.exit('Fewer than {} cells with Jacard sim > 0'.format(n_neighbors))
        else:
            curval = valssort[n_neighbors]
            for num in range(n_neighbors, numinset):
                if valssort[num] != curval:
                    break
                else:
                    num += 1
            minjacinset = cols[idx][:num]
            if num < n_neighbors:
                print('shouldnt end up here')
                sys.exit(-1)
            else:
                euc_dist = ((embedding[minjacinset,:]-embedding[i,:])**2).sum(axis=1)**.5
                euc_dist_sorted = np.sort(euc_dist)[::-1]
                bandwidth.append(np.mean(euc_dist_sorted[:n_neighbors]))
    return(np.array(bandwidth))


def compute_affinity(dist_to_predict, dist_to_nn, bw):
    affinity = dist_to_predict - dist_to_nn
    affinity[affinity < 0] = 0
    affinity = affinity * -1
    affinity = np.exp(affinity / (bw - dist_to_nn))
    return(affinity)


def dist_from_adj(adjacency, embed, nndist):
   
    dist = lil_matrix(adjacency.shape)
    indices = adjacency.indices
    indptr = adjacency.indptr
    ncells = adjacency.shape[0]

    tic = time.perf_counter()
    for i in range(ncells):
        for j in range(indptr[i], indptr[i+1]):
            col = indices[j]
           
            d = (((embed[i,:] - embed[col,:])**2).sum()**.5) - nndist[i]
            if d == 0:
                dist[i, col] = np.nan
            else:
                dist[i, col] = d
        
        if (i % 2000) == 0:
            toc = time.perf_counter()
            print('%d out of %d %.2f seconds elapsed' % (i, ncells, toc-tic))

    return(csr_matrix(dist))


def select_topK(dist, n_neighbors=20):
    indices = dist.indices
    indptr = dist.indptr
    data = dist.data
    nrows = dist.shape[0]

    final_data = []
    final_col_ind = []

    tic = time.perf_counter()
    for i in range(nrows):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        final_data.append(rowvals[idx[(-1*n_neighbors):]])
        final_col_ind.append(cols[idx[(-1*n_neighbors):]])
            
    final_data = np.concatenate(final_data)
    final_col_ind = np.concatenate(final_col_ind)
    final_row_ind = np.tile(np.arange(nrows), (n_neighbors, 1)).reshape(-1, order='F')
                
    result = csr_matrix((final_data, (final_row_ind, final_col_ind)), shape=(nrows, dist.shape[1]))
    return(result)


class pyWNN():
    
    def __init__(self, adata, reps=['X_pca', 'X_apca', 'X_spca'], n_neighbors=20, npcs=[20, 20, 20], seed=14, distances=None):
    
        self.seed = seed
        np.random.seed(seed)
        
       
        if len(reps) not in [2, 3]:
            sys.exit('WNN目前支持2或3个模态')
        
        self.adata = adata.copy()
        self.n_modalities = len(reps)  
        self.reps = [r + '_norm' for r in reps]
        self.npcs = npcs
        
        
        for i, r in enumerate(reps):
            self.adata.obsm[self.reps[i]] = preprocessing.normalize(adata.obsm[r][:, 0:npcs[i]])

        self.n_neighbors = n_neighbors
        if distances is None:
            print('计算KNN距离矩阵（默认Scanpy实现）')
            
            self.distances = []
            self.distances_200 = []
            for i in range(self.n_modalities):
                key = str(i+1)
                sc.pp.neighbors(
                    self.adata, 
                    n_neighbors=n_neighbors, 
                    n_pcs=npcs[i], 
                    use_rep=self.reps[i], 
                    metric='euclidean', 
                    key_added=key
                )
                self.distances.append(f'{key}_distances')
                
                sc.pp.neighbors(
                    self.adata, 
                    n_neighbors=200, 
                    n_pcs=npcs[i], 
                    use_rep=self.reps[i], 
                    metric='euclidean', 
                    key_added=f'{key}_200'
                )
                self.distances_200.append(f'{key}_200_distances')
            
            self.all_distances = self.distances + self.distances_200
        else:
            self.all_distances = distances
            self.distances = distances[:self.n_modalities]
            self.distances_200 = distances[self.n_modalities:]
            
        
        for d in self.all_distances:
            if type(self.adata.obsp[d]) is not csr_matrix:
                self.adata.obsp[d] = csr_matrix(self.adata.obsp[d])
        
        
        self.NNdist = []  
        self.NNidx = []   
        self.NNadjacency = []  
        self.BWs = []     

        for i in range(self.n_modalities):
          
            nn = get_nearestneighbor(self.adata.obsp[self.distances[i]])
            
            dist_to_nn = ((self.adata.obsm[self.reps[i]] - self.adata.obsm[self.reps[i]][nn, :])**2).sum(axis=1)**.5
            
            nn_adj = (self.adata.obsp[self.distances[i]] > 0).astype(int)
            nn_adj_wdiag = nn_adj.copy()
            nn_adj_wdiag.setdiag(1)  
            
            bw = compute_bw(nn_adj_wdiag, self.adata.obsm[self.reps[i]], n_neighbors=self.n_neighbors)
            
            self.NNidx.append(nn)
            self.NNdist.append(dist_to_nn)
            self.NNadjacency.append(nn_adj)
            self.BWs.append(bw)

        self.weights = []  
        self.WNN = None
        self.WNNdist = None
    
    def compute_weights(self):
       
        affinity_ratios = []
        self.within = []  
        self.cross = []   
        
        for i in range(self.n_modalities):
            
            within_predict = self.NNadjacency[i].dot(self.adata.obsm[self.reps[i]]) / (self.n_neighbors - 1)
            within_predict_dist = ((self.adata.obsm[self.reps[i]] - within_predict)**2).sum(axis=1)**.5
            self.within.append(within_predict_dist)
            
          
            cross_predicts = []
            for j in range(self.n_modalities):
                if j != i:  
                    cross_predicts.append(self.NNadjacency[j].dot(self.adata.obsm[self.reps[i]]) / (self.n_neighbors - 1))
           
            cross_predict = np.mean(cross_predicts, axis=0)
            cross_predict_dist = ((self.adata.obsm[self.reps[i]] - cross_predict)**2).sum(axis=1)**.5
            self.cross.append(cross_predict_dist)
            
            
            within_affinity = compute_affinity(within_predict_dist, self.NNdist[i], self.BWs[i])
            cross_affinity = compute_affinity(cross_predict_dist, self.NNdist[i], self.BWs[i])
            affinity_ratios.append(within_affinity / (cross_affinity + 1e-4))  
        
       
        exp_ratios = np.exp(affinity_ratios)
        sum_exp = np.sum(exp_ratios, axis=0)  
        self.weights = [exp_ratios[i] / sum_exp for i in range(self.n_modalities)]

   
    def compute_wnn(self, adata):
        print('计算模态权重')
        self.compute_weights()
        
       
        print('构建联合邻接矩阵')
        union_adj_mat = None
        for d in self.distances_200:
            if union_adj_mat is None:
                union_adj_mat = self.adata.obsp[d].copy()
            else:
                union_adj_mat += self.adata.obsp[d]
        union_adj_mat = (union_adj_mat > 0).astype(int)  

       
        print('计算联合近邻的加权距离')
        full_dists = []
        for i in range(self.n_modalities):
            dist = dist_from_adj(
                adjacency=union_adj_mat,
                embed=self.adata.obsm[self.reps[i]],
                nndist=self.NNdist[i]
            )
            full_dists.append(dist)
        
       
        weighted_dist = csr_matrix(union_adj_mat.shape)
        for i in range(self.n_modalities):
           
            dist = diags(-1 / (self.BWs[i] - self.NNdist[i]), format='csr').dot(full_dists[i])
            dist.data = np.exp(dist.data)
           
            ind = np.isnan(dist.data)
            dist.data[ind] = 1
            
            dist = diags(self.weights[i]).dot(dist)
            weighted_dist += dist

        
        print('选择Top K近邻')
        self.WNN = select_topK(weighted_dist, n_neighbors=self.n_neighbors)
        
        WNNdist = self.WNN.copy()
        x = (1 - WNNdist.data) / 2
        x[x < 0] = 0
        x[x > 1] = 1
        WNNdist.data = np.sqrt(x)
        self.WNNdist = WNNdist
        

        
        adata.obsp['WNN'] = self.WNN
        adata.obsp['WNN_distance'] = self.WNNdist
        adata.obsm['Weights'] = np.array(self.weights).T  
        
        for i in range(self.n_modalities):
            adata.obsm[self.reps[i]] = self.adata.obsm[self.reps[i]]
       
        adata.uns['WNN'] = {
            'connectivities_key': 'WNN',
            'distances_key': 'WNN_distance',
            'params': {
                'n_neighbors': self.n_neighbors,
                'method': 'WNN',
                'random_state': self.seed,
                'metric': 'euclidean',
                'use_rep': self.reps,
                'n_pcs': self.npcs
            }
        }
        return(adata)