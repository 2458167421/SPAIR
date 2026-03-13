import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from .utils import coor_transform, find_similar_index

def get_transform(adata_list, dst_id, src_id_list, target_label_id, threshold=50, max_iterations=200, tolerance=1e-3,
                  seed=0):
    transform_matrix = {}
    if not isinstance(target_label_id, (list, np.ndarray)):
        target_label_id = [target_label_id]  
    for src_id in src_id_list:
        if src_id == dst_id:
            transform_matrix[src_id] = np.eye(3)
            continue
       
        dst_coor_list = []
        src_coor_list = []
        for label_id in target_label_id:
            dst_coor = adata_list[dst_id][label_id == adata_list[dst_id].obs['mclust']].obsm['spatial']
            src_coor = adata_list[src_id][label_id == adata_list[src_id].obs['mclust']].obsm['spatial']
            if 'mclust' not in adata_list[dst_id].obs.columns:
                dst_coor = adata_list[dst_id][label_id == adata_list[dst_id].obs['cluster']].obsm['spatial']
                src_coor = adata_list[src_id][label_id == adata_list[src_id].obs['cluster']].obsm['spatial']
            dst_coor_list.append(dst_coor)
            src_coor_list.append(src_coor)
        
        dst_coor = np.concatenate(dst_coor_list, axis=0)
        src_coor = np.concatenate(src_coor_list, axis=0)
        print(f'>>> INFO: dst slice id: {dst_id}, src slice id: {src_id}')
        print('>>> INFO: dst coordination shape:', dst_coor.shape)
        print('>>> INFO: src coordination shape:', src_coor.shape)


        T = Tricp(src_coor, dst_coor, threshold, max_iterations, tolerance, seed)
        src_coor = coor_transform(src_coor, T).T
        transform_matrix[src_id] = T

        plt.scatter(x=dst_coor[:, 0], y=dst_coor[:, 1], label='dst point cloud')
        plt.scatter(x=src_coor[:, 0], y=src_coor[:, 1], label='src point cloud')
        plt.show()

    return transform_matrix


def calculate_alignment_score(coor_list, label_list, knears=1):
    from collections import Counter

    pair_label_list = []

    for i in range(len(coor_list) - 1):
        sim_index = find_similar_index(
            np.ascontiguousarray(coor_list[i].T[:, :2]).astype(np.float32),
            np.ascontiguousarray(coor_list[i + 1].T[:, :2]).astype(np.float32),
            top_k=knears
        )[1]

        pair_label_list.append([
            Counter(list(label_list[i + 1][sim_index[j]])).most_common(1)[0][0]
            for j in range(len(sim_index))
        ])

    return np.sum([
        np.sum(label_list[i] == pair_label_list[i])
        for i in range(len(coor_list) - 1)
    ]) / np.sum([coor_list[i].shape[1] for i in range(len(coor_list) - 1)])





def Tricp(src, dst, threshold=50, max_iterations=5000, tolerance=1e-3, seed=0):
    
    np.random.seed(seed)
    if dst.shape[0] > src.shape[0]:
        src = src.astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)].astype(np.float32)
    elif dst.shape[0] < src.shape[0]:
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)].astype(np.float32)
        dst = dst.astype(np.float32)

    
    src = np.unique(src, axis=0)
    dst = np.unique(dst, axis=0)

    
    cur_src = np.hstack((src, np.array([1] * src.shape[0]).reshape(-1, 1))).T
    prev_error = 0

   
    best_error = float('inf')
    best_transform = np.eye(3)
    for i in range(max_iterations):
        distances, indices = nearest_neighbor_pl(cur_src[:2, :].T, dst)
        M, _, _ = best_fit_transform_pl(cur_src[:2, :].T, dst, indices)
        cur_src = coor_transform(cur_src[:2, :].T, M)
        print(i)
        mean_error = np.mean(distances)
        
        
        if np.abs(prev_error - mean_error) < tolerance:
            if threshold < mean_error:
                
                num_attempts = 20  
                perturb_strategies = [
                    ('fine', 0.2, 0.5), 
                    ('medium', 0.5, 2),
                    ('coarse', 1.0, 10)
                ]
                
                for strategy in perturb_strategies:
                    for j in range(num_attempts):
                        
                        perturb_factor = min(1.0, mean_error/threshold)
                        rotate_range = perturb_factor * np.pi/6
                        trans_range = perturb_factor * 5
                        print(j)
                        
                        strategy_name, rot_factor, trans_factor = strategy
                        rotate_deg = np.random.uniform(-rotate_range*rot_factor, 
                                                     rotate_range*rot_factor)
                        tx = np.random.uniform(-trans_range*trans_factor,
                                             trans_range*trans_factor)
                        ty = np.random.uniform(-trans_range*trans_factor,
                                             trans_range*trans_factor)
                        
                       
                        scale_factor = np.random.uniform(0.9, 1.1)
                        R = np.array([
                            [scale_factor*np.cos(rotate_deg), -np.sin(rotate_deg), tx],
                            [np.sin(rotate_deg), scale_factor*np.cos(rotate_deg), ty],
                            [0, 0, 1]
                        ])
                        
                        
                        new_src = coor_transform(cur_src[:2, :].T, R)
                        
                        
                        new_distances, _ = nearest_neighbor_pl(new_src, dst)
                        new_mean_error = np.mean(new_distances)
                        
                        
                        if new_mean_error < best_error:
                            best_error = new_mean_error
                            best_transform = R
                
                
                rotate_deg = np.pi * 2 / 3
                rotation_transform = np.array([
                    [np.cos(rotate_deg), np.sin(rotate_deg), 0.5], 
                    [-np.sin(rotate_deg), np.cos(rotate_deg), 0.5], 
                    [0, 0, 1]
                ])
                cur_src = coor_transform(cur_src[:2, :].T, rotation_transform)
                
                
                cur_src = coor_transform(cur_src[:2, :].T, best_transform)
                prev_error = best_error  
                
                
                delta_error = prev_error - best_error
                if delta_error > 0:
                    tx += 0.5 * np.sign(tx)
                    ty += 0.5 * np.sign(ty)
                    rotate_deg += 0.1 * np.sign(rotate_deg)
                
            else:
                break
        prev_error = mean_error

    print(f'>>> INFO: current distance: {mean_error}')
    M, _, _ = best_fit_transform(src, cur_src[:2, :].T)
    return M



def icp(src, dst, threshold=50, max_iterations=5000, tolerance=1e-3, seed=0):

    # sample spots from two sets to the same number
    np.random.seed(seed)
    if (dst.shape[0] > src.shape[0]):
        src = src.astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)].astype(np.float32)
    elif (dst.shape[0] < src.shape[0]):
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)].astype(np.float32)
        dst = dst.astype(np.float32)

    # init
    cur_src = np.hstack((src, np.array([1] * src.shape[0]).reshape(-1, 1))).T
    prev_error = 0

    # train ICP
    for _ in range(max_iterations):
        distances, indices = nearest_neighbor(cur_src[:2, :].T, dst)
        M, _, _ = best_fit_transform(cur_src[:2, :].T, dst[indices])
        cur_src = coor_transform(cur_src[:2, :].T, M)

        mean_error = np.mean(distances)
        if (np.abs(prev_error - mean_error) < tolerance):
            # stuck in local optimum -> rotate src
            if (threshold < mean_error):
                rotate_deg = np.pi * 2 / 3
                cur_src = coor_transform(cur_src[:2, :].T, np.array([
                    [np.cos(rotate_deg), np.sin(rotate_deg), 0.5], 
                    [-np.sin(rotate_deg), np.cos(rotate_deg), 0.5], 
                    [0, 0, 1]
                ]))
            else:
                break
        prev_error = mean_error

    print(f'>>> INFO: current distance: {mean_error}')
    M, _, _ = best_fit_transform(src, cur_src[:2,:].T)
    return M

def nearest_neighbor(src, dst):

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def nearest_neighbor_pl(src, dst):
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    line_distances = []
    for i in range(src.shape[0]):
        p = src[i]
        p1 = dst[indices[i, 0]]
        p2 = dst[indices[i, 1]]
        line_distance = point_to_line_distance(p, p1, p2)
        line_distances.append(line_distance)
    return np.array(line_distances), indices[:, 0]


def point_to_line_distance(p, p1, p2):
    numerator = np.linalg.norm(np.cross(p2 - p1, p1 - p))
    denominator = np.linalg.norm(p2 - p1)
    return numerator / denominator


def best_fit_transform_pl(A, B, indices):
    assert A.shape[0] == len(indices)

   
    m = A.shape[1]
    H = np.zeros((m + 1, m + 1))
    b = np.zeros(m + 1)

    for i in range(A.shape[0]):
        p = A[i]
        p1 = B[indices[i]]
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(B)
        _, two_indices = neigh.kneighbors(p1.reshape(1, -1), return_distance=True)
        p2 = B[two_indices[0, 1]]
        d = p1 - p2
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        e = np.dot(n, p - p1)
        J = np.zeros((1, m + 1))
        J[0, :m] = n
        J[0, m] = 0
        H += np.dot(J.T, J)
        b += -e * J.flatten()

   
    reg = 1e-6 * np.eye(m + 1)
    H += reg

    x = np.linalg.solve(H, b)
    R = np.array([[1, -x[2]], [x[2], 1]])
    t = x[:2]

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t



def best_fit_transform(A, B):
    assert A.shape == B.shape

  
    m = A.shape[1]

   
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

   
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

  
    t = centroid_B.T - np.dot(R, centroid_A.T)

   
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def coor_transform(coor, T):
    coor_homogeneous = np.hstack((coor, np.ones((coor.shape[0], 1))))
    transformed_coor = np.dot(T, coor_homogeneous.T)
    return transformed_coor[:2, :]




'''for i in range(max_iterations):
        distances, indices = nearest_neighbor_pl(cur_src[:2, :].T, dst)
        M, _, _ = best_fit_transform_pl(cur_src[:2, :].T, dst, indices)
        cur_src = coor_transform(cur_src[:2, :].T, M)
        print(i)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            # 陷入局部最优 -> 旋转源点云
            if threshold < mean_error:
                rotate_deg = np.pi * 2 / 5
                cur_src = coor_transform(cur_src[:2, :].T, np.array([
                    [np.cos(rotate_deg), np.sin(rotate_deg), 0.5],
                    [-np.sin(rotate_deg), np.cos(rotate_deg), 0.5],
                    [0, 0, 1]
                ]))
            else:
                break
        prev_error = mean_error'''