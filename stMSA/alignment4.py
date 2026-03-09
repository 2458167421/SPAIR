import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .utils import coor_transform, find_similar_index

# ======================== 主要修改部分 ========================
def get_transform(adata_list, dst_id, src_id_list, target_label_id, 
                 threshold=50, max_iterations=50000, tolerance=1e-3, 
                 seed=0, overlap_ratio=0.8):  # 新增overlap_ratio参数
    
    transform_matrix = {}

    for src_id in src_id_list:
        if src_id == dst_id:
            transform_matrix[src_id] = np.eye(3)
            continue

        dst_coor = adata_list[dst_id][target_label_id == adata_list[dst_id].obs['mclust']].obsm['spatial']
        src_coor = adata_list[src_id][target_label_id == adata_list[src_id].obs['mclust']].obsm['spatial']

        print(f'>>> INFO: dst slice id: {dst_id}, src slice id: {src_id}')
        print('>>> INFO: dst coordination shape:', dst_coor.shape)
        print('>>> INFO: src coordination shape:', src_coor.shape)

        # 传递overlap_ratio参数到icp
        T = icp(src_coor, dst_coor, threshold, max_iterations, tolerance, seed, overlap_ratio)
        src_coor = coor_transform(src_coor, T).T
        transform_matrix[src_id] = T

        plt.scatter(x=dst_coor[:, 0], y=dst_coor[:, 1], label='dst point cloud')
        plt.scatter(x=src_coor[:, 0], y=src_coor[:, 1], label='src point cloud')
        plt.title(f'Overlap Ratio: {overlap_ratio}')
        plt.show()

    return transform_matrix

def icp(src, dst, threshold=50, max_iterations=50000, tolerance=1e-3, 
       seed=0, overlap_ratio=0.8):  # 新增overlap_ratio参数
    
    np.random.seed(seed)
    # 保持采样逻辑不变
    if dst.shape[0] > src.shape[0]:
        src = src.astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)].astype(np.float32)
    elif dst.shape[0] < src.shape[0]:
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)].astype(np.float32)
        dst = dst.astype(np.float32)

    # 初始化
    cur_src = np.hstack((src, np.ones((src.shape[0], 1)))).T
    prev_error = 0
    best_error = np.inf
    best_T = np.eye(3)  # 保存最优变换矩阵

    for iter in range(max_iterations):
        # 1. 最近邻搜索
        distances, indices = nearest_neighbor(cur_src[:2, :].T, dst)
        
        # 2. 裁剪步骤：保留距离最小的前overlap_ratio比例点对
        n_points = len(distances)
        k = max(int(n_points * overlap_ratio), 10)  # 确保至少3个点
        
        sorted_indices = np.argsort(distances)
        selected_src_indices = sorted_indices[:k]
        selected_dst_indices = indices[selected_src_indices]
        
        # 3. 计算当前最优变换
        M, R, t = best_fit_transform(
            cur_src[:2, :].T[selected_src_indices],
            dst[selected_dst_indices]
        )
        
        # 4. 应用变换并更新当前点云
        cur_src_transformed = coor_transform(cur_src[:2, :].T, M)
        
        # 5. 计算误差（仅基于裁剪点）
        current_error = np.mean(distances[selected_src_indices])
        
        # 保存最优结果
        if current_error < best_error:
            best_error = current_error
            best_T = M
        
        # 6. 收敛判断（基于裁剪后的误差）
        if np.abs(prev_error - current_error) < tolerance:
            if threshold < current_error:
                # 添加随机扰动跳出局部最优
                rotate_deg = np.random.uniform(0, np.pi/2)
                translate = np.random.uniform(-10, 10, size=2)
                M = np.array([
                    [np.cos(rotate_deg), -np.sin(rotate_deg), translate[0]],
                    [np.sin(rotate_deg), np.cos(rotate_deg), translate[1]],
                    [0, 0, 1]
                ])
                cur_src = coor_transform(cur_src[:2, :].T, M)
            else:
                break
        prev_error = current_error

    print(f'>>> Final trimmed error: {best_error:.4f}')
    return best_T
# ======================== 修改结束 ========================

# 以下为原有辅助函数（保持不变）
def calculate_alignment_score(coor_list, label_list, knears=1):
    from collections import Counter

    pair_label_list = []
    for i in range(len(coor_list)-1):
        sim_index = find_similar_index(
            np.ascontiguousarray(coor_list[i].T[:, :2]).astype(np.float32), 
            np.ascontiguousarray(coor_list[i+1].T[:, :2]).astype(np.float32),
            top_k=knears
        )[1]
        pair_label_list.append([
            Counter(list(label_list[i+1][sim_index[j]])).most_common(1)[0][0]
            for j in range(len(sim_index))
        ])
    return np.sum([
        np.sum(label_list[i] == pair_label_list[i])
        for i in range(len(coor_list)-1)
    ]) / np.sum([coor_list[i].shape[1] for i in range(len(coor_list)-1)])

def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

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
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t