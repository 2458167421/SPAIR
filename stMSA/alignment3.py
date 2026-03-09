import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .utils import coor_transform, find_similar_index

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
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t

'''
def get_transform1(adata_list, dst_id, src_id_list, target_label_id, threshold=50, max_iterations=5000, tolerance=1e-3, seed=0):
    
    transform_matrix = {}

    if not isinstance(target_label_id, (list, np.ndarray)):
        target_label_id = [target_label_id]  # 如果输入不是列表或数组，将其转换为列表
    for src_id in src_id_list:
        if (src_id == dst_id):
            transform_matrix[src_id] = np.eye(3)
            continue
        # 合并多个 landmark 域
        dst_coor_list = []
        src_coor_list = []
        for label_id in target_label_id:
            dst_coor = adata_list[dst_id][label_id == adata_list[dst_id].obs['mclust']].obsm['spatial']
            src_coor = adata_list[src_id][label_id == adata_list[src_id].obs['mclust']].obsm['spatial']
            dst_coor_list.append(dst_coor)
            src_coor_list.append(src_coor)
        # 合并多个 landmark 域的坐标
        dst_coor = np.concatenate(dst_coor_list, axis=0)
        src_coor = np.concatenate(src_coor_list, axis=0)

        print(f'>>> INFO: dst slice id: {dst_id}, src slice id: {src_id}')
        print('>>> INFO: dst coordination shape:', dst_coor.shape)
        print('>>> INFO: src coordination shape:', src_coor.shape)

        T = icp(src_coor, dst_coor, threshold, max_iterations, tolerance, seed)
        src_coor = coor_transform(src_coor, T).T
        transform_matrix[src_id] = T

        plt.scatter(x=dst_coor[:, 0], y=dst_coor[:, 1], label='dst point cloud')
        plt.scatter(x=src_coor[:, 0], y=src_coor[:, 1], label='src point cloud')
        plt.show()

    return transform_matrix

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


'''



def get_transform(adata_list, dst_id, src_id_list, target_label_id, 
                 threshold=50, max_iterations=50000, tolerance=1e-3, 
                  seed=0):  # 新增trim_ratio参数 trim_ratio=0.8,
    transform_matrix = {}
    if not isinstance(target_label_id, (list, np.ndarray)):
        target_label_id = [target_label_id]  # 如果输入不是列表或数组，将其转换为列表
    for src_id in src_id_list:
        if (src_id == dst_id):
            transform_matrix[src_id] = np.eye(3)
            continue
        # 合并多个 landmark 域
        dst_coor_list = []
        src_coor_list = []
        for label_id in target_label_id:
            dst_coor = adata_list[dst_id][label_id == adata_list[dst_id].obs['mclust']].obsm['spatial']
            src_coor = adata_list[src_id][label_id == adata_list[src_id].obs['mclust']].obsm['spatial']
            dst_coor_list.append(dst_coor)
            src_coor_list.append(src_coor)
        # 合并多个 landmark 域的坐标
        dst_coor = np.concatenate(dst_coor_list, axis=0)
        src_coor = np.concatenate(src_coor_list, axis=0)
        print(f'>>> INFO: dst slice id: {dst_id}, src slice id: {src_id}')
        print('>>> INFO: dst coordination shape:', dst_coor.shape)
        print('>>> INFO: src coordination shape:', src_coor.shape)
        # 使用 TRICP 算法
        T = tricp(src_coor, dst_coor, threshold, max_iterations, 
                tolerance,  seed)  # 改为tricp函数 trim_ratio,
        src_coor = coor_transform(src_coor, T).T
        transform_matrix[src_id] = T
        plt.scatter(x=dst_coor[:, 0], y=dst_coor[:, 1], label='dst point cloud')
        plt.scatter(x=src_coor[:, 0], y=src_coor[:, 1], label='src point cloud')
        plt.show()
    return transform_matrix


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



def tricp(src, dst, threshold=50, max_iterations=5000, 
        tolerance=1e-3, trim_ratio=0.5, seed=0):
    
    # 点云采样优化
    np.random.seed(seed)
    if dst.shape[0] != src.shape[0]:
        n_samples = min(dst.shape[0], src.shape[0])
        src = src[np.random.choice(src.shape[0], n_samples, replace=False)].astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], n_samples, replace=False)].astype(np.float32)
    
    # 初始化参数
    cur_src = np.hstack((src, np.ones((src.shape[0], 1)))).T  # shape (3, N)
    prev_error = np.inf
    best_T = np.eye(3)
    best_error = np.inf
    
    # 动态trimming机制
    dynamic_trim_ratio = trim_ratio  # 初始trim比例

    for _ in range(max_iterations):
        # 1. 最近邻搜索
        distances, indices = nearest_neighbor(cur_src[:2, :].T, dst)
        
        # 2. 动态剪枝策略
        sorted_indices = np.argsort(distances)
        keep_num = max(int(len(distances)*dynamic_trim_ratio), 10)
        keep_indices = sorted_indices[:keep_num]
        
        # 3. 计算最优变换（仅使用保留点）
        M, R, t = best_fit_transform(
            cur_src[:2, :].T[keep_indices], 
            dst[indices][keep_indices]
        )
        
        # 4. 应用变换并更新误差（修正维度对齐）
        transformed_src = (M[:2, :2] @ cur_src[:2, :]) + M[:2, 2:]  # shape (2, N)
        current_error = np.mean(distances[keep_indices])
        
        # 5. 自适应trim调整
        if current_error < best_error:
            best_error = current_error
            best_T = M
            dynamic_trim_ratio = min(trim_ratio + 0.1, 0.95)
        else:
            dynamic_trim_ratio = max(trim_ratio - 0.1, 0.5)
        
        # 6. 收敛判断
        if np.abs(prev_error - current_error) < tolerance:
            if best_error < threshold:
                break
            # 局部最优处理策略优化
            rotate_deg = np.random.choice([np.pi/2, np.pi, 3*np.pi/2])
            M = np.array([
                [np.cos(rotate_deg), -np.sin(rotate_deg), np.random.uniform(-1,1)], 
                [np.sin(rotate_deg), np.cos(rotate_deg), np.random.uniform(-1,1)], 
                [0, 0, 1]
            ])
        
        prev_error = current_error
        cur_src[:2, :] = transformed_src  # 直接更新坐标矩阵

    print(f'>>> INFO: Final trimmed distance: {best_error:.4f}')
    return best_T


def tricp2(src, dst, threshold=50, max_iterations=50000, tolerance=1e-7, seed=0):
    """
    TRICP 算法的实现。

    参数:
    src (numpy.ndarray): 源点集，形状为 (n, m)，其中 n 是点的数量，m 是维度（例如 2D 或 3D）。
    dst (numpy.ndarray): 目标点集，形状为 (p, m)。
    threshold (float): 距离阈值，用于修剪不匹配的点对。
    max_iterations (int): 最大迭代次数。
    tolerance (float): 收敛阈值，当误差变化小于此值时停止迭代。
    seed (int): 随机数种子，用于点集采样。

    返回:
    numpy.ndarray: 最终的变换矩阵，形状为 (m+1, m+1) 的齐次变换矩阵。
    """
    np.random.seed(seed)
    # 将 NaN 转换为 0
    src = np.nan_to_num(src)
    dst = np.nan_to_num(dst)
    # 采样到相同数量的点
    if dst.shape[0] > src.shape[0]:
        src = src.astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)].astype(np.float32)
    elif dst.shape[0] < src.shape[0]:
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)].astype(np.float32)
        dst = dst.astype(np.float32)
    
    
    cur_src = np.hstack((src, np.ones((src.shape[0], 1)))).T  # 转换为齐次坐标，形状为 (m+1, n)
    #cur_src = np.hstack((src, np.array([1] * src.shape[0]).reshape(-1, 1))).T
    prev_error = 0  # 用于存储上一次的误差
    for i in range(max_iterations):
        # 找到最近邻
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(cur_src[:-1, :].T, return_distance=True)  # 不包括齐次坐标的最后一维
        # 修剪不匹配的点对
        valid_indices = np.where(distances.ravel() < threshold)[0]
        valid_src = cur_src[:-1, valid_indices].T  # 有效的源点集
        valid_dst = dst[indices.ravel()[valid_indices]]  # 对应的目标点集
        # 计算最佳变换矩阵
        T, _, _ = best_fit_transform(valid_src, valid_dst)
        # 应用变换矩阵更新源点集
        cur_src[:-1, :] = np.dot(T[:-1, :-1], cur_src[:-1, :]) + T[:-1, -1].reshape(-1, 1)
        # 计算平均误差
        mean_error = np.mean(distances[valid_indices])
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    print(f'>>> INFO: current distance: {mean_error}')
    # 最终的变换矩阵计算
    M, _, _ = best_fit_transform(src, cur_src[:-1, :].T)
    return M



def tricp1(src, dst, threshold=50, max_iterations=50000, tolerance=1e-7, seed=0, th=50):
    """
    TRICP 算法的实现。

    参数:
    src (numpy.ndarray): 源点集，形状为 (n, m)，其中 n 是点的数量，m 是维度（例如 2D 或 3D）。
    dst (numpy.ndarray): 目标点集，形状为 (p, m)。
    threshold (float): 距离阈值，用于修剪不匹配的点对。
    max_iterations (int): 最大迭代次数。
    tolerance (float): 收敛阈值，当误差变化小于此值时停止迭代。
    seed (int): 随机数种子，用于点集采样。

    返回:
    numpy.ndarray: 最终的变换矩阵，形状为 (m+1, m+1) 的齐次变换矩阵。
    """
    
    np.random.seed(seed)
    # 将 NaN 转换为 0
    src = np.nan_to_num(src)
    dst = np.nan_to_num(dst)
    # 采样到相同数量的点
    if dst.shape[0] > src.shape[0]:
        src = src.astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)].astype(np.float32)
    elif dst.shape[0] < src.shape[0]:
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)].astype(np.float32)
        dst = dst.astype(np.float32)
    
    
    #cur_src = np.hstack((src, np.ones((src.shape[0], 1)))).T  # 转换为齐次坐标，形状为 (m+1, n)
    cur_src = np.hstack((src, np.array([1] * src.shape[0]).reshape(-1, 1))).T
    prev_error = 0  # 用于存储上一次的误差
    for i in range(max_iterations):
        # 找到最近邻
        distances, indices = nearest_neighbor(cur_src[:2, :].T, dst) # 不包括齐次坐标的最后一维
        # 修剪不匹配的点对
        valid_indices = np.where(distances.ravel() < th)[0]
        valid_src = cur_src[:-1, valid_indices].T  # 有效的源点集
        valid_dst = dst[indices.ravel()[valid_indices]]  # 对应的目标点集
        # 计算最佳变换矩阵
        T, _, _ = best_fit_transform(valid_src, valid_dst)
        # 应用变换矩阵更新源点集
        cur_src = coor_transform(cur_src[:-1, :].T, T)
        #cur_src = np.vstack((cur_src, np.ones((1, cur_src.shape[1]))))  # 恢复齐次坐标
        # 计算平均误差
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
    # 最终的变换矩阵计算
    M, _, _ = best_fit_transform(src, cur_src[:-1, :].T)
    return M













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
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t


