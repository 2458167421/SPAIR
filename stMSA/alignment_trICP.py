import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .utils import coor_transform, find_similar_index

def nearest_neighbor0(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def nearest_neighbor(src, dst):
    # 检查并处理 src 中的 NaN 值
    src = np.nan_to_num(src, nan=0)
    # 检查并处理 dst 中的 NaN 值
    dst = np.nan_to_num(dst, nan=0)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src)
    distances = distances.flatten()
    indices = indices.flatten()
    return distances, indices

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


def get_transform(adata_list, dst_id, src_id_list, target_label_id, threshold=50, max_iterations=1000, tolerance=1e-3, seed=0,distance_threshold=1.0):
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
        T = tricp(src_coor, dst_coor, threshold, max_iterations, tolerance, seed,distance_threshold)
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
    M, _, _ = best_fit_transform(src, cur_src[:2, :].T)
    return M



def tricp(src, dst, threshold=50, max_iterations=5000, tolerance=1e-3, seed=0, distance_threshold=1.0):
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
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(cur_src[:2, :].T, dst)
        # 剔除距离过远的点对
        valid_indices = distances < distance_threshold
        valid_src = cur_src[:2, :].T[valid_indices]
        valid_dst = dst[indices][valid_indices]

        M, _, _ = best_fit_transform(valid_src, valid_dst)
        cur_src = coor_transform(cur_src[:2, :].T, M)
        print(i)
        print(_)
        mean_error = np.mean(distances[valid_indices])
        if (np.abs(prev_error - mean_error) < tolerance):
            # stuck in local optimum -> rotate src
            if (threshold < mean_error):
                rotate_deg = np.pi * 2/ 3
                cur_src = coor_transform(cur_src[:2, :].T, np.array([
                    [np.cos(rotate_deg), np.sin(rotate_deg), 0.5],
                    [-np.sin(rotate_deg), np.cos(rotate_deg), 0.5],
                    [0, 0, 1]
                ]))
            else:
                break
        prev_error = mean_error   
    
    print(f'>>> INFO: current distance: {mean_error}')
    M, _, _ = best_fit_transform(src, cur_src[:2, :].T)
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



def tricp4(src, dst, threshold=50, max_iterations=50000, tolerance=1e-7, seed=0, trim_threshold=20):
    """
    TRICP 算法的实现。

    参数:
    src (numpy.ndarray): 源点集，形状为 (n, m)，其中 n 是点的数量，m 是维度（例如 2D 或 3D）。
    dst (numpy.ndarray): 目标点集，形状为 (p, m)。
    threshold (float): 用于局部最优的判断距离阈值。
    trim_threshold (float): 用于修剪不匹配的点对的距离阈值。
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
    prev_error = 0  # 用于存储上一次的误差
    for i in range(max_iterations):
        # 找到最近邻
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(cur_src[:-1, :].T, return_distance=True)  # 不包括齐次坐标的最后一维
        # 修剪不匹配的点对，使用 trim_threshold
        valid_indices = np.where(distances.ravel() < trim_threshold)[0]
        valid_src = cur_src[:-1, valid_indices].T  # 有效的源点集
        valid_dst = dst[indices.ravel()[valid_indices]]  # 对应的目标点集
        # 计算最佳变换矩阵
        T, _, _ = best_fit_transform(valid_src, valid_dst)
        # 应用变换矩阵更新源点集
        cur_src[:-1, :] = np.dot(T[:-1, :-1], cur_src[:-1, :]) + T[:-1, -1].reshape(-1, 1)
        # 计算平均误差
        mean_error = np.mean(distances[valid_indices])
        if np.abs(prev_error - mean_error) < tolerance:
            # 陷入局部最优解 -> 旋转源点集，使用 threshold 进行判断
            if threshold < mean_error:
                rotate_deg = np.pi * 2 / 3
                rotation_matrix = np.array([
                    [np.cos(rotate_deg), np.sin(rotate_deg), 0.5],
                    [-np.sin(rotate_deg), np.cos(rotate_deg), 0.5],
                    [0, 0, 1]
                ])
                transformed_points = coor_transform(cur_src[:-1, :].T, rotation_matrix).T[:-1, :]
                cur_src[:-1, :] = transformed_points
            else:
                break
        prev_error = mean_error
    print(f'>>> INFO: current distance: {mean_error}')
    # 最终的变换矩阵计算
    M, _, _ = best_fit_transform(src, cur_src[:-1, :].T)
    return M


def tricp4(src, dst, threshold=50, max_iterations=50000, tolerance=1e-7, seed=0):
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
            # 陷入局部最优解 -> 旋转源点集
            if threshold < mean_error:
                rotate_deg = np.pi * 2 / 3
                rotation_matrix = np.array([
                    [np.cos(rotate_deg), np.sin(rotate_deg), 0.5],
                    [-np.sin(rotate_deg), np.cos(rotate_deg), 0.5],
                    [0, 0, 1]
                ])
                cur_src[:-1, :] = coor_transform(cur_src[:-1, :].T, rotation_matrix).T
            else:
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
