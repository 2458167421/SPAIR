import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .utils import coor_transform, find_similar_index


import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def get_transform(adata_list, dst_id, src_id_list, target_label_id, threshold=50, max_iterations=5000, tolerance=1e-3, seed=0):
    transform_matrix = {}

    for src_id in src_id_list:
        if (src_id == dst_id):
            transform_matrix[src_id] = np.eye(3)
            continue

        dst_coor = adata_list[dst_id][target_label_id == adata_list[dst_id].obs['mclust']].obsm['spatial']
        src_coor = adata_list[src_id][target_label_id == adata_list[src_id].obs['mclust']].obsm['spatial']

        print(f'>>> INFO: dst slice id: {dst_id}, src slice id: {src_id}')
        print('>>> INFO: dst coordination shape:', dst_coor.shape)
        print('>>> INFO: src coordination shape:', src_coor.shape)

        T = nicp(src_coor, dst_coor, threshold, max_iterations, tolerance, seed)
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


def nicp(src, dst, threshold=50, max_iterations=5000, tolerance=1e-3, seed=0):
    # sample spots from two sets to the same number
    np.random.seed(seed)
    if (dst.shape[0] > src.shape[0]):
        src = src.astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)].astype(np.float32)
    elif (dst.shape[0] < src.shape[0]):
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)].astype(np.float32)
        dst = dst.astype(np.float32)

    # calculate normals and curvatures
    src_normals, src_curvatures = calculate_normals_and_curvatures(src)
    dst_normals, dst_curvatures = calculate_normals_and_curvatures(dst)

    # init
    cur_src = np.hstack((src, np.array([1] * src.shape[0]).reshape(-1, 1))).T
    prev_error = 0

    # train NICP
    for i in range(max_iterations):
        distances, indices = nearest_neighbor_with_normals_and_curvatures(cur_src[:2, :].T, dst, src_normals, dst_normals,
                                                                          src_curvatures, dst_curvatures)
        M, _, _ = best_fit_transform(cur_src[:2, :].T, dst[indices])
        cur_src = coor_transform(cur_src[:2, :].T, M)
        print (i)
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


def calculate_normals_and_curvatures(points, k=10):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)
    _, indices = neigh.kneighbors(points)

    normals = []
    curvatures = []

    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        cov = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        curvature = eigenvalues.min() / eigenvalues.sum()

        normals.append(normal)
        curvatures.append(curvature)

    return np.array(normals), np.array(curvatures)


def nearest_neighbor_with_normals_and_curvatures(src, dst, src_normals, dst_normals, src_curvatures, dst_curvatures):
    distances = []
    indices = []
    for i in range(src.shape[0]):
        min_distance = np.inf
        min_index = -1
        for j in range(dst.shape[0]):
            point_distance = np.linalg.norm(src[i] - dst[j])
            normal_distance = np.linalg.norm(src_normals[i] - dst_normals[j])
            curvature_distance = np.abs(src_curvatures[i] - dst_curvatures[j])
            combined_distance = point_distance + 0.1 * normal_distance + 0.1 * curvature_distance
            if combined_distance < min_distance:
                min_distance = combined_distance
                min_index = j
        distances.append(min_distance)
        indices.append(min_index)
    return np.array(distances), np.array(indices)


# ICP algorithm from https://github.com/ClayFlannigan/icp/blob/master/icp.py
def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def coor_transform(coor, T):
    coor_homogeneous = np.hstack((coor, np.ones((coor.shape[0], 1))))
    transformed_coor = np.dot(T, coor_homogeneous.T)
    return transformed_coor[:2, :]


