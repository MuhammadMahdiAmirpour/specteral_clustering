# import numpy as np
# import matplotlib.pyplot as plt
# 
# from kmeans import k_means_clustering
# from spectral import spectral_clustering, laplacian
# from metrics import clustering_score
# 
# def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
#     """
#     Construct the affinity matrix for spectral clustering based on the given data.
# 
#     Parameters:
#     - data: numpy array, mxn representing m points in an n-dimensional dataset.
#     - affinity_type: str, type of affinity matrix to construct. Options: 'knn' or 'rbf'.
#     - k: int, the number of nearest neighbors for the KNN affinity matrix (default: 3).
#     - sigma: float, bandwidth parameter for the RBF kernel (default: 1.0).
# 
#     Returns:
#     - affinity_matrix: numpy array, the constructed affinity matrix based on the specified type.
#     """
# 
#     # TODO: Compute pairwise distances
# #     no_of_obs, of_features = data.shape
# #     i, j = np.triu_indices(no_of_obs, k=1)
# #     a, b = data[i], data[j]
# #     upper_triangle_distance = np.sqrt(np.sum((a-b)**2, axis=1))
# #     d_mat = np.zeros((no_of_obs, no_of_obs))
# #     d_mat[i, j] = upper_triangle_distance
# #     d_mat = d_mat + d_mat.T
#     d_mat = np.linalg.norm(data[:, None] - data, axis=2)
# 
#     if affinity_type == 'knn':
# #         # TODO: Find k nearest neighbors for each point -> create a binary matrix
# #         nearest_neighbors = np.argpartition(d_mat, kth=k, axis=1)[:, :k]
# #         # Create an array of zeros with the same shape as the input matrix
# #         result = np.zeros_like(d_mat)
# #         # Set the corresponding elements in the result array to 1
# #         result[np.arange(d_mat.shape[0])[:, np.newaxis], nearest_neighbors] = 1
# #         result = result.astype(int)
# # 
# #         # TODO: Construct symmetric affinity matrix
# #         d_mat_sym = d_mat[result]
# # 
# #         # TODO: Return affinity matrix
# #         return d_mat_sym
# 
#         nearest_neighbors = np.argpartition(d_mat, kth=k, axis=1)[:, :k]
#         row_indices = np.arange(data.shape[0])[:,None]
#         col_indices = nearest_neighbors.flatten()
#         affinity_matrix = np.zeros((data.shape[0], data.shape[0]))
#         affinity_matrix[row_indices, col_indices] = 1
#         affinity_matrix += affinity_matrix.T
#         return affinity_matrix
# 
#     elif affinity_type == 'rbf':
#         # TODO: Apply RBF kernel
# #         no_of_obs, no_of_features = data.shape
# #         i, j = np.triu_indices(no_of_obs, k=1)
# #         a, b = data[i], data[j]
# #         upper_triangle_distance = np.exp(-np.sum((a-b)**2, axis=1)/(2*(sigma**2)))
# #         rbf_kernel = np.zeros((no_of_obs, no_of_obs))
# #         rbf_kernel[i, j] = upper_triangle_distance
# #         rbf_kernel =  rbf_kernel + rbf_kernel.T
#         # TODO: Return affinity matrix
# #         return rbf_kernel
#         affinity_matrix = np.exp(-d_mat**2/(2*(sigma**2)))
#         return affinity_matrix
# 
#     else:
#         raise Exception("invalid affinity matrix type")
# 
# 
# if __name__ == "__main__":
#     datasets = ['blobs', 'circles', 'moons']
#     algorithms = ['K-means', 'RBF + spectral', 'KNN + Spectral']
# 
#     # TODO: Create and configure plot
# #     fig, ax = plt.subplot(3, 4)
# #     ax[0].axis('tight') # turns off the axis lines and the labels
# #     ax[0].axis('off') # changes x and y axis limits such that all data is show
# 
#     for ds_name in datasets:
#         dataset = np.load("datasets/%s.npz" % ds_name)
#         X = dataset['data']     # feature points
#         y = dataset['target']   # ground truth labels
#         n = len(np.unique(y))   # number of clusters
# 
#         k = 3
#         sigma = 1.0
# 
#         y_km = k_means_clustering(X, n)
#         Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
#         y_rbf = spectral_clustering(Arbf, n)
#         Aknn = construct_affinity_matrix(X, 'knn', k=k)
#         y_knn = spectral_clustering(Aknn, n)
# 
#         print("K-means on %s:" % ds_name, clustering_score(y, y_km))
#         print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
#         print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))
# 
#         # TODO: Create subplots
# 
#     # TODO: Show subplots
# #     plt.show()
# 
import numpy as np
import matplotlib.pyplot as plt

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score


def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
    # Compute pairwise distances
    pairwise_distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)

    if affinity_type == 'knn':
        # Find k nearest neighbors for each point
         nearest_neighbors = np.argpartition(pairwise_distances, k, axis=1)[:, :k]
 
         row_indices = np.arange(data.shape[0])[:, np.newaxis]  # Create row indices with shape (n_data, 1)
         # col_indices = nearest_neighbors.flatten()  # Flatten nearest neighbors into 1D array
         col_indices = np.arange(data.shape[1])[np.newaxis, :]
         affinity_matrix = np.zeros((data.shape[0], data.shape[0]))
         affinity_matrix[row_indices,col_indices] = 1  # Assign 1s efficiently using indices
         affinity_matrix += affinity_matrix.T  # Enforce symmetry by adding the transpose
 
         # Return affinity matrix
         return affinity_matrix


    elif affinity_type == 'rbf':
        # Apply RBF kernel
        affinity_matrix = np.exp(-((pairwise_distances**2) / (2 * sigma ** 2)))

        # Return affinity matrix
        return affinity_matrix

    else:
        raise Exception("invalid affinity matrix type")

import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(ax, X, y, title):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='black')
    ax.set_title(title)
    return scatter

if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']

    # Create and configure plot
    fig, axes = plt.subplots(nrows=len(datasets), ncols=4, figsize=(16, 12))

    for i, ds_name in enumerate(datasets):
        dataset = np.load("datasets/%s.npz" % ds_name)
        X = dataset['data']  # feature points
        y = dataset['target']  # ground truth labels
        n = len(np.unique(y))  # number of clusters

        k = 3
        sigma = 1.0

        y_km, _ = k_means_clustering(X, n)
        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)
        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        # Print clustering scores
        print("K-means on %s:" % ds_name, clustering_score(y, y_km))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))

        # Create subplots
        scatter_km = plot_scatter(axes[i, 0], X, y_km, 'K-means')
        scatter_rbf = plot_scatter(axes[i, 1], X, y_rbf, 'RBF Affinity')
        scatter_knn = plot_scatter(axes[i, 2], X, y_knn, 'KNN Affinity')
        scatter_true = plot_scatter(axes[i, 3], X, y, 'Ground Truth')

    # Show subplots
    plt.show()
