import numpy as np
import matplotlib.pyplot as plt

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score
from sklearn.metrics.pairwise import rbf_kernel

def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - affinity_type: str, type of affinity matrix to construct. Options: 'knn' or 'rbf'.
    - k: int, the number of nearest neighbors for the KNN affinity matrix (default: 3).
    - sigma: float, bandwidth parameter for the RBF kernel (default: 1.0).

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix based on the specified type.
    """
    m,n = data.shape
    pairwise_distances = np.linalg.norm(np.tile(data.reshape((m,1,n)),(1,m,1)) - data,axis=2)
    if affinity_type == 'knn':
        affinity_matrix = np.zeros((m,m))
        affinity_matrix[np.arange(m).reshape(m,1), np.argsort(pairwise_distances)[:,:k]] = 1
        return (((affinity_matrix + affinity_matrix.T)/2) >= 0.5).astype(int)

    elif affinity_type == 'rbf':
        return np.exp(-np.square(pairwise_distances)/(2*np.square(sigma)))
        
        
    else:
        raise Exception("invalid affinity matrix type")

if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']
    fig, plots = plt.subplots(3,4,figsize=(14,10))
    i = 0
    for ds_name in datasets:
        dataset = np.load("datasets/%s.npz" % ds_name)
        X = dataset['data']     # feature points
        y = dataset['target']   # ground truth labels
        n = len(np.unique(y))   # number of clusters
        print(n)
        k = 3
        sigma = 1.0

        y_km, _ = k_means_clustering(X, n)
        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)
        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        print("K-means on %s:" % ds_name, clustering_score(y, y_km))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))

        plots[i][0].scatter(X[:,0],X[:,1],c= y,cmap= 'Accent',marker= 'o')
        plots[i][0].set_title("ground truth for %s" %ds_name)

        plots[i][1].scatter(X[:,0],X[:,1],c= y_km,cmap= 'Accent',marker= 'o')
        plots[i][1].set_title("kmeans clusteration for %s" %ds_name)

        plots[i][2].scatter(X[:,0],X[:,1],c= y_rbf,cmap= 'Accent',marker= 'o')
        plots[i][2].set_title("rbf clusteration for %s" %ds_name)

        plots[i][3].scatter(X[:,0],X[:,1],c= y,cmap= 'Accent',marker= 'o',)
        plots[i][3].set_title("knn clusteration for %s" %ds_name)   

        i += 1
    # TODO: Show subplots
    plt.tight_layout()
    plt.show()
