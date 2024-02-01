import numpy as np

def k_means_clustering(data, k, max_iterations=100):
    # Initialize centroids randomly
    m, _ = data.shape
    centroids = data[np.random.choice(m, k, replace=False)]
    # Iterate until convergence or maximum iterations reached
    for _ in range(max_iterations):
        # Assign data points to closest centroids
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # Update centroids as the mean of assigned data points
        old_centroids = centroids
        centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.all(old_centroids - centroids < 0.00001):
            break

    return labels,centroids

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    np.random.seed(1)  # Set random seed for reproducibility
    random_points = np.random.randint(0, 100, (100, 2))
    data = make_blobs(n_samples=100, n_features=2)
    random_points = data[0]
    labels, centroids = k_means_clustering(random_points, 3)
    plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=range(len(centroids)), marker="*", s=200)

    n_samples = 100
    mu1 = np.array([0,0,0])
    sigma1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data1 = np.random.multivariate_normal(mu1, sigma1, size=n_samples)
    mu2 = np.array([1,1,1])
    sigma2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data2 = np.random.multivariate_normal(mu2, sigma2, size=n_samples)
    data = np.vstack((data1, data2))
    
    k = 2

    labels, centroids = k_means_clustering(data, k)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, s=20, cmap='viridis')
    ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c="black", s=100, marker='*')
    plt.show()

