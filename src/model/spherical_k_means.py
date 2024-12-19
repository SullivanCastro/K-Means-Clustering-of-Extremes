import numpy as np
from preprocessing import Preprocessing
from sklearn.preprocessing import normalize
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans


class ExtremeSphericalKMeans:

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, n_init: int = 10, threshold: float = 0.1) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.threshold = threshold
        self.cluster_centers_ = None
        self.labels_ = None

    
    def _project_to_l1_unit_ball(self, v: ArrayLike) -> ArrayLike:
        """
        Projects a vector onto the L1 unit ball (sum of absolute values equals 1).
        """
        return v / np.linalg.norm(v, ord=1, axis=1, keepdims=True)


    def _initialize_centroids(self, X: ArrayLike, labels: ArrayLike) -> ArrayLike:
        """
        Initialize centroids randomly from the dataset.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            indices = np.where(labels == i)[0]
            centroids[i] = X[indices == i].mean(axis=0)
        return centroids


    def _compute_cosine_similarity(self, X: ArrayLike, centroids: ArrayLike) -> ArrayLike:
        """
        Compute cosine similarity between data points and centroids.
        """
        return 1 - np.dot(X, centroids.T)
    

    def _compute_centroid(self, X: ArrayLike, labels: ArrayLike) -> ArrayLike:
        """
        Compute the centroid of a cluster.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k] = X[labels == k].mean(axis=0)
        return centroids
    

    def _labelize(self, X: ArrayLike, centroids: ArrayLike) -> ArrayLike:
        """
        Assign labels to each data point based on the closest centroid.
        """
        similarity = self._compute_cosine_similarity(X, centroids)
        return np.argmax(similarity, axis=1)


    def fit(self, X: ArrayLike) -> None:
        """
        Fit the spherical k-means clustering on the data.

        Parameters
        ----------
        X : ArrayLike
            The input data to cluster, normalized beforehand.
        """
        # Preprocess the data (normalize each vector to have unit norm)
        self.X = Preprocessing.process(X, self.threshold)

        # Initialize labels
        labels = np.random.randint(low=0, high=self.n_clusters, size=self.X.shape[0])
        
        # Compute initial centroids
        new_centroids = self._compute_centroid(self.X, labels)
        centroids = new_centroids.copy() + 4
        
        for _ in range(self.max_iter):
            # Compute new labels
            labels = self._labelize(self.X, new_centroids)

            # Compute new centroids
            new_centroids = self._compute_centroid(self.X, labels)
            
            # Projection on L1 unit ball
            new_centroids = self._project_to_l1_unit_ball(new_centroids)


            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids.copy()

        self.cluster_centers_ = centroids
        self.labels_ = labels



    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : ArrayLike
            New data to predict.

        Returns
        -------
        labels : ArrayLike
            Index of the cluster each sample belongs to.
        """
        # Preprocess the data (normalize each vector to have unit norm)
        X = Preprocessing.process(X, self.threshold)
        
        # Compute cosine similarity with cluster centers
        similarity = self._compute_cosine_similarity(X, self.cluster_centers_)
        return np.argmax(similarity, axis=1)
