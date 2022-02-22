from abc import ABC, abstractmethod
import numpy as np

class ClusterModel(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class MyKMeans(ClusterModel):
    """
    A class to perform k-means clustering.

    Attributes:
    ----------
    n_clusters: int
        number of clusters
    max_iter: int
        maximal number of iterations
    tol: float
        accuracy
    seed: int
        a random seed
    inertia_: float
        the within-cluster sum of squares
    _shape: int
        number of objects    
    centroids: dict
        centroid center coordinates
    labels_: numpy.ndarray
        cluster labels
    _stop: bool
        stop condtion 

    Methods:
    --------
    _init_centroids(X):
        Initialize starting centroid centers
    _calculate_distance(arr1, arr2):
        Calculates Euclidean distance between two arrays.
    _get_shape(X):
        Gets number of objects.
    _set_labels():
        Convert cluster labels to the numpy array.
    _stop_cycle(old_centroid, new_centroid):
        Stops the loop when the relative difference between the old and new centroids is less than the given precision.
    _get_loss(X):
        Computes the within-cluster sum of squares.
    _run_single_kmeans(X):
        Runs a single iteration of the algorithm.
    fit(X):
        Compute k-means clustering.
    predict(X):
        Predict the closest cluster each sample in X belongs to.
    """

    def __init__(self, n_clusters, max_iter=100, tol=1e-10, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.inertia_ = 0.0 
        self._shape = None
        self.centroids = None 
        self.labels_ = None
        self._stop = False

    def _init_centroids(self, X):
        """
        Initialize starting centroid centers.
            Parameters:
            -----------
                X: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                self.centroids: dict
                    where the keys are the cluster indexes, the values are centroid coordinates
        """

        self._get_shape(X)
        self.centroids = {}
        np.random.seed(self.seed)
        X = np.random.permutation(X)

        for i in range(self.n_clusters):
            self.centroids[i] = X[i]
        
        return self.centroids

    @staticmethod
    def _calculate_distance(arr1, arr2):
        """
        Calculates Euclidean distance between two arrays.
            Parameters:
            -----------
                arr1: numpy.ndarray
                arr2: numpy.ndarray
            Return:
            -------
                dist: numpy.float64
        """

        dist = np.linalg.norm(arr1 - arr2)
        return dist

    def _get_shape(self, X):
        """
        Gets number of objects.
            Parameters:
            -----------
                X: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                None            
        """

        self._shape = X.shape[0]

    def _set_labels(self):
        """
        Convert cluster labels to the numpy array.
            Parameters:
            -----------
                None
            Return:
            -------
                None
        """
        self.labels_ = np.array(self.labels_).reshape((self._shape,))
    
    def _stop_cycle(self, old_centroid, new_centroid):
        """
        Stops the loop when the relative difference between the old and new centroids is less than the given precision.
            Parameters:
            -----------
                old_centroid: numpy.ndarray
                    Coordinates of the old centroid
                new_centroid: numpy.ndarray
                    Coordinates of the new centroid
            Return:
            -------
                None
        """

        difference = abs(np.sum((new_centroid-old_centroid)/old_centroid*100.0)) 
    
        if difference <= self.tol:
            self._stop = True
    def _get_loss(self, X):
        """
        Computes the within-cluster sum of squares.
            Parameters:
            -----------
                X: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                None
        """

        self.inertia_ = 0.0
        for cls_idx in range(self.n_clusters):
            for point in self.clusters[cls_idx]:
                loss = self._calculate_distance(point, self.centroids[cls_idx])
                self.inertia_ += loss

    def _run_single_kmeans(self, X):
        """
        Runs a single iteration of the algorithm.
            Parameters:
            -----------
                X: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                None
        """

        # initialize centroids 
        if self.centroids == None:
            self._init_centroids(X)
        
        # empty dict: {label : [data point]}
        self.labels_ = []
        self.clusters = {}
        for i in range(self.n_clusters):
            self.clusters[i] = []
    
        # assigns the labels to the data points accordingly the distance between
        # the data point and the center of the cluster
        for data_point in X:

            distances = [self._calculate_distance(data_point, self.centroids[centroid]) for centroid in self.centroids]
            point_label = np.argmin(distances)                         
            self.clusters[point_label].append(data_point)
            self.labels_.append(point_label)
        
        # update centroid
        old_centroids = dict(self.centroids)                         

        for cluster in self.clusters:
            self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)

        self._set_labels()
       
        for centroid in self.centroids:

            original_centroid = old_centroids[centroid]
            current_centroid = self.centroids[centroid]
            self._stop_cycle(original_centroid, current_centroid)
        
    def fit(self, X):
        """
        Compute k-means clustering.
            Parameters:
            -----------
                X: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                None
        """
        
        for iter in range(self.max_iter):

            old_loss = self.inertia_
            self._run_single_kmeans(X)
            self._get_loss(X)               
            if self._stop and iter > 10 and np.abs(old_loss - self.inertia_) < self.tol:
                nl = '\n'
                print('-' * 40)
                print(f"The maximum number of iterations is not reached.{nl}A number of iterations is {iter}. Desired accuracy achieved: {self.tol}")
                print('-' * 40)
                break
        

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
            Parameters:
            -----------
                X: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                prediction: list
                    The list of cluster indices
        """

        prediction = []
        for data_point in X:
            distances = [self._calculate_distance(data_point, self.centroids[centroid]) for centroid in self.centroids]
            point_label = np.argmin(distances)
            prediction.append(point_label)

        return prediction
