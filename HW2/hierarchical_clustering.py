import numpy as np


class HierarchicalClustering(object):
    def __init__(self, points: np.ndarray):
        self.N, self.D = points.shape
        self.points = points
        self.current_iteration = 0
        self.distances, self.cluster_ids = self.create_distances(points)
        self.clustering = np.zeros((self.N - 1, 4))
        self.cluster_sizes = np.zeros(self.N * 2 - 1)
        self.cluster_sizes[: self.N] = 1
        self.initial_distances = self.distances.copy()

    def pairwise_dist(self, x, y):
        """
        Pairwise Euclidean distance, copy over from kmeans!
        We won't explicitly test speed, but you may timeout if you don't do kmeans's fast pairwise.

        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        distance_component_1 = np.sum(np.square(x), axis = 1, keepdims = True) 
        distance_component_2 = np.sum(np.square(y), axis = 1, keepdims = True).T 
        distance_component_3 = 2 * (x @ y.T)
        return np.sqrt(distance_component_1 + distance_component_2 - distance_component_3)


    def create_distances(self, points: np.ndarray):
        """
        Create the pairwise distance matrix and index map, given points.
            The distance between a cluster and itself should be np.inf
        Args:
            points: N x D numpy array where N is the number of points
        Return:
            distances: N x N numpy array where distances[i][j] is the euclidean distance between points[i, :] and points[j, :].
                       distances[i, i] should always be np.inf in order to calculate the closest clusters more easily
            cluster_ids: (N,) numpy array where index_array[i] gives the cluster id of the i-th column
                         and i-th row of distances. Initially, each point i is assigned cluster id i
        """
        distances = self.pairwise_dist(points, points)
        np.fill_diagonal(distances, np.inf)

        cluster_ids = np.arange(points.shape[0])

        return distances, cluster_ids

    def iterate(self):
        """
        Performs one iteration of the algorithm
            1. Find the two closest clusters using self.distances (if there are multiple minimums, use the first occurence in flattened array)
            2. Replace first cluster's row and col with the newly combined cluster distances in self.distances,
               ensuring distances[i, i] is still np.inf
            3. Delete second cluster's row and col in self.distances
            4. Update self.cluster_ids where new cluster's id should be self.N + self.current_iteration,
               see definition in `create_distances` for more details
            5. Update self.cluster_sizes, where self.cluster_sizes[i] contains the number of points with cluster id i
            6. Update self.clustering, where
               self.clustering[self.current_iteration] = [first cluster id, second cluster id, distance between first and second clusters, size of new cluster]
            7. Update current_iteration
        Hint:
        You'll need to update self.distances, self.cluster_ids, self.cluster_sizes, self.clustering, and self.current_iteration

        While self.distances and self.cluster_ids only keeps information about the current clusters,
            self.cluster_sizes keep track of sizes for all clusters
        """
        flat_idx = np.argmin(self.distances)
        first_idx, second_idx = np.unravel_index(flat_idx, self.distances.shape)
        dist_first_and_second = self.distances[first_idx, second_idx]

        new_distances = np.minimum(self.distances[first_idx, :], self.distances[second_idx, :])

        self.distances[first_idx, :] = new_distances
        self.distances[:, first_idx] = new_distances
        self.distances[first_idx, first_idx] = np.inf

        
        self.distances = np.delete(self.distances, second_idx, axis=1)
        self.distances = np.delete(self.distances, second_idx, axis=0)

        first_cluster_id = self.cluster_ids[first_idx]
        second_cluster_id = self.cluster_ids[second_idx]

        new_cluster_id = self.N + self.current_iteration        
        self.cluster_ids[first_idx] = new_cluster_id
        self.cluster_ids = np.delete(self.cluster_ids, second_idx)
        
        new_cluster_size = self.cluster_sizes[first_cluster_id] + self.cluster_sizes[second_cluster_id]
        self.cluster_sizes[new_cluster_id] = new_cluster_size

        self.clustering[self.current_iteration] = [first_cluster_id, second_cluster_id, dist_first_and_second, new_cluster_size]

        self.current_iteration += 1

    def fit(self):
        """
        Fits the model on the dataset by calling `iterate`.
        Each call of `iterate` should combine two clusters, logging what was combined in self.clustering

        Return:
            self.clustering, where self.clustering[iteration_index] = [i, j, distance between i and j, size of new cluster]
        """
        for _ in range(self.N - 1):
            self.iterate()
            
        return self.clustering
