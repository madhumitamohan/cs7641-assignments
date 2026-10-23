import numpy as np


class KMeans(object):
    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        """
        Args:
            points: np.ndarray(N,D), where N is # points and D is the dimensionality
            k: int, number of clusters
            init: str, how to initialize the centers
            max_iters: int, maximum number of iterations
            rel_tol: float, convergence criteria with respect to relative change of loss (number between 0 and 1)
        """
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """
        Initialize the centers randomly.
        Use np.random.choice to pick K indices from the integers [0, N) without replacement,
        then use those indices to index into self.points.

        Update:
            self.centers: np.ndarray(K,D), randomly selected centers
        Return:
            self.centers, set it and also return it
        Hints:
            Numpy will let you index an array with an iterable!
            It will vstack the subsequent results together into a resultant array.
            We won't stop you from looping here, but it's a cool trick to know.
        """
        center_indices = np.random.choice(len(self.points), self.K, replace= False)
        self.centers = self.points[center_indices]
        return self.centers

    def kmpp_init(self):
        """
        We can do better than random sampling!
        KMeans++ uses the intuition that good initial clusters will be further away from each other.

        1. Sample 1% of the points from the dataset, uniformly at random and without replacement.
            This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
        2. From the above sample, select only one random point to be the first cluster center.
        3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
        4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center.
            In other words, we will choose the next center by picking the most remote data point
            instead of sampling randomly like in step 2. You may break ties arbitrarily.
        5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.

        Update:
            self.centers: np.ndarray(K,D), randomly selected centers
        Return:
            self.centers, set it and also return it
        """
        sample_size = int(len(self.points) * 1/100)
        sample_indices = np.random.choice(len(self.points), sample_size, replace= False)
        sample_points = self.points[sample_indices]

        first_centre_index = np.random.choice(len(sample_points), 1, replace= False)
        first_centre = self.points[first_centre_index]
        self.centers = first_centre

        while self.centers.shape[0] < self.K:
            point_centre_distances = pairwise_dist(sample_points, self.centers)
            nearest_centre_indices = np.argmin(point_centre_distances, axis=1)  
            nearest_centres = self.centers[nearest_centre_indices]   

            point_nearest_centres_distances = np.diag(pairwise_dist(sample_points, nearest_centres))

            farthest_point = np.argmax(point_nearest_centres_distances)
            self.centers = np.append(self.centers, self.points[farthest_point][None, :], axis=0)

        return self.centers


    def update_assignment(self):
        """
        Assign each point to the cluster whose center is the closest to that point.

        Update:
            self.assignments: np.ndarray(N,), self.assignments[i] is the index of the cluster to which self.points[i,:] belongs
        Return:
            self.assignments, set it and also return it
        Note:
            Don't overwork here. Call your pairwise_dist and work with that data to find the closest center per point.
            Looping may cause timeouts here.
        """
        point_centre_distances = pairwise_dist(self.points, self.centers)
        nearest_centre_indices = np.argmin(point_centre_distances, axis=1)
        self.assignments = nearest_centre_indices
        return self.assignments


    def update_centers(self):
        """
        Update the positions of the cluster centers to be the centroid of all data in the cluster.

        Loop through each cluster's index, 0...k-1:
            Find the subset of points belonging to the cluster with self.assignments.
            If that subset is non-empty, calculate the centroid, and update self.centers with that value.
            If the subset is empty, we need to pick a new value. Using the current state of self.centers:
                1. Calculate the pairwise distance from every point in self.points to every center in self.centers.
                2. Find the point that is farthest away from every other center.
                3. Set the center for this empty to that lonely point.

        Update:
            self.centers: np.ndarray(K,D)[float], self.centers[i,:] should be the center for the i^th cluster
        Return:
            self.centers, set it and also return it
        Note:
            Points is non-typed. It might be an integer type. However, the dtype of the centers should definitely be a float.
                e.g., if your cluster has 1 and 2 in it, the center should be 1.5
            Watch out for dtype casting! You may get unexpected behavior.
        """
        self.centers = self.centers.astype(float)
        for i in range(self.K):
            mask = (self.assignments == i)
            cluster_points = self.points[mask]
            if len(cluster_points) > 0:
                self.centers[i] = np.mean(cluster_points, axis=0)
            else:
                point_centre_distances = pairwise_dist(self.points, self.centers)
                nearest_centre_indices = np.argmin(point_centre_distances, axis=1)
                farthest_point_idx = np.argmax(nearest_centre_indices)
                self.centers[i] = self.points[farthest_point_idx]
        return self.centers

    def get_loss(self):
        """
        The loss will be defined as the sum of the squared distances between each point and it's respective center.

        Update:
            self.loss: float, the objective function of KMeans
        Return:
            self.loss, set it and also return it
        """
        diff = self.points - self.centers[self.assignments]
        self.loss = float(np.sum(np.square(diff)))
        return self.loss

    def train(self):
        """
        Train KMeans to cluster the data:
            0. Recall that self.centers have already been initialized in __init__
            1. Update the cluster assignment for each point (update_assignment)
            2. Update the cluster centers based on the new assignments (update_centers)
            3. Calculate the loss (get_loss) and check if the model has converged to break the loop early.
               - The convergence criteria is measured by whether the relative change in loss from the previous
                 iteration is less than the relative tolerance threshold (self.rel_tol).
               - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
            4. Iterate through steps 1 to 3 at most max_iters times.

         Update:
             self.centers: np.ndarray(K,D)
             self.assignments: np.ndarray(N,)
             self.loss: float
             All of these values will be updated in the functions you already implemented, you just need to call them.
         Return:
             (self.centers, self.assignments, self.loss)
         Note:
             Do not loop over all the points in every iteration. This may result in time out errors.
        """
        loss_prev = None
        for iter in range(self.max_iters):
            self.update_assignment()
            self.update_centers()
            loss_curr = self.get_loss()
            if loss_prev is not None:
                if abs(loss_curr - loss_prev)/loss_prev < self.rel_tol:
                    break
            loss_prev = loss_curr
        return (self.centers, self.assignments, self.loss)



def pairwise_dist(x, y):
    """
    Calculates the Euclidean distance between every cross pair of points, one from each input.
    This function is to be space and time efficient.
    1. No looping, only numpy functions.
    2. No intermediate arrays larger than the output, e.g., (NxMxD).

    You'll need to follow the instructions in the notebook, where the Euclidean distance
    is expressed as a combination of several terms, sums of rows of X^2, sums of rows of Y^2,
    and a cross term containing the dot product of X[i, :] and Y[j, :]

    Args:
        x: np.ndarray(N,D), first set of points
        y: np.ndarray(M,D), second set of points
    Return:
        dist: np.ndarray(N,M), where dist[i, j] is the
            euclidean distance between x[i, :] and y[j, :]
    """
    distance_component_1 = np.sum(np.square(x), axis = 1, keepdims = True) 
    distance_component_2 = np.sum(np.square(y), axis = 1, keepdims = True).T 
    distance_component_3 = 2 * (x @ y.T)
    return np.sqrt(distance_component_1 + distance_component_2 - distance_component_3)


def adjusted_rand_statistic(xGroundTruth, xPredicted):
    """
    Calculate ARI by enumerating pairs of points and binning them by agreement.

    The idea is to make the comparison of Predicted and Ground truth data points.
        1. Iterate over all distinct pairs of points (looping is permitted!).
        2. Compare the prediction pair label with the ground truth pair.
        3. Based on the analysis, we can figure out whether both points fall under TP/FP/FN/FP
           i.e. if a pair falls under TP, increment by 2 (one for each point in the pair).
        4. Then calculate the adjusted rand index value

    Args:
        xPredicted: list of length N, the predicted cluster labels
        xGroundTruth: list of length N, the "correct" cluster labels
    Return:
        ARI: float, computed ARI value
    """
    N = len(xGroundTruth)
    tp_count, fp_count, tn_count, fn_count = 0,0,0,0

    for i in range(N):
        for j in range(i+1, N):
            same_cluster_gt = (xGroundTruth[i] == xGroundTruth[j])
            same_cluster_pred = (xPredicted[i] == xPredicted[j])
            if same_cluster_gt and same_cluster_pred:
                tp_count += 1
            elif not same_cluster_gt and not same_cluster_pred:
                tn_count += 1
            elif not same_cluster_gt and same_cluster_pred:
                fp_count += 1
            elif same_cluster_gt and not same_cluster_pred:
                fn_count += 1

    numerator = 2*((tp_count*tn_count) - (fp_count * fn_count))
    denominator = (tp_count + fn_count)*(fn_count + tn_count)+ (tp_count + fp_count)*(fp_count + tn_count)
    return float(numerator/denominator)



def silhouette_score(X, labels):
    """
    The idea is to calculate the mean distance between a point and the other points
    in its cluster (the intra cluster distance) and the mean distance between a point and the
    other points in its closest cluster (the inter cluster distance)
        1.  Calculate the pairwise_dist between all points to each other (N x N)
        2.  Loop over all points in the provided data (X) and for each point calculate:

            Intra Cluster Distances (point p to the other points in its own cluster)
                a. Identify all points in the same cluster as p (excluding p itself)
                b. Calculate the mean pairwise_dist between p and the other points
                c. If there are no other points in the same cluster, assign an
                   intra-cluster distance of 0

            Inter Cluster Distances (point p to the points in the closest cluster)
                a. Loop over all clusters except for p's cluster
                b. For each cluster, identify all points belonging to it
                c. Calculate the mean pairwise_dist between p and those points
                d. Set the inter-cluster distance to the minimum mean pairwise_dist
                   among all clusters. Again, if there are no other clusters, use
                   an inter-cluster distance of 0.

        3. Calculate the silhouette scores for each point using
                s_i = (mu_out(x_i) - mu_in(x_i)) / max(mu_out(x_i), mu_in(x_i))
        4. Average the silhouette score across all points

    Args:
        X: np.ndarray(N,D), the coordinates of the data
        labels: np.ndarray(N,), the corresponding labels of the data
    Return:
        silhouette score: final coefficient value of type np.float64
    Note:
        You can refer to the Clustering Evaluation slides from Lecture for additional info if needed
    """
    dist_between_points = pairwise_dist(X, X)
    unique_clusters = np.unique(labels)
    N = len(X)

    s_scores = []

    for index in range(N):

        inter_cluster_dist = []

        datapoint = X[index]
        own_cluster = labels[index]

        #Intra cluster distance
        same_cluster_mask = (labels == own_cluster)
        same_cluster_mask[index] = False 

        if same_cluster_mask.sum() == 0:
            mu_in = 0
        else:
            mu_in = np.mean(dist_between_points[index, same_cluster_mask])

        #Inter cluster
        for c in unique_clusters:
            if c == own_cluster:
                continue
            other_cluster_mask = (labels == c)
            if other_cluster_mask.sum() == 0:
                continue

            mean_dist = np.mean(dist_between_points[index, other_cluster_mask])
            
            inter_cluster_dist.append(mean_dist)

        if len(inter_cluster_dist) == 0: #Only one cluster
            mu_out = 0
        else:
            mu_out = min(inter_cluster_dist)

        #SC score for point i
        denom = max(mu_in, mu_out)

        if denom == 0:
            s_i = 0
        else:
            s_i = (mu_out - mu_in)/denom

        s_scores.append(s_i)

    return np.mean(s_scores)
        