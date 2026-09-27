import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-20


def complete_(data):
    """
    Find and return all rows in the data for which there is no missingness (nan).
    You should not use looping. Numpy supports indexing with Boolean arrays.

    Args:
        data: np.ndarray(N,D+1), data where the last column is the labels
    Return:
        labeled_complete: np.ndarray(?,D+1), the rows of data that have all features and a label
    """
    raise NotImplementedError


def incomplete_(data):
    """
    Find and return all rows in the data for which there is missingness (nan) in the features, but not the label.
    You should not use looping. Numpy supports indexing with Boolean arrays.

    Args:
        data: np.ndarray(N,D+1), data where the last column is the labels
    Return:
        labeled_incomplete: np.ndarray(?,D+1), the rows of data that have a label but are missing at least 1 feature
    """
    raise NotImplementedError


def unlabeled_(data):
    """
    Find and return all rows in the data for which there is missingness (nan) in the label, but not the features.
    You should not use looping. Numpy supports indexing with Boolean arrays.

    Args:
        data: np.ndarray(N,D+1), data where the last column is the labels
    Return:
        unlabeled_complete: np.ndarray(?,D+1), the rows of data that have all features but are missing a label
    """
    raise NotImplementedError


class CleanData:
    def __init__(self):
        pass

    def pairwise_dist_missingness_aware(self, x, y):
        """
        A missingness-aware distance metric computes the Euclidean distance using only
        the features where both points have valid (non-NaN) values, then applies a
        correction factor to account for the missing features.

        The correction factor is sqrt(D/D_valid),
        where D_valid is the number of features where both points had valid (non-NaN) values.
        An example is worked out in the notebook.

        Args:
            x: np.ndarray(N,D), points
            y: np.ndarray(M,D), points
        Returns:
            dist_missingness_aware: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :] with a correction factor applied based on total number of features / non-nan values in X[i] and Y[i]
        Note:
            You are not permitted to use loops. Instead, use broadcasting tricks.
            You may use the large intermediate technique where first you broadcast up to (N,M,D) then sum over D.
            Of course, you will need to take care of nans. np.isnan and np.where will be especially useful.
        """
        raise NotImplementedError

    def __call__(self, incomplete_points, complete_points, K, **kwargs):
        """
        This function should fill in missing feature values by sampling the
        average value for said features from the K-nearest neighbors of a data point.

        Args:
            incomplete_points: np.ndarray(N_incomplete, D+1), the incomplete labeled observations, labels at the end
            complete_points: np.ndarray(N_complete, D+1), the complete labeled observations, labels at the end
            K: int, corresponding to the number of nearest neighbors you should average over
        Return:
            imputed_data: np.ndarray(N_complete+N_incomplete, D+1), containing both the complete points and recently filled points (in that order)
        Starting Code:
            data = np.vstack((complete_points, incomplete_points))
            imputed_data = data.copy()  # write your computed values into this
            pw_dist = self.pairwise_dist_missingness_aware(data[:,:-1], data[:,:-1])
            unprocessed_neighbor_idxs = np.argsort(pw_dist, axis=1)
            ...
        Notes:
            1. You need to find the k-closest points that actually have the feature you're looking for.
               You should temporarily ignore points with the same missingness.
            2. Don't write into the data you're using. That would make this iterative.
               Write into a write-only copy, and return that at the end.
            3. Don't include the labels on the distance function.
               Categorical variables are ill-defined on Euclidean distance.
        """
        raise NotImplementedError


def median_clean_data(data):
    """
    A simpler approach, replace every NaN with the median of the column.

    Args:
        data: np.ndarray(N, D+1), data with missing features, but non-missing labels (last element)
    Return:
        imputed_data: np.ndarray(N, D+1), data with missingness imputed by the median
    Notes:
        When taking the median of any feature, do not count the NaN value.
    """
    raise NotImplementedError
