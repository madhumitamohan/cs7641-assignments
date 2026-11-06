from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt


class PCAReduction(object):
    def __init__(self):
        pass

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform SVD on a matrix. You should use numpy's SVD.

        This is very similar to what you will have already done in the SVD question.
        Again, you should read the documentation to get the shapes and orientations that you want.

        However, this time, you'll first need to center the data
        by subtracting the mean of each feature from each point.

        Args:
            X: np.ndarray(N, D), each row represents a data point with D features
        Return:
            U: np.ndarray(N, min(N,D))
            S: np.ndarray(min(N,D))
            Vt: np.ndarray(min(N,D), D)
        Note:
            Sigma is a diagonal matrix, but there's no need to store all of those zeros.
            The return shape is 2d, 1 for the channels, 1 for a simple array of the singular values.
        """
        raise NotImplementedError

    def compute_principal_components(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Call your svd function to obtain the principal components (the right singular vectors of SVD).
        Then, pick the top k principal components that capture the most variance.

        Args:
            X: np.ndarray(N, D), each row represents a data point with D features
            k: int, the number of principal components to retain.
        Return:
            components: (k, D) numpy array where each row represents a principal component.
        """
        raise NotImplementedError

    def project(self, data: np.ndarray, components: np.ndarray) -> np.ndarray:
        """
        Given a dataset and a collection of principal components,
        project each sample in the dataset onto the principal components.
        This transforms the data from the original X-space to the reduced Z-space.

        Recall that the principal components were built off of centered data.
        This has functional usage, since you can't be certain that
        your test data will have the same distribution as your train data.
        Thus, again, subtract the mean before projecting.


        Args:
            data: np.ndarray(N, D), where each row is a data sample
            components: np.ndarray(k, D), where each row represents a principal component
        Return:
            projection: np.ndarray(N, k), the data represented in the reduced Z-space
        Note:
            While we won't check for looping, you may experience timeouts for slow solutions.
        """
        raise NotImplementedError
