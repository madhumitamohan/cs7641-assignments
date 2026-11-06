import unittest
import numpy as np
from sklearn.datasets import load_wine


class EF_Test(unittest.TestCase):
    def __init__(self):
        wine_data = load_wine()
        self.dataset = wine_data.data
        N, D = self.dataset.shape
        k = min(N, D)
        self.shape_of_U = N, k
        self.shape_of_S = (k,)
        self.shape_of_V = k, D
        self.principal_components_shape = 2, D
        mean = np.mean(self.dataset, axis=0)
        centered = self.dataset - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.principal_components_sum = float(np.sum(Vt[:2, :]))
        test_components = self.dataset[0:2]
        test_data = self.dataset[2:5]
        test_mean = np.mean(test_data, axis=0)
        test_centered = test_data - test_mean
        self.projections = test_centered @ test_components.T


if __name__ == "__main__":
    unittest.main()
