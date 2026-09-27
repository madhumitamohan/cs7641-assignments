import time
import unittest
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gmm import *
from gmm import image_frob_dist
from hierarchical_clustering import *
from kmeans import *
from semisupervised import *
from dbscan import *
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, mean_squared_error
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from sklearn.metrics.pairwise import euclidean_distances as sk_distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import nan_euclidean_distances

dataset = pd.read_csv("./data/creditcard.csv")
X = dataset.iloc[:, dataset.columns != "Class"].to_numpy()
y_true = dataset["Class"].to_numpy().reshape(-1)


def print_success_message():
    print("UnitTest passed successfully!")


class KMeansTests(unittest.TestCase):
    def runTest(self):
        pass

    def test_pairwise_dist(self, pwd=pairwise_dist):
        x = X[:2, :2]
        y = X[3:6, 3:5]
        d1 = pairwise_dist(x, y)
        d2 = sk_distance(x, y)
        self.assertTrue(
            d1.shape == d2.shape,
            msg="Incorrect matrix shape. Expected: %s got: %s" % (d1.shape, d2.shape),
        )
        self.assertTrue(
            np.allclose(d1, d2, atol=0.0001), msg="Incorrect distance values"
        )
        print_success_message()

    def test_pairwise_speed(self, pwd=pairwise_dist):
        x = X[:2000, :3]
        y = X[:2000, 4:7]
        tic = time.perf_counter()
        times_pairwise = []
        times_sklearn = []
        for _ in range(10):
            _ = pairwise_dist(x, y)
            t1 = time.perf_counter() - tic
            tic = time.perf_counter()
            _ = sk_distance(x, y)
            t2 = time.perf_counter() - tic
            times_pairwise.append(t1)
            times_sklearn.append(t2)
        t1 = np.mean(times_pairwise)
        t2 = np.mean(times_sklearn)
        ratio = t1 / t2
        self.assertTrue(
            ratio < 10,
            msg="Your implementation is >10x slower than sklearn. Did you use broadcasting?",
        )
        print_success_message()

    def test_kmeans_loss(self, km=KMeans):
        points = X[:10, :5]
        cluster_idx = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        centers = np.array(
            [
                [-0.29934073, 0.22471242, -0.65337045, 0.31246784, -0.24212177],
                [-0.0697861, -0.54223103, 0.13456489, -0.65834621, -0.01247639],
            ]
        )
        kmeans = km(points, len(np.unique(cluster_idx)))
        kmeans.assignments = cluster_idx
        kmeans.centers = centers
        loss = kmeans.get_loss()
        self.assertTrue(
            np.isclose(loss, 58.51198579201331),
            msg="Expected: 58.51198579201331 got: %s" % loss,
        )
        print_success_message()

    def test_update_centers(self, km=KMeans):
        points = X[:10, :5]
        cluster_idx = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        old_centers = np.array(
            [
                [-0.29934073, 0.22471242, -0.65337045, 0.31246784, -0.24212177],
                [-0.0697861, -0.54223103, 0.13456489, -0.65834621, -0.01247639],
            ]
        )
        kmeans = km(points, len(np.unique(cluster_idx)))
        kmeans.assignments = cluster_idx
        kmeans.centers = old_centers
        new_centers = kmeans.update_centers()
        expected_centers = [
            [0.05615009, 0.76448958, 0.77586313, 0.26788289, 0.25860003],
            [-1.00093697, -0.070298, 1.42609331, 0.09097307, 0.44775155],
        ]
        self.assertTrue(
            np.allclose(new_centers, expected_centers, atol=0.0001),
            msg="Incorrect centers, check that means are computed correctly",
        )
        print_success_message()

    def test_init(self, km=KMeans):
        class_0_idx = np.where(y_true == 0)[0]
        class_1_idx = np.where(y_true == 1)[0]
        class_0_rand = np.random.choice(
            class_0_idx, 5500 - len(class_1_idx), replace=False
        )
        selected_idx = np.concatenate((class_0_rand, class_1_idx))
        x = X[selected_idx]
        y = y_true[selected_idx]
        x = x[:, ::-1]
        cluster_1_num = 3
        cluster_2_num = 4
        kmeans = km(x, 2)
        plt.figure(1)
        colors = ["#4EACC5", "#FF9C34"]
        for k, col in enumerate(colors):
            cluster_data = y == k
            plt.scatter(
                x[cluster_data, cluster_1_num],
                x[cluster_data, cluster_2_num],
                c=col,
                marker=".",
                s=20,
            )
        centers_init1 = kmeans.init_centers()
        try:
            centers_init2 = kmeans.kmpp_init()
            plt.scatter(
                centers_init1[:, cluster_1_num],
                centers_init1[:, cluster_2_num],
                c="k",
                s=50,
            )
            plt.scatter(
                centers_init2[:, cluster_1_num],
                centers_init2[:, cluster_2_num],
                c="r",
                s=50,
            )
            plt.legend(["1", "2", "random", "km++"])
            plt.title("K-Means++ Initialization")
        except NotImplementedError:
            plt.scatter(
                centers_init1[:, cluster_1_num],
                centers_init1[:, cluster_2_num],
                c="k",
                s=50,
            )
            plt.legend(["1", "2", "random"])
            plt.title("Without K-Means++ Initialization")
        finally:
            plt.xticks([])
            plt.yticks([])
            plt.show()

    def test_train(self, km=KMeans):
        np.random.seed(0)
        points = X[:50, :5]
        expected_centers = np.array(
            [
                [0.00841, 0.27861, 1.08147, 0.38133, -0.24516],
                [-1.3216, -1.0435, -0.18346, -0.12679, 2.60304],
            ]
        )
        expected_assignments = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        kmeans = km(points, 2, init="random", max_iters=10)
        centers, assignments, loss = kmeans.train()
        mse = mean_squared_error(centers, expected_centers)
        self.assertTrue(
            mse <= 0.0001, msg=f"Centers not updated correctly. MSE: {mse:.5f}."
        )
        self.assertTrue(
            np.array_equal(expected_assignments, assignments[:10]),
            msg=f"Wrong cluster assignments",
        )
        print_success_message()

    def test_silhouette_score(self):
        data = X[6325:6337]
        labels = np.asarray([0, 1, 2, 3, 2, 2, 1, 3, 0, 4, 4, 1])
        r1 = sklearn_silhouette_score(data, labels)
        r2 = silhouette_score(data, labels)
        print("Expected value: ", r1)
        print("Your value: ", r2, "\n")
        self.assertTrue(
            np.allclose(r1, r2, atol=0.0001),
            msg="Incorrect silhouette-score measure calculation",
        )
        print_success_message()

    def test_adjusted_rand_statistic(self):
        xPredicted = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
        xGroundTruth = y_true[6325:6337].tolist()
        r1 = adjusted_rand_score(xGroundTruth, xPredicted)
        r2 = adjusted_rand_statistic(xGroundTruth, xPredicted)
        print("Expected value: ", r1)
        print("Your value: ", r2, "\n")
        self.assertTrue(
            np.allclose(r1, r2, atol=0.0001),
            msg="Incorrect rand statistic measure calculation",
        )
        print_success_message()


class GMMTests(unittest.TestCase):
    def __init__(self) -> None:
        super().__init__()
        np.random.seed(5)
        self.data = np.array(
            [
                [0.44122749, -0.33087015, 2.43077119, -0.25209213],
                [0.10960984, 1.58248112, -0.9092324, -0.59163666],
                [0.18760323, -0.32986996, -1.19276461, -0.20487651],
                [-0.35882895, 0.6034716, -1.66478853, -0.70017904],
                [1.15139101, 1.85733101, -1.51117956, 0.64484751],
            ]
        )
        self.points = np.array(
            [
                [-0.98060789, -0.85685315, -0.87187918],
                [-0.42250793, 0.99643983, 0.71242127],
                [0.05914424, -0.36331088, 0.00328884],
                [-0.10593044, 0.79305332, -0.63157163],
                [-0.00619491, -0.10106761, -0.05230815],
                [0.24921766, 0.19766009, 1.33484857],
                [-0.08687561, 1.56153229, -0.30585302],
                [-0.47773142, 0.10073819, 0.35543847],
                [0.26961241, 1.29196338, 1.13934298],
                [0.4944404, -0.33633626, -0.10061435],
                [1.41339802, 0.22125412, -1.31077313],
                [-0.68956523, -0.57751323, 1.15220477],
                [-0.10716398, 2.26010677, 0.65661947],
                [0.12480683, -0.43570392, 0.97217931],
                [-0.24071114, -0.82412345, 0.56813272],
            ]
        )
        self.mu = np.array(
            [
                [-0.69166075, -0.39675353, -0.6871727],
                [0.04221375, 0.58281521, -1.10061918],
                [1.62434536, -0.61175641, -0.52817175],
            ]
        )
        self.sigma = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        self.pi = np.ones(3) / 3

    def test_helper_functions(self):
        my_softmax = GMM.softmax(self.data)
        expected_softmax = np.array(
            [
                [0.10782656, 0.04982049, 0.78844897, 0.05390398],
                [0.16080479, 0.70138883, 0.05805256, 0.07975382],
                [0.39637071, 0.23624673, 0.0996817, 0.26770086],
                [0.21741805, 0.56913777, 0.05890126, 0.15454293],
                [0.27040961, 0.54778227, 0.01886611, 0.16294201],
            ]
        )
        self.assertTrue(
            np.allclose(expected_softmax, my_softmax),
            "Your softmax does not work within the expected range.",
        )
        my_logsumexp = GMM.logsumexp(self.data)
        expected_logsumexp = np.array(
            [2.66845878, 1.93717399, 1.11300859, 1.16710435, 2.4592084]
        )
        self.assertTrue(
            np.allclose(expected_logsumexp, my_logsumexp),
            "Your logsumexp does not work within the expected range.",
        )
        print_success_message()

    def test_full_stack(self):
        """
        This test may be oversensitive.
        """
        points = self.points
        mu = self.mu
        pi = self.pi
        sigma = self.sigma
        data = self.data
        sigma_grad = np.array(
            [
                [
                    [0.30792796, 0.07909229, -0.11016917],
                    [0.07909229, 0.86422655, 0.06975468],
                    [-0.11016917, 0.06975468, 0.63212106],
                ],
                [
                    [0.30792796, 0.07909229, -0.11016917],
                    [0.07909229, 0.86422655, 0.06975468],
                    [-0.11016917, 0.06975468, 0.63212106],
                ],
                [
                    [0.30792796, 0.07909229, -0.11016917],
                    [0.07909229, 0.86422655, 0.06975468],
                    [-0.11016917, 0.06975468, 0.63212106],
                ],
            ]
        )
        my_multinormalpdf = GMM.multinormalPDF(points, mu[0], sigma_grad[0])
        expected_multinormal_pdf = np.array(
            [
                0.12412461,
                0.0109187066,
                0.0283199962,
                0.0484688341,
                0.0396140713,
                0.00035509416,
                0.0115479475,
                0.0506551884,
                0.000337348839,
                0.00634219531,
                0.000120587696,
                0.00814168432,
                0.000576457373,
                0.00182882581,
                0.0149926366,
            ]
        )
        self.assertTrue(
            np.allclose(expected_multinormal_pdf, my_multinormalpdf),
            "Your multinormal pdf does not work within the expected range.",
        )
        sigma_now = sigma * 0.5
        my_ll_joint = GMM(points, 3)._ll_joint(pi, mu, sigma_now)
        expected_ll_joint = np.array(
            [
                [-3.14500571, -5.9868382, -9.77969574],
                [-6.78800137, -6.48987436, -11.13068168],
                [-3.85727081, -4.92976126, -5.6097372],
                [-4.57751893, -3.10185949, -7.79374338],
                [-3.7760537, -4.38470224, -5.9616179],
                [-8.14285688, -8.93840538, -8.83268311],
                [-7.16176025, -4.42191156, -10.51659372],
                [-4.19600894, -5.43855224, -8.52284947],
                [-9.92767754, -8.38773887, -11.05576292],
                [-4.570244, -4.86506515, -4.35105393],
                [-8.01779049, -4.87074452, -4.16657737],
                [-6.23169506, -9.77278544, -10.99472719],
                [-12.02202766, -8.73921549, -15.46516011],
                [-6.23729264, -8.15640353, -7.34637072],
                [-4.77749941, -7.65996291, -7.54112612],
            ]
        )
        self.assertTrue(
            np.allclose(my_ll_joint, expected_ll_joint),
            "Your _ll_joint does not work within the expected range.",
        )
        my_estep = GMM(points, 3)._E_step(pi, mu, sigma_now)
        expected_estep = np.array(
            [
                [0.94372325, 0.05503671, 0.00124004],
                [0.42366876, 0.57082286, 0.00550839],
                [0.65984771, 0.22577041, 0.11438188],
                [0.18470545, 0.80788672, 0.00740783],
                [0.60368247, 0.32845499, 0.06786254],
                [0.5120336, 0.23109797, 0.25686843],
                [0.06053431, 0.93735212, 0.00211357],
                [0.76813271, 0.22172086, 0.01014643],
                [0.16700188, 0.77894757, 0.05405055],
                [0.33447807, 0.24907403, 0.4164479],
                [0.01402184, 0.3262493, 0.65972887],
                [0.96383555, 0.0279336, 0.00823084],
                [0.0361238, 0.96272152, 0.00115468],
                [0.67723134, 0.09937515, 0.22339351],
                [0.8936077, 0.05003903, 0.05635326],
            ]
        )
        self.assertTrue(
            np.allclose(my_estep, expected_estep),
            "Your E Step does not work within the expected range.",
        )
        my_pi, my_mu, my_sigma = GMM(points, 3)._M_step(expected_estep)
        expected_pi = np.array([0.4828419, 0.39149886, 0.12565925])
        expected_mu = np.array(
            [
                [-0.26263543, -0.23026888, 0.37410807],
                [0.02946666, 0.96190945, 0.19697914],
                [0.64855787, -0.0282273, -0.12987796],
            ]
        )
        expected_sigma = np.array(
            [
                [
                    [0.18480413, 0.08971316, 0.08911991],
                    [0.08971316, 0.36907686, 0.08744919],
                    [0.08911991, 0.08744919, 0.48008412],
                ],
                [
                    [0.17533767, -0.0757907, -0.08561511],
                    [-0.0757907, 0.73833814, 0.16291358],
                    [-0.08561511, 0.16291358, 0.52631415],
                ],
                [
                    [0.35145756, 0.10609808, -0.51519293],
                    [0.10609808, 0.15893304, -0.08535478],
                    [-0.51519293, -0.08535478, 0.99893729],
                ],
            ]
        )
        self.assertTrue(
            np.allclose(my_pi, expected_pi),
            "pi; Your M Step does not work within the expected range (independent of E step).",
        )
        self.assertTrue(
            np.allclose(my_mu, expected_mu),
            "mu; Your M Step does not work within the expected range (independent of E step).",
        )
        self.assertTrue(
            np.allclose(my_sigma, expected_sigma),
            "sigma; Your M Step does not work within the expected range (independent of E step).",
        )
        print_success_message()

    def pseudo_test_image_compression(self):
        """
        This is a pseudo test because no criterion are actually tested.
        It's just printing stuff for you to diagnose by hand.
        The autograder tests compare you against invariants of proper implementations, not exactly similar performance anyways.
        """
        image = imageio.imread("./data/images/gmm_test_images/test-parrot.jpg")
        K_vals = [8, 8, 8, 8, 16, 16, 16, 16]
        max_iters = [5, 10, 15, 20, 5, 10, 15, 20]
        assert len(K_vals) == len(max_iters)
        N = len(K_vals)
        fig, axes = plt.subplots(N, 2, figsize=(8, 2.5 * N), constrained_layout=True)
        for i in range(N):
            axes[i, 0].imshow(image)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")
            reconstruction = cluster_pixels_gmm(image.copy(), K_vals[i], max_iters[i])
            distance = image_frob_dist(image, reconstruction)
            axes[i, 1].imshow(reconstruction.astype(int))
            axes[i, 1].set_title(
                f"""Reconstruction with {K_vals[i]} comps for {max_iters[i]} iters
Distance: {distance}""",
                fontsize=11,
            )
            axes[i, 1].axis("off")
        plt.show()
        print(
            "More iterations should monotonically improve the loss, watch the TDQM progress bars."
        )
        print(
            "More components should improve the reconstruction error, check the output plots."
        )
        print(
            "These are separate runs because our pipeline doesn't have checkpointing, so individual runs may be subject to bad fits, just as K-means can settle into a bad solution."
        )


class SemisupervisedTests(unittest.TestCase):
    def test_data_separating_methods(self):
        data = np.array(
            [
                [1.0, 2.0, 3.0, 1],
                [1.0, np.nan, 3.0, 1],
                [7.0, np.nan, 9.0, 0],
                [7.0, 8.0, 9.0, 0],
                [26.0, 27.0, 28.0, np.nan],
                [2.0, 3.0, 4.0, np.nan],
                [16.0, 17.0, 18.0, 1],
                [np.nan, 17.0, 18.0, 1],
                [11.0, 12.0, 13.0, np.nan],
                [22.0, 23.0, 24.0, 0],
                [np.nan, 23.0, 24.0, 0],
                [19.0, 20.0, 21.0, np.nan],
            ]
        )
        complete_answer = np.array(
            [
                [1.0, 2.0, 3.0, 1.0],
                [7.0, 8.0, 9.0, 0.0],
                [16.0, 17.0, 18.0, 1.0],
                [22.0, 23.0, 24.0, 0.0],
            ]
        )
        incomplete_answer = np.array(
            [
                [1.0, np.nan, 3.0, 1.0],
                [7.0, np.nan, 9.0, 0.0],
                [np.nan, 17.0, 18.0, 1.0],
                [np.nan, 23.0, 24.0, 0.0],
            ]
        )
        unlabeled_answer = np.array(
            [
                [26.0, 27.0, 28.0, np.nan],
                [2.0, 3.0, 4.0, np.nan],
                [11.0, 12.0, 13.0, np.nan],
                [19.0, 20.0, 21.0, np.nan],
            ]
        )
        my_complete = complete_(data)
        my_incomplete = incomplete_(data)
        my_unlabeled = unlabeled_(data)
        self.assertTrue(
            my_complete.shape == complete_answer.shape,
            msg="Expected %s as complete shape but got %s instead"
            % (complete_answer.shape, my_complete.shape),
        )
        self.assertTrue(
            np.all(np.isclose(my_complete, complete_answer, equal_nan=True)),
            msg="Incorrect complete_ method. Check for no NaN values",
        )
        self.assertTrue(
            my_incomplete.shape == incomplete_answer.shape,
            msg="Expected %s as incomplete shape but got %s instead"
            % (incomplete_answer.shape, my_incomplete.shape),
        )
        self.assertTrue(
            np.all(np.isclose(my_incomplete, incomplete_answer, equal_nan=True)),
            msg="Incorrect incomplete_ method. Check if only features have NaN values",
        )
        self.assertTrue(
            my_unlabeled.shape == unlabeled_answer.shape,
            msg="Expected %s as unlabeled shape but got %s instead"
            % (unlabeled_answer.shape, my_unlabeled.shape),
        )
        self.assertTrue(
            np.all(np.isclose(my_unlabeled, unlabeled_answer, equal_nan=True)),
            msg="Incorrect unlabeled_ method. Check if only lables have NaN values",
        )
        print_success_message()

    def test_pairwise_dist_missingness_aware(self):
        self.knn_cleaner = CleanData()
        x = np.array([[2.0, np.nan, 3.0], [6.0, np.nan, 9.0], [7.0, 11.0, np.nan]])
        y = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [3.0, 5.0, np.nan]])
        output_missingness_aware = self.knn_cleaner.pairwise_dist_missingness_aware(
            x, y
        )
        distances_sklearn = nan_euclidean_distances(x, y)
        self.assertTrue(
            np.all(
                np.isclose(output_missingness_aware, distances_sklearn, equal_nan=True)
            ),
            msg="Distances calculation for missingness aware are incorrect",
        )
        print_success_message()

    def test_cleandata(self):
        self.knn_cleaner = CleanData()
        complete_data = np.array(
            [
                [1.0, 2.0, 3.0, 1],
                [7.0, 8.0, 9.0, 0],
                [16.0, 17.0, 18.0, 1],
                [22.0, 23.0, 24.0, 0],
            ]
        )
        incomplete_data = np.array(
            [
                [1.0, np.nan, 3.0, 1],
                [7.0, np.nan, 9.0, 0],
                [np.nan, 17.0, 18.0, 1],
                [np.nan, 23.0, 24.0, 0],
            ]
        )
        correct_clean_data = np.array(
            [
                [1.0, 2.0, 3.0, 1.0],
                [7.0, 8.0, 9.0, 0.0],
                [16.0, 17.0, 18.0, 1.0],
                [22.0, 23.0, 24.0, 0.0],
                [19.0, 23.0, 24.0, 0.0],
                [7.0, 5.0, 9.0, 0.0],
                [19.0, 17.0, 18.0, 1.0],
                [1.0, 5.0, 3.0, 1.0],
            ]
        )
        clean_data = self.knn_cleaner(incomplete_data, complete_data, 2)
        self.assertTrue(
            clean_data.shape == correct_clean_data.shape,
            msg="Expected %s as clean data shape but got %s instead"
            % (correct_clean_data.shape, clean_data.shape),
        )
        if np.all(np.isclose(clean_data, correct_clean_data)):
            print_success_message()
            return
        clean_data_sorted = np.array(sorted([tuple(row) for row in clean_data]))
        correct_clean_data_sorted = np.array(
            sorted([tuple(row) for row in correct_clean_data])
        )
        self.assertTrue(
            np.all(np.allclose(clean_data_sorted, correct_clean_data_sorted)),
            msg="Incorrect implementation. Check if all NaN values are replaced correctly",
        )
        print_success_message()

    def test_median_clean_data(self):
        complete_data = np.array(
            [
                [1.0, 2.0, 3.0, 1],
                [7.0, 8.0, 15.0, 0],
                [16.0, 17.0, 18.0, 1],
                [22.0, 23.0, 24.0, 0],
                [1.0, np.nan, 3.0, 1],
                [7.0, 20.0, np.nan, 0],
                [np.nan, 17.0, 18.0, 1],
                [np.nan, 23.0, 24.0, 0],
            ]
        )
        correct_clean_data = np.array(
            [
                [1.0, 2.0, 3.0, 1.0],
                [7.0, 8.0, 15.0, 0.0],
                [16.0, 17.0, 18.0, 1.0],
                [22.0, 23.0, 24.0, 0.0],
                [1.0, 17.0, 3.0, 1.0],
                [7.0, 20.0, 18.0, 0.0],
                [7.0, 17.0, 18.0, 1.0],
                [7.0, 23, 24.0, 0.0],
            ]
        )
        clean_data = median_clean_data(complete_data)
        self.assertTrue(
            clean_data.shape == correct_clean_data.shape,
            msg="Expected %s as median method clean data shape but got %s instead"
            % (correct_clean_data.shape, clean_data.shape),
        )
        if np.all(np.isclose(clean_data, correct_clean_data)):
            print_success_message()
            return
        clean_data_sorted = np.array(sorted([tuple(row) for row in clean_data]))
        correct_clean_data_sorted = np.array(
            sorted([tuple(row) for row in correct_clean_data])
        )
        self.assertTrue(
            np.all(np.allclose(clean_data_sorted, correct_clean_data_sorted)),
            msg="Incorrect implementation. Check if each feature's median replaces all NaN values for the feature",
        )
        print_success_message()


class HierarchicalClusteringTests(unittest.TestCase):
    def __init__(self) -> None:
        super().__init__()
        self.data_1d = np.array([[0], [-1], [4], [5], [100]])
        self.data_2d = np.array([[3, 3], [8, 8], [3, 6], [8, 10]])

    def assert_array_equal(self, actual, correct, name):
        self.assertTrue(
            correct.shape == actual.shape,
            msg="Incorrect %s shape. Expected: %s got: %s"
            % (name, correct.shape, actual.shape),
        )
        self.assertTrue(
            np.array_equal(correct, actual),
            msg="Incorrect %s. Expected: %s got: %s" % (name, correct, actual),
        )

    def assert_array_allclose(self, actual, correct, name):
        self.assertTrue(
            correct.shape == actual.shape,
            msg="Incorrect %s shape. Expected: %s got: %s"
            % (name, correct.shape, actual.shape),
        )
        self.assertTrue(
            np.allclose(correct, actual),
            msg="%s not close enough to correct answer. Expected: %s got: %s"
            % (name, correct, actual),
        )

    def test_create_distance(self):
        correct_distances_1d = np.array(
            [
                [np.inf, 1, 4, 5, 100],
                [1, np.inf, 5, 6, 101],
                [4, 5, np.inf, 1, 96],
                [5, 6, 1, np.inf, 95],
                [100, 101, 96, 95, np.inf],
            ]
        )
        correct_cluster_ids_1d = np.array([0, 1, 2, 3, 4])
        hc = HierarchicalClustering(self.data_1d)
        my_distances_1d, my_cluster_ids_1d = hc.distances, hc.cluster_ids
        self.assert_array_equal(
            my_distances_1d, correct_distances_1d, "distances (1d case)"
        )
        self.assert_array_equal(
            my_cluster_ids_1d, correct_cluster_ids_1d, "cluster ids (1d case)"
        )
        correct_distances_2d = np.array(
            [
                [np.inf, 7.07106781, 3, 8.60232527],
                [7.07106781, np.inf, 5.38516481, 2],
                [3.0, 5.38516481, np.inf, 6.40312424],
                [8.60232527, 2, 6.40312424, np.inf],
            ]
        )
        correct_cluster_ids_2d = np.array([0, 1, 2, 3])
        hc = HierarchicalClustering(self.data_2d)
        my_distances_2d, my_cluster_ids_2d = hc.distances, hc.cluster_ids
        self.assert_array_allclose(
            my_distances_2d, correct_distances_2d, "distances (2d case)"
        )
        self.assert_array_equal(
            my_cluster_ids_2d, correct_cluster_ids_2d, "cluster ids (2d case)"
        )
        print_success_message()

    def test_iterate_1d(self):
        correct_current_iteration = 1
        correct_distances = np.array(
            [
                [np.inf, 4.0, 5.0, 100.0],
                [4.0, np.inf, 1.0, 96.0],
                [5.0, 1.0, np.inf, 95.0],
                [100.0, 96.0, 95.0, np.inf],
            ]
        )
        correct_cluster_ids = np.array([5, 2, 3, 4])
        correct_clustering = np.array(
            [[0, 1, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        correct_cluster_sizes = np.array([1, 1, 1, 1, 1, 2, 0, 0, 0])
        hc = HierarchicalClustering(self.data_1d)
        hc.iterate()
        self.assertTrue(
            hc.current_iteration == correct_current_iteration,
            msg="self.current_iteration should be 1",
        )
        self.assert_array_allclose(hc.distances, correct_distances, "distances")
        self.assert_array_equal(hc.cluster_ids, correct_cluster_ids, "cluster ids")
        self.assert_array_equal(hc.clustering, correct_clustering, "clustering")
        self.assert_array_equal(
            hc.cluster_sizes, correct_cluster_sizes, "cluster sizes"
        )
        print_success_message()

    def test_iterate_2d(self):
        correct_current_iteration = 1
        correct_distances = np.array(
            [
                [np.inf, 7.07106781, 3.0],
                [7.07106781, np.inf, 5.38516481],
                [3.0, 5.38516481, np.inf],
            ]
        )
        correct_cluster_ids = np.array([0, 4, 2])
        correct_clustering = np.array([[1, 3, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]])
        correct_cluster_sizes = np.array([1, 1, 1, 1, 2, 0, 0])
        hc = HierarchicalClustering(self.data_2d)
        hc.iterate()
        self.assertTrue(
            hc.current_iteration == correct_current_iteration,
            msg="self.current_iteration should be 1",
        )
        self.assert_array_allclose(
            hc.distances, correct_distances, "distances (2d case)"
        )
        self.assert_array_equal(
            hc.cluster_ids, correct_cluster_ids, "cluster ids (2d case)"
        )
        self.assert_array_equal(
            hc.clustering, correct_clustering, "clustering (2d case)"
        )
        self.assert_array_equal(
            hc.cluster_sizes, correct_cluster_sizes, "cluster sizes (2d case)"
        )
        print_success_message()

    def test_fit(self):
        correct_current_iteration = 3
        correct_distances = np.array([[np.inf]])
        correct_cluster_ids = np.array([6])
        correct_clustering = np.array(
            [[1.0, 3.0, 2.0, 2.0], [0.0, 2.0, 3.0, 2.0], [5.0, 4.0, 5.38516481, 4.0]]
        )
        correct_cluster_sizes = np.array([1, 1, 1, 1, 2, 2, 4])
        hc = HierarchicalClustering(self.data_2d)
        clustering = hc.fit()
        self.assertTrue(
            hc.current_iteration == correct_current_iteration,
            msg="self.current_iteration should be 1",
        )
        self.assert_array_allclose(hc.distances, correct_distances, "distances")
        self.assert_array_equal(hc.cluster_ids, correct_cluster_ids, "cluster ids")
        self.assert_array_allclose(clustering, correct_clustering, "clustering")
        self.assert_array_equal(
            hc.cluster_sizes, correct_cluster_sizes, "cluster sizes"
        )
        print_success_message()


class DBScanTests(unittest.TestCase):
    def test_region_query(self):
        self.eps = 0.2
        self.minPoints = 3
        self.x = np.array(
            [
                [4.44921468, 0.99067444],
                [0.96302552, 1.79996854],
                [4.28865161, 1.07127892],
                [0.93103109, 2.09965384],
                [0.87714783, 2.20467543],
                [4.31595483, 1.05237385],
                [0.82344716, 2.14171114],
                [4.20605954, 1.16302588],
                [4.34484718, 0.96594819],
                [0.81225409, 2.08657429],
            ]
        )
        self.dbscan = DBSCAN(eps=self.eps, minPts=self.minPoints, dataset=self.x)
        neighborhood = self.dbscan.regionQuery(8)
        correct_neighborhood = [0, 2, 5, 8]
        self.assertFalse(
            set(neighborhood) - set(correct_neighborhood),
            msg="Expected %s as neighbors but got %s instead"
            % (correct_neighborhood, neighborhood),
        )
        print_success_message()

    def test_expand_cluster(self):
        self.eps = 0.2
        self.minPoints = 3
        self.x = np.array(
            [
                [4.44921468, 0.99067444],
                [0.96302552, 1.79996854],
                [4.28865161, 1.07127892],
                [0.93103109, 2.09965384],
                [0.87714783, 2.20467543],
                [4.31595483, 1.05237385],
                [0.82344716, 2.14171114],
                [4.20605954, 1.16302588],
                [4.34484718, 0.96594819],
                [0.81225409, 2.08657429],
            ]
        )
        self.dbscan = DBSCAN(eps=self.eps, minPts=self.minPoints, dataset=self.x)
        cluster_idx = np.zeros(len(self.x)) - 1
        visited_indices = {0}
        neighbor_indices = np.array([0, 2, 5, 8], dtype=int)
        self.dbscan.expandCluster(0, neighbor_indices, 0, cluster_idx, visited_indices)
        correct_cluster_idx = [0.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0]
        correct_visited_indices = {0, 2, 5, 7, 8}
        self.assertTrue(
            np.array_equal(cluster_idx, correct_cluster_idx),
            msg="You should've visited {0, 2, 5, 7, 8}, did you catch'em all?",
        )
        print_success_message()
