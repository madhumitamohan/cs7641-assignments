import unittest
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from audiocompression import AudioCompression
from cross_validation import CrossValidation
from pca_reduction import PCAReduction
from feature_reduction import FeatureReduction
from logistic_regression import LogisticRegression
from regression import *
from smote import SMOTE
from utilities.local_tests_folder.ac_test import AC_Test
from utilities.local_tests_folder.ef_test import EF_Test
from utilities.local_tests_folder.xv_test import XV_Test
from utilities.local_tests_folder.feature_reduction_test import FeatureReduction_Test
from utilities.local_tests_folder.lr_test import LogisticRegression_Test
from utilities.local_tests_folder.regression_test import Regression_Test
from utilities.local_tests_folder.smote_test import SMOTE_Test


def print_success_message(msg):
    print(f'UnitTest passed successfully for "{msg}"!')


class TestPCAReduction(unittest.TestCase):
    """
    Tests for Q2: PCA Reduction
    """

    def test_svd(self):
        """
        Test correct implementation of SVD calculation for PCA
        """
        pca = PCAReduction()
        test_ef = EF_Test()
        try:
            U, S, V = pca.svd(test_ef.dataset)
        except MemoryError:
            self.assertTrue(False, "Your solution is allocating too much memory.")
        self.assertEqual(U.shape, test_ef.shape_of_U, "Shape of U is incorrect")
        self.assertEqual(S.shape, test_ef.shape_of_S, "Shape of S is incorrect")
        self.assertEqual(V.shape, test_ef.shape_of_V, "Shape of V is incorrect")
        reconstructed = U @ np.diag(S) @ V
        self.assertTrue(
            np.allclose(
                reconstructed, test_ef.dataset - np.mean(test_ef.dataset, axis=0)
            ),
            "Your SVD does not reconstruct the centered data. Did you remember to center the data?",
        )
        success_msg = "SVD calculation"
        print_success_message(success_msg)

    def test_compute_principal_components(self):
        pca = PCAReduction()
        test_ef = EF_Test()
        components = pca.compute_principal_components(test_ef.dataset, 2)
        self.assertEqual(
            components.shape,
            test_ef.principal_components_shape,
            "Principal components matrix shape is incorrect",
        )
        self.assertTrue(
            np.allclose(np.sum(components), test_ef.principal_components_sum),
            "Principal components are incorrect",
        )
        success_msg = "Principal Components"
        print_success_message(success_msg)

    def test_project(self):
        pca = PCAReduction()
        test_ef = EF_Test()
        test_components = test_ef.dataset[0:2]
        test_data = test_ef.dataset[2:5]
        projections = pca.project(test_data, test_components)
        self.assertEqual(
            projections.shape,
            (3, 2),
            f"Projection matrix shape is incorrect. Expected {3, 2} got {projections.shape}",
        )
        self.assertTrue(
            np.allclose(projections, test_ef.projections, rtol=0.01),
            msg="The projection is incorrect",
        )
        success_msg = "Projection"
        print_success_message(success_msg)


class TestAudioCompression(unittest.TestCase):
    """
    Tests for Audio Compression
    """

    def test_svd_stereo(self):
        """
        Test correct implementation of SVD calculation for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        U, S, Vt = ac.svd(test_ac.base_data)
        self.assertTrue(np.allclose(np.absolute(U), np.absolute(test_ac.U)))
        self.assertTrue(np.allclose(S, test_ac.S))
        self.assertTrue(np.allclose(np.absolute(Vt), np.absolute(test_ac.Vt)))
        success_msg = "SVD calculation - stereo audio"
        print_success_message(success_msg)

    def test_compress_stereo(self):
        """
        Test correct implementation of audio compression for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        U, S, Vt = test_ac.U, test_ac.S, test_ac.Vt
        U_c, S_c, Vt_c = ac.compress(U, S, Vt, 2)
        self.assertTrue(np.allclose(np.absolute(U_c), np.absolute(test_ac.U_c)))
        self.assertTrue(np.allclose(S, test_ac.S))
        self.assertTrue(np.allclose(np.absolute(Vt), np.absolute(test_ac.Vt)))
        success_msg = "Audio compression - stereo audio"
        print_success_message(success_msg)

    def test_rebuild_svd_stereo(self):
        """
        Test correct implementation of SVD reconstruction for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        X_approx = ac.rebuild_svd(test_ac.U_c, test_ac.S_c, test_ac.Vt_c)
        self.assertTrue(np.allclose(X_approx, test_ac.X_approx))
        success_msg = "SVD reconstruction - stereo audio"
        print_success_message(success_msg)

    def test_recovered_variance_proportion_stereo(self):
        """
        Test correct implementation of recovered variance proportion calculation for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        rvp = ac.recovered_variance_proportion(test_ac.S, 2)
        self.assertTrue(np.allclose(rvp, test_ac.rvp))
        success_msg = "Recovered variance proportion - stereo audio"
        print_success_message(success_msg)


class TestCrossValidation(unittest.TestCase):
    """
    Tests for Hyperparameter Search
    """

    def test_check_data_leakage(self):
        """
        Test correct implementation of data leakage check between training and validation sets.
        """
        xv = CrossValidation()
        test_xv = XV_Test()
        print(f"Testing with {test_xv.data.shape[0]} unique samples.")
        assert xv.check_data_leakage(test_xv.train_data, test_xv.test_data) == False, (
            "Check Data Leakage Found leakage in distinct sets."
        )
        assert xv.check_data_leakage(test_xv.train_data, test_xv.train_data) == True, (
            "Check Data Leakage Failed to find leakage in identical sets."
        )
        assert xv.check_data_leakage(test_xv.train_data, test_xv.leaked_test) == True, (
            "Check Data Leakage Failed to find leakage in overlapping sets."
        )
        success_msg = "Check data leakage"
        print_success_message(success_msg)


class TestRegression(unittest.TestCase):
    """
    Tests for Regression
    """

    def setUp(self):
        """
        Run before each test.
        """
        housing = fetch_california_housing()
        data, targets = housing.data, housing.target
        targets = targets[:, np.newaxis]
        (self.train_data, self.test_data, self.train_targets, self.test_targets) = (
            train_test_split(
                data, targets, test_size=0.1, random_state=10, shuffle=True
            )
        )

    def test_add_bias_term(self):
        N, D = 20, 4
        np.random.seed(20)
        X = np.random.randn(N, D)
        X_bias = BaseRegressor._add_bias_term(X)
        self.assertEqual(
            X_bias.shape,
            (N, D + 1),
            msg=f"Expected shape: {N, D + 1}, got: {X_bias.shape}.",
        )
        np.testing.assert_array_equal(
            X_bias[:, 0],
            np.ones(N),
            err_msg="Your bias column of the augmented matrix should be all 1s.",
        )
        np.testing.assert_array_equal(
            X_bias[:, 1:],
            X,
            err_msg="Your feature columns after the bias term do not match the original input data.",
        )
        success_msg = "add bias term"
        print_success_message(success_msg)

    def test_rmse(self):
        N = 100
        np.random.seed(10)
        y = np.random.randn(N, 1)
        y_pred = y + np.random.randn(N, 1) * 0.5
        my_rmse = BaseRegressor.rmse(y_pred, y)
        sklearn_rmse = np.sqrt(mean_squared_error(y, y_pred))
        self.assertAlmostEqual(
            my_rmse,
            sklearn_rmse,
            places=5,
            msg=f"Your RMSE ({my_rmse}) does not match the sklearn's RMSE ({sklearn_rmse}).",
        )
        success_msg = "RMSE"
        print_success_message(success_msg)

    def test_linear_closed(self):
        self.setUp()
        reg = ClosedFormRegressor()
        reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        print("Test RMSE:", my_rmse)
        from sklearn.linear_model import LinearRegression

        sk_reg = LinearRegression(fit_intercept=True)
        sk_reg.fit(self.train_data, self.train_targets)
        sk_ypred = sk_reg.predict(self.test_data)
        sk_rmse = reg.rmse(sk_ypred, self.test_targets)
        self.assertAlmostEqual(
            my_rmse,
            sk_rmse,
            places=5,
            msg="sklearn has a LinearRegression class (which should be the exact same) and it performs differently than your implementation.",
        )

    def test_linear_gd(self):
        self.setUp()
        reg = GDRegressor(epochs=20)
        train_rmse = reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        plt.plot(train_rmse)
        plt.xlabel("epochs")
        plt.xticks(list(range(20)))
        plt.ylabel("Train RMSE")
        plt.title("Linear GD (should decrease over 20 epochs)")
        plt.show()
        print("Test RMSE:", my_rmse)
        self.assertLess(
            train_rmse[19], train_rmse[0], "Your train RMSE should be decreasing."
        )

    def test_linear_sgd(self):
        self.setUp()
        reg = SGDRegressor(epochs=20)
        train_rmse = reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        plt.plot(train_rmse)
        plt.xlabel("epochs")
        plt.xticks(list(range(20)))
        plt.ylabel("Train RMSE")
        plt.title("Linear SGD (should decrease over 20 epochs)")
        plt.show()
        print("Test RMSE:", my_rmse)
        self.assertLess(
            train_rmse[19], train_rmse[0], "Your train RMSE should be decreasing."
        )

    def test_linear_mbgd(self):
        self.setUp()
        reg = MBGDRegressor(epochs=20)
        train_rmse = reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        plt.plot(train_rmse)
        plt.xlabel("epochs")
        plt.xticks(list(range(20)))
        plt.ylabel("Train RMSE")
        plt.title("Linear MBGD (should decrease over 20 epochs)")
        plt.show()
        print("Test RMSE:", my_rmse)
        self.assertLess(
            train_rmse[19], train_rmse[0], "Your train RMSE should be decreasing."
        )

    def test_ridge_closed(self):
        self.setUp()
        reg = ClosedFormRidgeRegressor(C=1.0)
        reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        print("Test RMSE:", my_rmse)
        from sklearn.linear_model import Ridge

        sk_reg = Ridge(alpha=1.0)
        sk_reg.fit(self.train_data, self.train_targets)
        sk_ypred = sk_reg.predict(self.test_data)
        sk_rmse = reg.rmse(sk_ypred[:, np.newaxis], self.test_targets)
        self.assertAlmostEqual(
            my_rmse,
            sk_rmse,
            places=5,
            msg="sklearn has a Ridge class (which should be the exact same) and it performs differently than your implementation.",
        )

    def test_ridge_gd(self):
        self.setUp()
        reg = GDRidgeRegressor(epochs=20)
        train_rmse = reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        plt.plot(train_rmse)
        plt.xlabel("epochs")
        plt.xticks(list(range(20)))
        plt.ylabel("Train RMSE")
        plt.title("Ridge GD (should decrease over 20 epochs)")
        plt.show()
        print("Test RMSE:", my_rmse)
        self.assertLess(
            train_rmse[19], train_rmse[0], "Your train RMSE should be decreasing."
        )

    def test_ridge_sgd(self):
        self.setUp()
        reg = SGDRidgeRegressor(epochs=20)
        train_rmse = reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        plt.plot(train_rmse)
        plt.xlabel("epochs")
        plt.xticks(list(range(20)))
        plt.ylabel("Train RMSE")
        plt.title("Linear SGD (should decrease over 20 epochs)")
        plt.show()
        print("Test RMSE:", my_rmse)
        self.assertLess(
            train_rmse[19], train_rmse[0], "Your train RMSE should be decreasing."
        )

    def test_ridge_mbgd(self):
        self.setUp()
        reg = MBGDRidgeRegressor(epochs=20)
        train_rmse = reg.fit(self.train_data, self.train_targets)
        ypred = reg.predict(self.test_data)
        my_rmse = reg.rmse(ypred, self.test_targets)
        plt.plot(train_rmse)
        plt.xlabel("epochs")
        plt.xticks(list(range(20)))
        plt.ylabel("Train RMSE")
        plt.title("Linear MBGD (should decrease over 20 epochs)")
        plt.show()
        print("Test RMSE:", my_rmse)
        self.assertLess(
            train_rmse[19], train_rmse[0], "Your train RMSE should be decreasing."
        )


class TestLogisticRegression(unittest.TestCase):
    """
    Tests for Logistic Regression
    """

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.sigmoid_result_slice), "sigmoid incorrect"
        )
        success_msg = "Logistic Regression sigmoid"
        print_success_message(success_msg)

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        self.assertTrue(
            result.shape == test_lr.s.shape, "sigmoid incorrect: check shape"
        )
        result_slice = result[:4, :4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.sigmoid_result_slice), "sigmoid incorrect"
        )
        success_msg = "Logistic Regression sigmoid"
        print_success_message(success_msg)

    def test_bias_augment(self):
        """
        Test correct implementation of bias_augment
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.bias_augment(test_lr.x)
        result_slice_sum = np.sum(result[:4, :4])
        self.assertTrue(
            np.allclose(result_slice_sum, test_lr.bias_augment_slice_sum),
            "bias_augment incorrect",
        )
        success_msg = "Logistic Regression bias_augment"
        print_success_message(success_msg)

    def test_predict_probs(self):
        """
        Test correct implementation of predict_probs
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_probs(test_lr.x_aug, test_lr.theta)
        self.assertTrue(result.ndim == 2, "predict_probs incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.x_aug.shape[0],
            "predict_probs incorrect: check shape",
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.predict_probs_result_slice),
            "predict_probs incorrect",
        )
        success_msg = "Logistic Regression predict_probs"
        print_success_message(success_msg)

    def test_predict_labels(self):
        """
        Test correct implementation of predict_labels
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_labels(test_lr.h_x, test_lr.threshold)
        self.assertTrue(result.ndim == 2, "predict_labels incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.h_x.shape[0],
            "predict_labels incorrect: check shape",
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.predict_labels_result_slice),
            "predict_labels incorrect",
        )
        success_msg = "Logistic Regression predict_labels"
        print_success_message(success_msg)

    def test_loss(self):
        """
        Test correct implementation of loss
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.loss(test_lr.y, test_lr.h_x)
        self.assertAlmostEqual(result, test_lr.loss_result, msg="loss incorrect")
        success_msg = "Logistic Regression loss"
        print_success_message(success_msg)

    def test_gradient(self):
        """
        Test correct implementation of gradient
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.gradient(test_lr.x_aug, test_lr.y, test_lr.h_x)
        self.assertTrue(result.ndim == 2, "gradient incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.x_aug.shape[1], "gradient incorrect: check shape"
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.gradient_result_slice),
            "gradient incorrect",
        )
        success_msg = "Logistic Regression gradient"
        print_success_message(success_msg)

    def test_accuracy(self):
        """
        Test correct implementation of accuracy
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.accuracy(test_lr.y, test_lr.y_hat)
        self.assertAlmostEqual(result, test_lr.accuracy_result, "accuracy incorrect")
        success_msg = "Logistic Regression accuracy"
        print_success_message(success_msg)

    def test_evaluate(self):
        """
        Test correct implementation of evaluate
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.evaluate(test_lr.x, test_lr.y, test_lr.theta, test_lr.threshold)
        self.assertAlmostEqual(
            result[0], test_lr.evaluate_result[0], msg="evaluate incorrect"
        )
        self.assertAlmostEqual(
            result[1], test_lr.evaluate_result[1], msg="evaluate incorrect"
        )
        success_msg = "Logistic Regression evaluate"
        print_success_message(success_msg)

    def test_fit(self):
        """
        Test correct implementation of fit
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.fit(
            test_lr.x,
            test_lr.y,
            test_lr.x,
            test_lr.y,
            test_lr.lr,
            test_lr.epochs,
            test_lr.threshold,
        )
        self.assertTrue(result.ndim == 2, "fit incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.theta.shape[0], "fit incorrect: check shape"
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.fit_result_slice), "fit incorrect"
        )
        success_msg = "Logistic Regression fit"
        print_success_message(success_msg)


class TestFeatureReduction(unittest.TestCase):
    """
    Tests for Feature Reduction
    """

    def test_forward_selection(self):
        fr = FeatureReduction()
        test_fr = FeatureReduction_Test()
        student_features = fr.forward_selection(
            test_fr.X, test_fr.y, test_fr.significance_level
        )
        correct_features = test_fr.correct_forward
        val = set(student_features) == set(correct_features)
        error_msg = """Your forward selection function did not yield the correct features.
        Common issues with this function are:
        (1) Not using the given regression function
        (2) Not adding a bias term
        (3) Not using the significance level properly
        (4) The order in which the features are removed is incorrect"""
        self.assertTrue(val, error_msg)
        success_msg = "Forward Selection"
        print_success_message(success_msg)


class TestSMOTE(unittest.TestCase):
    def test_simple_confusion_matrix(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        true, pred = sm_data.simple_classification_vectors
        student_conf = sm.generate_confusion_matrix(y_true=true, y_pred=pred)
        correct_conf = sm_data.simple_conf_matrix
        self.assertTrue(
            student_conf.shape == correct_conf.shape,
            f"The confusion matrix should be a square matrix (u, u), where u is the number of unique labels present. Expected {correct_conf.shape}, got {student_conf.shape}",
        )
        self.assertTrue(
            isinstance(student_conf[0, 0], (int, np.integer)),
            f"The entries of a confusion matrix should be integers. Expected int-like, got {type(student_conf[0, 0])}. Try setting the dtype of your numpy array.",
        )
        self.assertFalse(
            np.array_equal(student_conf, correct_conf.T),
            'You have the axes inverted conceptually. Make sure to put "true" and "predicted" on the left and top respectively.',
        )
        self.assertTrue(
            np.array_equal(student_conf, correct_conf),
            "Confusion matrix incorrectly calculated.",
        )
        print_success_message("simple confusion matrix")

    def test_complex_confusion_matrix(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        true, pred = sm_data.multiclass_classification_vectors
        student_conf = sm.generate_confusion_matrix(y_true=true, y_pred=pred)
        correct_conf = sm_data.multiclass_conf_matrix
        self.assertFalse(
            student_conf.shape == (2, 2),
            "In section 6.1, you should design your functions for a multiclass classification result. Your function might only be supporting a binary classification.",
        )
        self.assertTrue(
            student_conf.shape == correct_conf.shape,
            f"The confusion matrix should be a square matrix (u, u), where u is the number of unique labels present in either vector. Expected (4,4), got {student_conf.shape}.",
        )
        self.assertTrue(
            isinstance(student_conf[0, 0], (int, np.integer)),
            f"The entries of a confusion matrix should be integers. Expected int-like, got {type(student_conf[0, 0])}. Try setting the dtype of your numpy array.",
        )
        self.assertFalse(
            np.array_equal(student_conf, correct_conf.T),
            'You have the axes inverted conceptually. Make sure to put "true" and "predicted" on the left and top respectively.',
        )
        self.assertTrue(
            np.array_equal(student_conf, correct_conf),
            "Confusion matrix incorrectly calculated.",
        )
        print_success_message("multiclass confusion matrix")

    def test_interpolate(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        start, end = sm_data.interpolation_vectors
        coeff = sm_data.inter_coeff
        student_point = np.array(sm.interpolate(start, end, coeff))
        correct_point = np.array(sm_data.correct_point)
        flipped_point = np.array(sm_data.flipped_coeff)
        self.assertTrue(
            student_point.shape == correct_point.shape,
            f"The interpolated point should belong to the same dimension space as the input points. Expected {correct_point.shape}, got {student_point.shape}.",
        )
        self.assertTrue(
            isinstance(student_point[0], (float, np.floating)),
            f"The interpolated point should be an array of floats. Expected float-like, got {type(student_point[0])}. Try setting the dtype of your numpy array.",
        )
        self.assertFalse(
            np.allclose(student_point, flipped_point),
            "You flipped the start and the end, or you did the interpolation calculation backwards. You're close!",
        )
        self.assertTrue(
            np.allclose(student_point, correct_point),
            f"""Your calculation of the interpolation was incorrect. Expected
{correct_point}
 got
{student_point}""",
        )
        print_success_message("interpolation")

    def test_knn(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        points = sm_data.knn_points
        student_nns = sm.k_nearest_neighbors(points, 3)
        correct_nns = sm_data.three_nn
        self.assertTrue(
            student_nns.shape == correct_nns.shape,
            f"Each point should map to its k nearest neighbors, in the shape (N, k). Expected {correct_nns.shape}, got {student_nns.shape}.",
        )
        self.assertTrue(
            isinstance(student_nns[0, 0], (int, np.integer)),
            f"The entries of nearest neighbors array should be integer indices. Expected int-like, got {type(student_nns[0, 0])}. Try setting the dtype of your numpy array.",
        )
        is_correct = True
        for i in range(correct_nns.shape[0]):
            if set(correct_nns[i]) != set(student_nns[i]):
                is_correct = False
                break
        self.assertTrue(is_correct, f"One or more of your neighborhoods is incorrect.")
        print_success_message("k nearest neighbors")

    def test_smote(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        X = sm_data.X
        y = sm_data.y
        student_output = sm.smote(X, y, 3, [0, 1])
        self.assertTrue(
            len(student_output) == 2,
            f"You should return an iterable (e.g., Tuple) containing two items: synthetic X and synthetic y. Expected 2, got {len(student_output)}",
        )
        student_synth_X, student_synth_y = student_output
        self.assertTrue(
            student_synth_y.shape == (30,),
            f"Since |maj|=40 and |min|=10, you should have generated 30 points. Expected (30,), got {student_synth_y.shape}.",
        )
        self.assertTrue(
            student_synth_X.shape == (30, 4),
            f"Since |maj|=40 and |min|=10, you should have generated 30 points. Expected (30,4), got {student_synth_X.shape}.",
        )
        self.assertTrue(
            np.all(student_synth_y == 1),
            "The generated y values should be all the minority label: 1",
        )
        X_minority = X[y == 1]
        X_floor = np.min(X_minority, axis=0)
        X_ceil = np.max(X_minority, axis=0)
        is_correct = True
        for i in range(student_synth_X.shape[0]):
            for j in range(student_synth_X.shape[1]):
                if student_synth_X[i, j] < X_floor[j]:
                    is_correct = False
                    break
                if student_synth_X[i, j] > X_ceil[j]:
                    is_correct = False
                    break
        self.assertTrue(
            is_correct,
            f"You are generating points outside of the convex hull of the minority class. Either you are incorrectly applying interpolate or interpolate is incorrect.",
        )
        print_success_message("SMOTE")

    def test_threshold_eval(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        threshold = sm_data.threshold_roc
        y_true = sm_data.y_true_roc
        y_pred = sm_data.y_pred_roc
        fpr_tpr = sm.threshold_eval(y_true, y_pred, threshold)
        self.assertTrue(
            len(fpr_tpr) == 2,
            f"Threshold function should return a tuple of size 2, containing (FPR, TPR). But the function returned {len(fpr_tpr)} elements.",
        )
        self.assertTrue(
            isinstance(fpr_tpr[0], float) and isinstance(fpr_tpr[1], float),
            f"Threshold function should return a tupe of floats, containing (FPR, TPR). But the function returned types ({type(fpr_tpr[0])}) and ({type(fpr_tpr[1])}).",
        )
        self.assertTrue(
            np.isclose(fpr_tpr[0], sm_data.fpr_test),
            f"The False Positive Rate at threshold {threshold} is incorrect. Please check the order and the value of the FPR and TPR in the return tuple.",
        )
        self.assertTrue(
            np.isclose(fpr_tpr[1], sm_data.tpr_test),
            f"The True Positive Rate at threshold {threshold} is incorrect. Please check the order and the value of the FPR and TPR in the return tuple.",
        )
        print_success_message("Threshold evaluation")

    def test_integrate_curve(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        roc_points = sm_data.fpr_tpr_test
        auc_stu = sm.integrate_curve(roc_points)
        auc_exp = sm_data.auc
        self.assertTrue(
            isinstance(auc_stu, float),
            f"The AUC value should be a float, but the function returned is of type {type(auc_stu)}.",
        )
        self.assertTrue(
            np.isclose(auc_stu, auc_exp),
            f"The AUC value is incorrect. Expected {auc_exp}, got {auc_stu}.",
        )
        print_success_message("Integrating ROC Curve to get AUC")

    def test_generate_roc(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        y_true = sm_data.y_true_roc
        y_pred = sm_data.y_pred_roc
        roc_points = sm.generate_roc(y_true, y_pred)
        size_roc = len(sm_data.fpr_tpr_test)
        self.assertTrue(
            len(roc_points) == size_roc,
            f"The ROC curve should have {size_roc} points, ensure that all the thresholds values including the edge cases are considered. ",
        )
        self.assertTrue(
            np.array(roc_points).shape[1] == 2,
            f"The ROC Curve should be a 2D array of shape (n, 2), where n is the number of thresholds considered.",
        )
        self.assertTrue(
            np.allclose(roc_points, sm_data.fpr_tpr_test),
            f"The ROC curve points are incorrect. Please check the order and the values of fpr and tpr.",
        )
        print_success_message("Generate ROC Points")
