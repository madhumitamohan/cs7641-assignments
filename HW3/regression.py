from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseRegressor(ABC):
    """
    Anything that inherits ABC is an "Abstract Base Class."
    In an abstract class, you may have abstract methods.
    Abstract methods must be overriden by any classes that inherit from the ABC.

    This class acts as a backbone for other regressor classes.
    Any utils that are common to any regressor are implemented here.
    Any utils that are specific to how linear regression is done will be set as an abstract method.
    """

    def __init__(self, use_bias_term: bool = True):
        """
        There are very few truly shared hyperparameters.

        Args:
            use_bias_term: bool
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
        """
        self.weights = None
        self.use_bias_term = use_bias_term

    @staticmethod
    def _add_bias_term(X: np.ndarray) -> np.ndarray:
        """
        Linear regression can't model intercepts unless you give it a column full of ones.
        This private function simply prepends a column of ones to the given data.

        Args:
            X: np.ndarray(N, D)
        Return:
            augmented_X: np.ndarray(N, 1+D), where the first feature is all ones
        """
        bias_array = np.ones((X.shape[0], 1))
        x_aug = np.concatenate((bias_array, X), axis=1)
        return x_aug

    @staticmethod
    def rmse(y_predicted: np.ndarray, y_actual: np.ndarray) -> float:
        """
        Calculate the root mean square error between predicted values and the ground truth.

        Args:
            y_predicted: np.ndarray(N, 1), the predicted values
            y_actual: np.ndarray(N, 1), the ground truth values
        Return:
            rmse: float
        """
        return np.sqrt(np.mean((y_predicted - y_actual) ** 2))

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        THIS IS AN ABSTRACT METHOD; it will be implemented in the subclasses, not here.
        The real crux of the problem is finding out the optimal weights.
        There's many ways to do it, and you'll implement those by overriding this function in subclasses.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights
        Return:
            nothing
        """
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Using the weights from the most previous fit,
        predict the ground truth value for the given test data.

        Args:
            X_test: np.ndarray(N, D), data for which we want to estimate the corresponding y value
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        if self.weights is None:
            raise RuntimeError("Model has not yet been fit.")
        if self.use_bias_term:
            X_test = self._add_bias_term(X_test)
        return X_test @ self.weights


class ClosedFormRegressor(BaseRegressor):
    def __init__(self, use_bias_term: bool = True):
        """
        Args:
            use_bias_term: bool
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
        """
        super().__init__(use_bias_term)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the weights by using the closed-form pseudo-inverse approach.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights -> np.ndarray(D,1)
        Return:
            nothing
        """
        if self.use_bias_term:
            X = self._add_bias_term(X)
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y


class GDRegressor(BaseRegressor):
    def __init__(
        self, use_bias_term: bool = True, epochs: int = 5, learning_rate: float = 1e-07
    ):
        """
        Args:
            use_bias_term: bool
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
            self.epochs: int, the max number of times over which all data is passed to fit the weights
            self.lr: float, the term by which the gradient is multiplied to make small steps
        """
        super().__init__(use_bias_term)
        self.epochs = epochs
        self.lr = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the weights by using whole dataset gradient descent.
        Initialize weights to 0, then use self.epochs and self.lr for the iterative steps.

        After each epoch, you should calculate the training RMSE and append that to a list to be returned.
        Recall you have the utilities from BaseRegressor inherited.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights -> np.ndarray(D,1)
        Return:
            training_loss_per_epoch: list[float], the RMSE of the training data (X) per epoch
        """

        if self.use_bias_term:
            X = self._add_bias_term(X)

        self.weights = np.zeros((X.shape[1], 1))
        training_loss_per_epoch = []


        for epoch in range(self.epochs):


            gradient = (X.T @ (X @ self.weights - y))/ X.shape[0]
            self.weights = self.weights - (self.lr * gradient)

            
            err = self.rmse(X @ self.weights, y)
            training_loss_per_epoch.append(err)

        return training_loss_per_epoch
            


class SGDRegressor(BaseRegressor):
    def __init__(
        self, use_bias_term: bool = True, epochs: int = 5, learning_rate: float = 1e-07
    ):
        """
        Args:
            use_bias_term: bool
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
            self.epochs: int, the max number of times over which all data is passed to fit the weights
            self.lr: float, the term by which the gradient is multiplied to make small steps
        """
        super().__init__(use_bias_term)
        self.epochs = epochs
        self.lr = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the weights by using stochastic gradient descent.
        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        Basically, assume the data has already been shuffled and just iterate.
        Initialize weights to 0, then use self.epochs and self.lr for the iterative steps.

        After each epoch, you should calculate the training RMSE and append that to a list to be returned.
        Recall you have the utilities from BaseRegressor inherited.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights -> np.ndarray(D,1)
        Return:
            training_loss_per_epoch: list[float], the RMSE of the training data (X) per epoch
        """
        if self.use_bias_term:
            X = self._add_bias_term(X)

        self.weights = np.zeros((X.shape[1], 1))
        training_loss_per_epoch = []


        for epoch in range(self.epochs):
            

            for index in range(X.shape[0]):
                x_i = X[index:index+1]
                y_i = y[index:index+1]

                gradient = (x_i.T @ (x_i @ self.weights - y_i))
                self.weights = self.weights - (self.lr * gradient)

            
            err = self.rmse(X @ self.weights, y)
            training_loss_per_epoch.append(err)

        return training_loss_per_epoch


class MBGDRegressor(BaseRegressor):
    def __init__(
        self,
        use_bias_term: bool = True,
        batch_size: int = 5,
        epochs: int = 100,
        learning_rate: float = 1e-07,
    ):
        """
        Args:
            use_bias_term: bool
            batch_size: int, size of each batch
            epochs: int, the max number of times over which all data is passed to fit the weights
            learning_rate: float, the term by which the gradient is multiplied to make small steps
        Fields:
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
            self.batch_size: int, size of each batch
            self.epochs: int, the max number of times over which all data is passed to fit the weights
            self.lr: float, the term by which the gradient is multiplied to make small steps
        """
        super().__init__(use_bias_term)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the weights by using mini-batch gradient descent.
        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        Basically, assume the data has already been shuffled and just iterate.
        Initialize weights to 0, then use self.batch_size, self.epochs, and self.lr for the iterative steps.

        After each epoch, you should calculate the training RMSE and append that to a list to be returned.
        Recall you have the utilities from BaseRegressor inherited.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights -> np.ndarray(D,1)
        Return:
            training_loss_per_epoch: list[float], the RMSE of the training data (X) per epoch
        """
        if self.use_bias_term:
            X = self._add_bias_term(X)

        self.weights = np.zeros((X.shape[1], 1))
        training_loss_per_epoch = []


        for epoch in range(self.epochs):
            

            for index in range(0, X.shape[0], self.batch_size):
                x_i = X[index:index+self.batch_size]
                y_i = y[index:index+self.batch_size]

                gradient = (x_i.T @ (x_i @ self.weights - y_i))/self.batch_size
                self.weights = self.weights - (self.lr * gradient)

            
            err = self.rmse(X @ self.weights, y)
            training_loss_per_epoch.append(err)

        return training_loss_per_epoch


class ClosedFormRidgeRegressor(BaseRegressor):
    def __init__(self, use_bias_term: bool = True, C: float = 1.0):
        """
        Args:
            use_bias_term: bool
            C: float
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
            self.C: float value, value of the ridge regularization constant
        """
        super().__init__(use_bias_term)
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the weights by using the closed-form ridge regression equation.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights
        Return:
            nothing
        """
        if self.use_bias_term:
            X = self._add_bias_term(X)
            
        D = X.shape[1]
        I = np.identity(D)
        if self.use_bias_term:
            I[0,0] = 0.0
        
        inverse_term = ((X.T @ X) + (self.C * I))
        inverse = np.linalg.inv(inverse_term)
        self.weights = (inverse @ X.T) @ y 


class GDRidgeRegressor(BaseRegressor):
    def __init__(
        self,
        use_bias_term: bool = True,
        epochs: int = 500,
        learning_rate: float = 1e-07,
        C: float = 1.0,
    ):
        """
        Args:
            use_bias_term: bool
            epochs: int, the max number of times over which all data is passed to fit the weights
            learning_rate: float, the term by which the gradient is multiplied to make small steps
            C: float
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
            self.epochs: int, the max number of times over which all data is passed to fit the weights
            self.lr: float, the term by which the gradient is multiplied to make small steps
            self.C: float value, value of regularization constant
        """
        super().__init__(use_bias_term)
        self.epochs = epochs
        self.lr = learning_rate
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a ridge regression model using gradient descent.

        Don't apply regularization to the bias term, if one exists.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights
        Return:
            training_loss_per_epoch: list[float], the RMSE of the training data (X) per epoch
        """
        if self.use_bias_term:
            X = self._add_bias_term(X)

        N = X.shape[0]
        self.weights = np.zeros((X.shape[1], 1))
        training_loss_per_epoch = []


        for epoch in range(self.epochs):

            reg_term = self.C * self.weights/N
            reg_term[0] = 0.0
            
            gradient = reg_term + (X.T @ (X @ self.weights - y))/ X.shape[0]
            self.weights = self.weights - (self.lr * gradient)

            
            err = self.rmse(X @ self.weights, y)
            training_loss_per_epoch.append(err)

        return training_loss_per_epoch


class SGDRidgeRegressor(BaseRegressor):
    def __init__(
        self,
        use_bias_term: bool = True,
        epochs: int = 20,
        learning_rate: float = 1e-08,
        C: float = 1.0,
    ):
        """
        Args:
            use_bias_term: bool
            epochs: int, the max number of times over which all data is passed to fit the weights
            learning_rate: float, the term by which the gradient is multiplied to make small steps
            C: float, value of regularization constant
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
                for linear regression to be able to model intercepts, you need to add a column of ones
                when True, you must add a bias term to your data in fit and predict
            self.epochs: int, the max number of times over which all data is passed to fit the weights
            self.lr: float, the term by which the gradient is multiplied to make small steps
            self.C: float value, value of regularization constant
        """
        super().__init__(use_bias_term)
        self.epochs = epochs
        self.lr = learning_rate
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Fit the weights by using stochastic gradient descent with ridge regularization.
        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        Basically, assume the data has already been shuffled and just iterate.
        Initialize weights to 0, then use self.epochs and self.lr for the iterative steps.

        After each epoch, you should calculate the training RMSE and append that to a list to be returned.
        Recall you have the utilities from BaseRegressor inherited.

        Don't apply regularization to the bias term, if one exists.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights -> np.ndarray(D,1)
        Return:
            training_loss_per_epoch: list[float], the RMSE of the training data (X) per epoch
        """
        if self.use_bias_term:
            X = self._add_bias_term(X)

        self.weights = np.zeros((X.shape[1], 1))
        training_loss_per_epoch = []


        for epoch in range(self.epochs):
            

            for index in range(X.shape[0]):
                x_i = X[index:index+1]
                y_i = y[index:index+1]

                reg_term = self.C * self.weights
                reg_term[0] = 0.0

                gradient = reg_term + (x_i.T @ (x_i @ self.weights - y_i))
                self.weights = self.weights - (self.lr * gradient)

            
            err = self.rmse(X @ self.weights, y)
            training_loss_per_epoch.append(err)

        return training_loss_per_epoch

class MBGDRidgeRegressor(BaseRegressor):
    def __init__(
        self,
        use_bias_term: bool = True,
        batch_size: int = 5,
        epochs: int = 20,
        learning_rate: float = 1e-08,
        C: float = 1.0,
    ):
        """
        Args:
            use_bias_term: bool
            batch_size: int, size of each batch
            epochs: int, the max number of times over which all data is passed to fit the weights
            learning_rate: float, the term by which the gradient is multiplied to make small steps
            C: float, value of regularization constant
        Fields:
            self.weights: Optional[np.ndarray(D,1)]
            self.use_bias_term: bool, whether to use a bias term
            self.batch_size: int, size of each batch
            self.epochs: int, the max number of times over which all data is passed
            self.lr: float, the term by which the gradient is multiplied
            self.C: float value, value of regularization constant
        """
        super().__init__(use_bias_term)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Fit the weights by using mini-batch gradient descent with ridge regularization.
        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        Basically, assume the data has already been shuffled and just iterate.
        Initialize weights to 0, then use self.batch_size, self.epochs, and self.lr for the iterative steps.

        After each epoch, you should calculate the training RMSE and append that to a list to be returned.
        Recall you have the utilities from BaseRegressor inherited.

        Don't apply regularization to the bias term, if one exists.

        Note:
            If self.use_bias_term is true, remember to call _add_bias_term to add a ones column to the data.
        Args:
            X: np.ndarray(N, D), the data for which we want to find optimal weights
                N is the number of instances
                D is the dimensionality of each instance
            y: np.ndarray(N, 1), the ground truth that we want to model
        Update:
            self.weights -> np.ndarray(D,1)
        Return:
            training_loss_per_epoch: list[float], the RMSE of the training data (X) per epoch
        """
        if self.use_bias_term:
            X = self._add_bias_term(X)

        self.weights = np.zeros((X.shape[1], 1))
        training_loss_per_epoch = []


        for epoch in range(self.epochs):
            

            for index in range(0, X.shape[0], self.batch_size):
                x_i = X[index:index+self.batch_size]
                y_i = y[index:index+self.batch_size]

                reg_term = self.C * self.weights/self.batch_size
                reg_term[0] = 0.0

                gradient = reg_term + (x_i.T @ (x_i @ self.weights - y_i))/self.batch_size
                self.weights = self.weights - (self.lr * gradient)

            
            err = self.rmse(X @ self.weights, y)
            training_loss_per_epoch.append(err)

        return training_loss_per_epoch
