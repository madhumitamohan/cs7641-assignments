import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CrossValidation:
    def check_data_leakage(self, X_train: np.ndarray, X_valid: np.ndarray) -> bool:
        """
        Check for data leakage between the training and validation sets.
        Returns True if there is data leakage and False otherwise.

        Args:
            X_train: (N,...) numpy array, where N is the number of training instances
            X_valid: (M,...) numpy array, where M is the number of validation instances
        Return:
            has_leakage: bool, True if there is data leakage and False otherwise
        Hints:
            While faster numpy-centric implementations are possible, a simple Python set-based approach is sufficient for this task.
            For an extra challenge, look into np.intersect1d and np.ascontiguousarray for a more efficient implementation.
        """
        raise NotImplementedError

    def cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kfold: int = 5,
        hyperparameter: float = 1e-05,
    ) -> List[float]:
        """
        1. Partition the data into 5 sequential folds (don't do random sampling, just sequential).
        2. Iterate over the folds, starting with the first.
          - Let that fold be the validation fold
          - Let all others be the training folds
          - Collect the training folds into one training matrix
          - Train with the training data, evaluate with the validation fold
          - Append the resulting accuracy onto a list
        Don't forget to do the same thing to y as you do to X, so that the labels align with their data.

        Args:
            X: np.ndarray(N, ...), where N is the number of instances
                (should work with arbitrary shape dimensions after N)
            y: np.ndarray(N, 1), the ground truth labels corresponding to X
            kfold: int, number of folds you should take while implementing cross validation
            hyperparameter: float, value of hyperparemter for tuning
        Return:
            scores_per_fold: list[float], model performance for each fold of the cross validation
        Hints:
            np.concatenate might be helpful to collect the training folds.
            Use the model like a black box! Don't go look at LeNet unless you want to,
              but this can be completed by just calling self.fit_model and self.evaluate_model
        """
        raise NotImplementedError

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, hyperparameters: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible hyperparameter values hyperparameters
        for the one that gives the maximum average score from cross-validation

        Args:
            X: (N,...) numpy array, where N is the number of instances
                (works with arbitrary shape dimensions after N)
            y: (N,1) numpy array, true labels
            hyperparameters: list of hyperparameters to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_hyperparameter: (float) the best value for the hyperparameter giving the highest average score
            best_score: (float) the average score achieved using the best_hyperparameter
            score_list: list[float] list of average scores for each hyperparameter value given in hyperparameters
        """
        best_score = 0.0
        best_hyperparameter = hyperparameters[0]
        score_list = []
        for hp in hyperparameters:
            score = self.cross_validation(X, y, kfold, hp)
            mean_score = np.mean(score)
            score_list.append(mean_score)
            if mean_score > best_score:
                best_score = float(mean_score)
                best_hyperparameter = hp
        return best_hyperparameter, best_score, score_list

    def fit_model(
        self, X: np.ndarray, y: np.ndarray, hyperparameter: float
    ) -> nn.Module:
        """
        FUNCTION PROVIDED TO STUDENTS

        Fits a LeNet-5 style CNN on the given dataset using the provided hyperparameter (learning rate).
        You may treat this function as a black-box: Don't worry about what the model or hyperparameter
        are doing and instead focus on how this function is being used in the context of hyperparameter search.

        Args:
            X: (N,...) numpy array, where N is the number of instances
                (works with arbitrary shape dimensions after N)
            y: (N,1) numpy array, true labels
            hyperparameter: float, value of hyperparemter for tuning (learning rate in this case)
        Returns:
            model: (nn.Module) trained model
        """
        X_tensor = (
            torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            if X.ndim == 3
            else torch.tensor(X, dtype=torch.float32)
        )
        y_tensor = torch.tensor(y, dtype=torch.long).squeeze()
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LeNet5(dropout_rate=0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=hyperparameter)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(5):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return model

    def evaluate_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
        """
        FUNCTION PROVIDED TO STUDENTS

        Evaluates the given model on the provided dataset and returns the RMSE error.

        Args:
            model: (nn.Module) trained model
            X: (N,...) numpy array, where N is the number of instances
                (works with arbitrary shape dimensions after N)
            y: (N,1) numpy array, true labels
        Returns:
            score: float, Accuracy of the model on the provided dataset
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        X_tensor = (
            torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            if X.ndim == 3
            else torch.tensor(X, dtype=torch.float32)
        )
        y_tensor = torch.tensor(y, dtype=torch.long).squeeze()
        X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            total = y_tensor.size(0)
            correct = (predicted == y_tensor).sum().item()
        return correct / total


class LeNet5(nn.Module):
    def __init__(self, dropout_rate: float):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
