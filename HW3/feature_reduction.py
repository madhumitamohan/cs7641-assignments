import math
import sys
from typing import List
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.1
    ) -> dict:
        """
        Args:
            data: (pandas data frame) contains the features
            target: (pandas series) represents target values to search to generate significant features
            significance_level: (list) threshold to reject the null hypothesis
        Return:
            best_features: (list) contains significant features. Each feature name is a string.
        Hint:
            Forward Selection Steps:
            1. Start with an empty list of selected features (null model)
            2. For each feature NOT yet included in the selected features:
                - Fit a simple regression model using the the selected features AND the feature under consideration
                - Use sm.OLS() with sm.add_constant() to add a bias constant to your features when fitting the model
                - Record the p-value of the feature under consideration
                - Use OLSResults.pvalues to access the pvalues of each feature in a fitted model
            3. Find the feature with the minimum p-value.
                - If the feature's p-value < significance level, add the feature to the selected features and repeat from Step 2.
                - Otherwise, stop and return the selected features

            - You can access the feature names using data.columns.tolist().
            - You can index into a Pandas dataframe using multiple column names at once in a list.
        """
        raise NotImplementedError

    def evaluate_features(data: pd.DataFrame, y: pd.Series, features: list) -> None:
        """
        PROVIDED TO STUDENTS

        Performs linear regression on the dataset only using the features discovered by feature reduction for each significance level.

        Args:
            data: (pandas data frame) contains the feature matrix
            y: (pandas series) output labels
            features: (python list) contains significant features. Each feature name is a string
        """
        data_curr_features = data[features]
        x_train, x_test, y_train, y_test = train_test_split(
            data_curr_features, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        print(f"RMSE: {rmse}")
        print()
