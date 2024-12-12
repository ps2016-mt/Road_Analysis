from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransform(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to apply logarithmic transformation
    to numerical features.
    """
    def __init__(self, columns):
        """
        Parameters:
            columns (list): List of numerical columns to transform.
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            # Apply log transformation (add 1 to avoid log(0))
            X[col] = np.log1p(X[col])
        return X
