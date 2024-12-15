from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.means_) / self.stds_


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.unique_values_ = {col: X[col].unique() for col in X.columns}
        return self

    def transform(self, X):
        encoded_data = []
        for col in X.columns:
            for val in self.unique_values_[col]:
                encoded_column = (X[col] == val).astype(int)
                encoded_column = encoded_column.rename(f"{col}_{val}")
                encoded_data.append(encoded_column)
        return pd.concat(encoded_data, axis=1)
