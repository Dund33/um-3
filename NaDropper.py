import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class HighNaNDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.columns_to_drop_ = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        nan_ratios = X.isna().mean()
        self.columns_to_drop_ = nan_ratios[nan_ratios > self.threshold].index.tolist()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        if not self.columns_to_drop_:
            return X

        X_transformed = X.drop(columns=self.columns_to_drop_)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided if X was a numpy array")

        if isinstance(input_features, (np.ndarray, list)):
            input_features = np.array(input_features)
            mask = ~np.isin(input_features, self.columns_to_drop_)
            return input_features[mask]
        else:
            # Handle case where input_features might be a pandas Index
            return input_features.difference(self.columns_to_drop_)