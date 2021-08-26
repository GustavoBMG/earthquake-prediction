import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.preprocessing import FunctionTransformer

def log_func(x):
    return np.log1p(x)

def log_func_inv(x):
    return np.expm1(x)

log_transformer = FunctionTransformer(func = log_func, inverse_func = log_func_inv, check_inverse = True)

class LogNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        
        super().__init__()
    
    def fit(self, X, y = None):

        self.transformer = log_transformer
        self.transformer.fit(X)

        return self

    def transform(self, X):
        
        new_features = pd.DataFrame(self.transformer.transform(X))

        return new_features