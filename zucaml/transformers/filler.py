import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class Filler(BaseEstimator, TransformerMixin):

    def __init__(self, params = {}):
        
        super().__init__()
        
        self.name = self.__class__.__name__
        
        self.params = params
    
    def fit(self, X, y = None):

        self.fill_value = 'median' if not 'fill_value' in self.params else self.params['fill_value']
        
        if isinstance(self.fill_value, str):
            if self.fill_value == 'median':
                self.fill_value = X.median().to_dict()
            elif self.fill_value == 'mean':
                self.fill_value = X.mean().to_dict()
            elif self.fill_value == 'mode':
                self.fill_value = X.median().to_dict()
            else:
                print(f'Error - fill function not found: {self.fill_value}')
                return

        return self

    def transform(self, X):
        
        new_features = X.fillna(self.fill_value)

        return new_features