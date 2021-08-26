import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class Clipper(BaseEstimator, TransformerMixin):

    def __init__(self, params = {}):
        
        super().__init__()
        
        self.name = self.__class__.__name__
        
        self.params = params
    
    def fit(self, X, y = None):
        
        self.min_max = {}
        
        for feature in X:
            
            max_value = X[feature].max()
            min_value = X[feature].min()
            
            #TODO: check which one is better
            for i in range(99, 0, -1):
                max_value = np.percentile(X[feature].dropna(), i)
                if max_value != np.inf and not np.isnan(max_value):
                    break

            for i in range(1, 100):
                min_value = np.percentile(X[feature].dropna(), i)
                if min_value != np.NINF and not np.isnan(min_value):
                    break

            self.min_max[feature] = {'min_value': min_value, 'max_value': max_value}
            
#             values_no_inf = X[feature].dropna()
#             values_median = values_no_inf.median()
#             values_no_inf[values_no_inf == np.inf] = values_median
#             values_no_inf[values_no_inf == np.NINF] = values_median

#             self.min_max[feature] = {'min_value': values_no_inf.min(), 'max_value': values_no_inf.max()}
        
        return self

    def transform(self, X):
        
        new_features = pd.DataFrame()
        
        for feature in X:
            new_features[feature] = np.clip(X[feature], self.min_max[feature]['min_value'], self.min_max[feature]['max_value'])

        return new_features