import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.preprocessing import KBinsDiscretizer

class KBins(BaseEstimator, TransformerMixin):

    def __init__(self, params = {}):
        
        super().__init__()
        
        self.name = self.__class__.__name__
        
        self.params = params
        self.params['encode'] = 'onehot-dense'
        self.params['strategy'] = 'uniform'
        self.params['n_bins'] = 10 if not 'n_bins' in self.params else self.params['n_bins']

    def fit(self, X, y = None):

        self.transformer = KBinsDiscretizer(**self.params)

        self.transformer.fit(X)
        
        self.name_features = [str(feature) + '_' + str(kbin) + '|kbin_encod' for kbin in range(0, self.params['n_bins']) for feature in X]

        return self

    def transform(self, X):

        new_features = pd.DataFrame(self.transformer.transform(X), columns = self.name_features)

        return new_features