import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.preprocessing import OneHotEncoder

class OneHot(BaseEstimator, TransformerMixin):

    def __init__(self, params = {}):
        
        super().__init__()
        
        self.name = self.__class__.__name__
        
        self.params = params
        self.params['dtype'] = np.uint8
        self.params['sparse'] = False
        self.params['handle_unknown'] = 'ignore'

    def fit(self, X, y = None):

        self.transformer = OneHotEncoder(**self.params)

        self.transformer.fit(X)

        self.name_features = [new_feature + '|onehot_encod' for new_feature in self.transformer.get_feature_names([feature for feature in X])]

        return self

    def transform(self, X):

        new_features = pd.DataFrame(self.transformer.transform(X), columns = self.name_features)

        return new_features