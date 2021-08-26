import pandas as pd

import multiprocessing

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

n_cores = multiprocessing.cpu_count()
rnd_slt = 123

class iForestScore(BaseEstimator, TransformerMixin):
    
    def __init__(self, params = {}):
        super().__init__()
        
        self.name = self.__class__.__name__
        
        self.params = params

        self.name_features = ['iforest_score']

    def fit(self, X, y = None):
        
        self.params['random_state'] = rnd_slt
        self.params['n_jobs'] = n_cores
        if 'n_estimators' not in self.params:
            self.params['n_estimators'] = 15
        if 'max_samples' not in self.params:
            self.params['max_samples'] = 0.2

        self.transformer = IsolationForest(**self.params)
        self.transformer.fit(X)
        
        score = self.get_score(X)
        
        self.scaler = StandardScaler()
        self.scaler.fit(score)
        
        return self
    
    def transform(self, X):

        score = self.get_score(X)
        score = self.scaler.transform(score)

        return score
    
    def get_score(self, X):
        
        score = -self.transformer.decision_function(X)
        return pd.DataFrame(score, columns = self.name_features)