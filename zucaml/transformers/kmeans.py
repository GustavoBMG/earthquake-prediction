import pandas as pd
import numpy as np

import multiprocessing

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

n_cores = multiprocessing.cpu_count()
rnd_slt = 123

class KMeansDistances(BaseEstimator, TransformerMixin):

    def __init__(self, params = {}):
        
        super().__init__()
        
        self.name = self.__class__.__name__
        
        self.params = params
        self.params['random_state'] = rnd_slt
        self.params['init'] = 'k-means++'
        self.params['n_clusters'] = 5 if not 'n_clusters' in self.params else self.params['n_clusters']
        
        number_clusters = self.params['n_clusters']

        self.clusters_distances = ['distance_' + str(cluster) + '|kmeans' for cluster in range(0, number_clusters)]
        self.clusters_names = ['cluster_' + str(cluster) + '|kmeans' for cluster in range(0, number_clusters)]
        
        self.name_features = self.clusters_distances + self.clusters_names
    
    def fit(self, X, y = None):

        transformer = KMeans(**self.params)

        self.transformer = Pipeline(
            [
                (
                    'scale',
                    StandardScaler(),
                ),
                (
                    self.name,
                    transformer,
                ),
            ]
        )
        self.transformer.fit(X)
        
        specified_clusters = self.transformer.predict(X)

        self.specifier = OneHotEncoder(dtype = np.uint8, sparse = False, handle_unknown = 'ignore')
        self.specifier.fit(specified_clusters.reshape(-1, 1))

        return self

    def transform(self, X):

        distances = pd.DataFrame(self.transformer.transform(X), columns = self.clusters_distances)
        clusters = pd.DataFrame(self.specifier.transform(self.transformer.predict(X).reshape(-1, 1)), index = distances.index, columns = self.clusters_names)
        new_features = pd.concat([distances, clusters], axis = 1)

        return new_features