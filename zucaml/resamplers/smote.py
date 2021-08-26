# import pandas as pd
# import numpy as np

# import multiprocessing

# from sklearn.base import BaseEstimator
# from sklearn.base import TransformerMixin

# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline

# n_cores = multiprocessing.cpu_count()
# rnd_slt = 123

# class KMeansDistances(BaseEstimator, TransformerMixin):

#     def __init__(self, params = {}):
        
#         super().__init__()
        
#         self.name = self.__class__.__name__
        
#         self.params = params
    
#     def fit(self, X, y = None):

#         params = {}
#         params['random_state'] = rnd_slt
#         params['init'] = 'k-means++'
#         params['n_clusters'] = 5 if not 'n_clusters' in self.params else self.params['n_clusters']

#         transformer = KMeans(**params)

#         self.transformer = Pipeline(
#             [
#                 (
#                     'scale',
#                     StandardScaler(),
#                 ),
#                 (
#                     self.name,
#                     transformer,
#                 ),
#             ]
#         )
#         self.transformer.fit(X)
        
#         specified_clusters = self.transformer.predict(X)

#         self.specifier = OneHotEncoder(dtype = np.uint8, sparse = False, handle_unknown = 'ignore')
#         self.specifier.fit(specified_clusters.reshape(-1, 1))
        
#         self.number_clusters = pd.Series(specified_clusters).nunique()

#         return self

#     def transform(self, X):
        
#         clusters_distances = ['distance_' + str(cluster) for cluster in range(0, self.number_clusters)]
#         clusters_names = ['cluster_' + str(cluster) for cluster in range(0, self.number_clusters)]

#         distances = pd.DataFrame(self.transformer.transform(X), columns = clusters_distances)
#         clusters = pd.DataFrame(self.specifier.transform(self.transformer.predict(X).reshape(-1, 1)), index = distances.index, columns = clusters_names)
#         new_features = pd.concat([distances, clusters], axis = 1)

#         return new_features