import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from zucaml.transformers.log import LogNormalizer
from zucaml.transformers.iforest import iForestScore
from zucaml.transformers.kmeans import KMeansDistances
from zucaml.transformers.onehot import OneHot
from zucaml.transformers.kbins import KBins
from zucaml.transformers.clipper import Clipper
from zucaml.transformers.filler import Filler

def _get_transformer(transformer):
    
    if isinstance(transformer, str):
        if transformer.lower() == 'passthrough':
            return 'passthrough'
        elif transformer.lower() == 'standard_scaler':
            return StandardScaler()
        elif transformer.lower() == 'minmax_scaler':
            return MinMaxScaler()
        elif transformer.lower() == 'log_normalizer':
            return LogNormalizer()
        elif transformer.lower() == 'iforest_score':
            return iForestScore()
        elif transformer.lower() == 'kmeans_distances':
            return KMeansDistances()
        elif transformer.lower() == 'onehot_encoder':
            return OneHot()
        elif transformer.lower() == 'bin':
            return KBins()
        elif transformer.lower() == 'clipper':
            return Clipper()
        elif transformer.lower() == 'filler':
            return Filler()
        else:
            print(f'\n Unknown transformer: {str(transformer)} \n')
            return
    else:
        return transformer

def get_transformer(transformer):
    
    if isinstance(transformer, list):
        return Pipeline([('step_' + str(i), _get_transformer(v)) for i, v in enumerate(transformer)])
    else:
        return _get_transformer(transformer)
    
def _get_features_names(transformer, features):

    if hasattr(transformer, 'name_features'):
        return transformer.name_features
    else:
        return features
    
def get_features_names(transformer, features):
    
    if isinstance(transformer, Pipeline):
        return _get_features_names(transformer.steps[-1][1], features)
    else:
        return _get_features_names(transformer, features)