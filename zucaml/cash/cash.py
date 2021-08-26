from zucaml.cash.xgboost import xgb
from zucaml.cash.random_forest import rft
from zucaml.cash.isolation_forest import iso
from zucaml.cash.neural_networks import nn
from zucaml.cash.stacking import stack
from zucaml.cash.grid import grid_search
from zucaml.cash.linear import lin

import zucaml.cash.training as mltraining

def train_score_model(params, train, test, metrics):
    
    if isinstance(params['family'], stack):
        return params['family'].train_score_model(params, train, test, metrics)
    else:
        return mltraining.train_score_model(params, train, test, metrics)