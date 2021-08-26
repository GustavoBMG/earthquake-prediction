import copy

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import zucaml.cash.training as mltraining
import zucaml.split as mlsplit
import zucaml.error_analysis as mlerror
import zucaml.global_vars as mlvars

class ModelPredictions(BaseEstimator, TransformerMixin):

    def __init__(self, model, label):
        
        super().__init__()
        
        self.name = self.__class__.__name__
        
        self.model = model
        
        self.label = label
        
        self.name_features = [label]
    
    def fit(self, X, y = None):

        return self

    def transform(self, X):
        
        predictions = pd.DataFrame(self.model.get_predictions(X), columns = [self.label])

        return predictions

class stack():
    
    def __init__(self, problem):
        self.problem = problem
        self.name = self.__class__.__name__ + ' ' + problem.name
    
    def train_score_model(self, params, train, test_, metrics):
        
        #### check
        for config_section in params:
            if config_section not in ('family', 'features', 'target', 'time_reference', 'strategy'):
                print(f'\n Unknown stack config section: {config_section} \n')

        #### params
        features_used = params['features']
        target = params['target']
        time_ref = params['time_reference']

        #### strategy
        base_configs = params['strategy']['base_configs']
        meta_config = copy.deepcopy(params['strategy']['meta_config'])
        split = params['strategy']['split']
        use_original = params['strategy']['original_features']
        use_metas = params['strategy']['meta_features']
        use_residuals = params['strategy']['base_residuals']

        #### split train in base and meta
        base, meta = mlsplit.split_by_time_ref(train, split, target, time_ref, self.problem)
        test = test_.copy()

        #### preprocess for base algorithms
        if 'preprocess' not in meta_config:
            meta_config['preprocess'] = {}
        if use_original:
            meta_config['preprocess']['original'] = {
                'features': features_used,
                'transformer': ['filler', 'clipper'],
            }

        #### train base and get predictions
        for i, base_config in enumerate(base_configs):
            
            base_features = base_config['features']
            
            label, _, _, model, _ = mltraining.train_score_model(base_config, base, None, metrics)
            
            #### add base predictions
            if use_metas[i]:
                base_transformer = ModelPredictions(model, label)
                meta_config['preprocess'][label] = {'features': base_features, 'transformer': base_transformer}
            
            # TODO: split and use other set to avoid overfit
            #### add residuals
            if use_residuals[i]:
                # get residuals
                residuals = mlerror.get_residuals(base, base_features, target, model, None)

                # train models for residuals
                models_labels = mlerror.autores(residuals, base_features, label, None)
                
                for residual_model, residual_label in models_labels:
                    residual_transformer = ModelPredictions(residual_model, residual_label)
                    meta_config['preprocess'][residual_label] = {'features': base_features, 'transformer': residual_transformer}

        return mltraining.train_score_model(meta_config, meta, test, metrics)