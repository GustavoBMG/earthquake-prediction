import multiprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import zucaml.global_vars as mlvars
import zucaml.metrics as mlmetrics

n_cores = multiprocessing.cpu_count()
rnd_slt = 123

class rft():
    
    def __init__(self, problem):
        self.problem = problem
        self.name = self.__class__.__name__ + ' ' + problem.name

    def get_config(self, params, train, features_used, target):

        #### sets
        X = train[features_used].copy()
        y = train[target].copy()

        #### algo params
        algo_params = {}
        algo_params['random_state'] = rnd_slt
        algo_params['n_jobs'] = n_cores

        for param in params:
            algo_params[param] = params[param]
        
        #### algo
        if self.problem == mlvars.problems.BINARY:
            return rft_cls(algo_params), X, y
        elif self.problem == mlvars.problems.REGRESSION:
            return rft_reg(algo_params), X, y
        else:
            print(f'\n Unknown objective for random forest: {str(self.problem)} \n')
            return

    def get_predictions(self, X):

        if self.problem == mlvars.problems.BINARY:
            return self.model.predict_proba(X)[:, 1]
        elif self.problem == mlvars.problems.REGRESSION:
            return self.model.predict(X)
        else:
            print(f'\n Unknown objective for radom forest: {str(self.problem)} \n')
            return

    def score_predictions(self, model, X, y, metrics, threshold_to_use, score_to_compare):
    
        self.model = model

        model_output = self.get_predictions(X)

        metrics_with_scores = mlmetrics.get_metrics(y.copy(), model_output, metrics, threshold_to_use, None)

        if not score_to_compare is None:
            if self.problem == mlvars.problems.BINARY:
                metrics_with_scores['overfit'] = score_to_compare - metrics_with_scores[metrics[0]]
            elif self.problem == mlvars.problems.REGRESSION:
                metrics_with_scores['overfit'] = metrics_with_scores[metrics[0]] - score_to_compare
            else:
                print(f'\n Unknown objective for random forest: {str(self.problem)} \n')
                return

        return metrics_with_scores, model_output

class rft_cls(RandomForestClassifier):

    def __init__(self, params):
        
        self.params = params
        
        super().__init__(**self.params)
        
    def fit(self, X, y):
        
        super().fit(X, y)

class rft_reg(RandomForestRegressor):

    def __init__(self, params):
        
        self.params = params
        
        super().__init__(**self.params)
        
    def fit(self, X, y):

        super().fit(X, y)