import multiprocessing

from sklearn.ensemble import IsolationForest

import zucaml.global_vars as mlvars
import zucaml.metrics as mlmetrics

n_cores = multiprocessing.cpu_count()
rnd_slt = 123

class iso():
    
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
            return iso_cls(algo_params), X, y
        else:
            print('\n Unknown objective for isolation forest \n')
            return
        
    def score_pred(self, model, X, y, metrics, threshold_to_use, score_to_compare):
    
        if self.problem == mlvars.problems.BINARY:
            model_output = -model.decision_function(X)
        else:
            print('\n Unknown objective for isolation forest \n')
            return

        metrics_with_scores = mlmetrics.get_metrics(y.copy(), model_output, metrics, threshold_to_use, 'bin')

        if not score_to_compare is None:
            if self.problem == mlvars.problems.BINARY:
                metrics_with_scores['overfit'] = score_to_compare - metrics_with_scores[metrics[0]]
            else:
                print('\n Unknown objective for isolation forest \n')
                return

        return metrics_with_scores, model_output

class iso_cls(IsolationForest):

    def __init__(self, params):
        
        self.params = params
        
        super().__init__(**self.params)
        
    def fit(self, X, y):
        
        super().fit(X, y)