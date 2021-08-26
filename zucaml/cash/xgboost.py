import multiprocessing

from xgboost import XGBClassifier
from xgboost import XGBRegressor

import zucaml.global_vars as mlvars
import zucaml.metrics as mlmetrics

n_cores = multiprocessing.cpu_count()
rnd_slt = 123

class xgb():
    
    def __init__(self, problem):
        self.problem = problem
        self.name = self.__class__.__name__ + ' ' + problem.name

    def get_config(self, params, train, features_used, target):

        #### sets
        X = train[features_used].copy()
        y = train[target].copy()

        #### algo params
        algo_params = {}
        algo_params['seed'] = rnd_slt
        algo_params['n_jobs'] = n_cores
        algo_params['tree_method'] = 'hist'
        algo_params['use_label_encoder'] = False

        for param in params:
            if param == 'scale_pos_weight':
                if params[param] == 'balanced':
                    params[param] = sum(train[target] == 0) / sum(train[target] == 1)

            algo_params[param] = params[param]

        #### algo
        if self.problem == mlvars.problems.BINARY:
            return xgb_cls(algo_params), X, y
        elif self.problem == mlvars.problems.REGRESSION:
            return xgb_reg(algo_params), X, y
        else:
            print(f'\n Unknown objective for xgboost: {str(self.problem)} \n')
            return

    def get_predictions(self, X):

        if self.problem == mlvars.problems.BINARY:
            return self.model.predict_proba(X)[:, 1]
        elif self.problem == mlvars.problems.REGRESSION:
            return self.model.predict(X)
        else:
            print(f'\n Unknown objective for xgboost: {str(self.problem)} \n')
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
                print(f'\n Unknown objective for xgboost: {str(self.problem)} \n')
                return

        return metrics_with_scores, model_output

class xgb_cls(XGBClassifier):

    def __init__(self, params):
        
        if not 'eval_metric' in params:
            params['eval_metric'] = 'logloss'

        self.params = params
        
        super().__init__(**self.params)
        
    def fit(self, X, y):

        super().fit(X, y)

class xgb_reg(XGBRegressor):

    def __init__(self, params):
        
        self.params = params
        
        super().__init__(**self.params)
        
    def fit(self, X, y):

        super().fit(X, y)

# def sigma(y_hat):
#     return 1.0 / (1.0 + np.exp(-y_hat))

# def custom_loss(y, y_hat):
    
#     p = sigma(y_hat)
    
#     weight = 1207.292922214436
    
#     #partial function or use weights
    
# #     return -y * np.log(p) / np.log(20) - (1 - y) * np.log(1.0 - p) / np.log(1.5)

#     return -y * np.log(p) * weight - (1 - y) * np.log(1.0 - p)

# def custom_obj(y, y_hat):

#     grad_value = grad(y, y_hat)
#     hess_value = hess(y, y_hat)
    
#     return grad_value, hess_value

# def grad(y, y_hat):
    
#     delta = 1
    
#     loss_up = custom_loss(y, y_hat + delta)
#     loss_down = custom_loss(y, y_hat - delta)
    
#     return (loss_up - loss_down) / (2 * delta)

# def hess(y, y_hat):
    
#     delta = 1
    
#     grad_up = grad(y, y_hat + delta)
#     grad_down = grad(y, y_hat - delta)
    
#     return (grad_up - grad_down) / (2 * delta)