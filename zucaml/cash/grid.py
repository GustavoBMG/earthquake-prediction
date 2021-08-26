import copy
import itertools

import pandas as pd
import numpy as np

import zucaml.cash.cv as mlcv
from zucaml.cash.training import train_score_model
import zucaml.results as mlresults
import zucaml.global_vars as mlvars

def get_grid(base_config, params):

    for section_param, param_values in params.items():
        section, param = section_param.split(':')
        if isinstance(section, dict):
            for section in base_config:
                if param in base_config[section]:
                    print(f'\n Param {str(section_param)} already in base config. Discarding: {str(param)} = {str(base_config[section][param])}\n')

    all_configs = []

    #https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    keys, values = zip(*params.items())

    for config in [dict(zip(keys, v)) for v in itertools.product(*values)]:
        cp = copy.deepcopy(base_config)
        for section_param in config:
            section, param = section_param.split(':')
            if not config[section_param] is None:
                cp[section][param] = config[section_param]
        all_configs.append(cp)

    return all_configs

#TODO: distribute
def grid_search(train, target, time_ref, problem, metrics, cv_strategy, k_fold, percentage_test, basic_configs_and_params):
    
    main_metric = metrics[0]
    
    df_folds = mlcv.get_folds(train, target, time_ref, problem, cv_strategy, k_fold, percentage_test)

    all_configs = []
    for basic_config, params in basic_configs_and_params:
        all_configs += get_grid(basic_config, params)

    all_scores = {}
    grid_board = pd.DataFrame({}, index = [])

    for config in all_configs:

        config_results = {}

        for fold in df_folds:

            label, results, register, _, _ = train_score_model(config, fold[0], fold[1], metrics)

            for metric in results:
                if metric.lower() != 'threshold':
                    if metric not in config_results:
                        config_results[metric] = []
                    config_results[metric].append(results[metric])

        for metric in config_results:
            config_results[metric] = np.mean(config_results[metric])

        mlresults.results_add_all(grid_board, label, config_results, register)

        all_scores[len(all_scores)] = config_results

    if problem == mlvars.problems.BINARY:
        best_score = np.NINF
    elif problem == mlvars.problems.REGRESSION:
        best_score = np.inf
    else:
        print(f'\n Unknown objective for grid search: {str(problem)} \n')
        return

    best_config_idx = None

    for config_idx, score in all_scores.items():

        if problem == mlvars.problems.BINARY:
            if score[main_metric] > best_score:
                best_score = score[main_metric]
                best_config_idx = config_idx
        elif problem == mlvars.problems.REGRESSION:
            if score[main_metric] < best_score:
                best_score = score[main_metric]
                best_config_idx = config_idx

    return grid_board, all_configs[best_config_idx]