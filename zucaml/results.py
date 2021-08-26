import pandas as pd
import numpy as np

results_format = {
    'Overfit': '{:.0%}',
    'Threshold': '{:.0%}',
    'F1': '{:.0%}',
    'F0.5': '{:.0%}',
    'F2': '{:.0%}',
    'F0.25': '{:.0%}',
    'Precision': '{:.0%}',
    'Recall': '{:.0%}',
}

params_rename = {
    'scale_pos_weight': 'class_weight',
}

def get_results(train, test, main_score):

    results = {}

    if not test is None:
        results[main_score] = test[main_score]
        results['Overfit'] = test['overfit']

    if not train['threshold'] is None:
        results['Threshold'] = train['threshold']

    if not test is None:
        for score in test:
            if score not in (main_score, 'threshold'):
                results[score.title()] = test[score]

    return results

def results_add(results, label, tag, scr):
    
    if isinstance(scr, dict):
        value_to_insert = str(scr)
    elif isinstance(scr, list):
        if len(scr) == 0:
            value_to_insert = 'None'
        elif len(scr) == 1:
            if isinstance(scr[0], dict):
                value_to_insert = str(scr[0])
            else:
                value_to_insert = scr[0]
        else:
            first = scr[0]
            value_to_insert = str(first) + ' x ' + str(len(scr))
            for v in range(1, len(scr)):
                if scr[v] != first:
                    value_to_insert = ', '.join([str(i) for i in scr])
                    break
    else:
        value_to_insert = scr

    results.loc[label, tag] = value_to_insert

def results_add_all(results, label, model_results, params):
    
    for result in model_results:
        results_add(results, label, result, model_results[result])

    for param in params:
        param_renamed = param if param not in params_rename else params_rename[param]
        results_add(results, label, param_renamed, params[param])
