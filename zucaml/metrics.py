import math
import copy

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

def get_metrics(actual, model_output, metrics, threshold_to_use, model_output_scaling):

    results = {}

    binary_metrics = []
    score_metrics = []

    for metric in metrics:
        if is_binary_metric(metric):
            binary_metrics.append(metric)
        else:
            score_metrics.append(metric)

    if len(binary_metrics) > 0:
        results = find_best_threshold_and_calculate(actual, model_output, binary_metrics, threshold_to_use, model_output_scaling)

    for metric in score_metrics:
        results[metric] = get_score_metric(metric, actual, model_output)

    if 'threshold' not in results:
        results['threshold'] = None

    return results

#############################################################################################################
### Score metrics
#############################################################################################################

def get_score_metric(metric, actual, model_output):
    
    if metric.lower() == 'roc_auc':
        return roc_auc_score(actual, model_output)
    elif metric.lower() == 'rmse':
        return math.sqrt(mean_squared_error(actual, model_output))
    else:
        print('Score metric not found: ' + metric)
        return
    
#############################################################################################################
### Binary metrics
#############################################################################################################

def is_binary_metric(metric):
    
    if (metric.lower().startswith('f')) or (metric.lower() == 'precision') or (metric.lower() == 'recall'):
        return True
    return False

def get_binary_metric(tp, fp, fn, metric, other_params = None):
    
    epsilon = np.finfo(float).eps
    
    if metric.lower().startswith('f'):
        
        beta = float(metric.lower().replace('f', ''))
        precision = get_binary_metric(tp, fp, fn, 'precision')
        recall = get_binary_metric(tp, fp, fn, 'recall')

        return (1.0 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall + epsilon)

    elif metric.lower() == 'precision':
        return tp / (tp + fp + epsilon)
    elif metric.lower() == 'recall':
        return tp / (tp + fn + epsilon)

def get_binary_metrics(actual, prediction, metrics):

    results = {}

    tp = np.sum(actual * prediction)
    fp = np.sum((1.0 - actual) * prediction)
    fn = np.sum(actual * (1.0 - prediction))

    for metric in metrics:
        results[metric] = get_binary_metric(tp, fp, fn, metric)

    return results

def find_best_threshold_and_calculate(actual, model_output, metrics, threshold_to_use, model_output_scaling):

    final_results = {metric: 0 for metric in metrics}

    if threshold_to_use is None:

        scale_to_search = [float(i) / 100.0 for i in range(0, 100)]
        if not model_output_scaling is None:
            if model_output_scaling == 'quantile':
                scale_to_search = np.percentile(model_output, range(0, 100))
            elif model_output_scaling == 'bin':
                scale_to_search = np.histogram(model_output, 99)[1]
            else:
                print('\n Scale search unknown \n')
                return

        for threshold in scale_to_search:
            binary_prediction = (model_output > threshold) * 1
            results = get_binary_metrics(actual, binary_prediction, metrics)

            if results[metrics[0]] > final_results[metrics[0]]:
                final_results = copy.deepcopy(results)
                final_results['threshold'] = threshold
    else:
        binary_prediction = (model_output > threshold_to_use) * 1
        final_results = get_binary_metrics(actual, binary_prediction, metrics)
        final_results['threshold'] = threshold_to_use

    return final_results