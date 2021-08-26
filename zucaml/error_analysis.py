import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

import zucaml.global_vars as mlvars
import zucaml.util as mlutil
from zucaml.transformers.filler import Filler
from zucaml.transformers.clipper import Clipper
from zucaml.feature_engineering import comparison_feature
from zucaml.feature_engineering import get_feat_label
from zucaml.cash.xgboost import xgb
from zucaml.cash.training import train_score_model

rnd_slt = 123

####################################
# Residuals
####################################

def get_residuals(df_, features, target, model, threshold):
    
    df = df_.copy()
    
    df['model_output'] = model.get_predictions(df[features])
    
    if model.problem == mlvars.problems.BINARY or model.problem == mlvars.problems.BINARY_AUTENC:
        if not threshold is None:
            df['binary_prediction'] = (df['model_output'] > threshold) * 1

            df['tp'] = ((df[target] == df['binary_prediction']) & (df['binary_prediction'] == 1)) * 1
            df['fp'] = ((df[target] != df['binary_prediction']) & (df['binary_prediction'] == 1)) * 1
            df['fn'] = ((df[target] != df['binary_prediction']) & (df['binary_prediction'] == 0)) * 1

        df['soft_fp'] = (1.0 - df[target]) * df['model_output']
        df['soft_fn'] = df[target] * (1.0 - df['model_output'])
        
        df['residual'] = df['soft_fp'] - df['soft_fn']
        
    elif model.problem == mlvars.problems.REGRESSION:
        df['residual'] = df['model_output'] - df[target]
    else:
        print(f'\n Unknown objective for residuals: {str(model.problem)} \n')
        return
    
    return df

####################################
# Automated residual feature engineering
####################################

def autores(residuals, features_used, label, residual_model):
    
    #### initial vars
    error_target = 'error_target'
    error_classes = []
    models_labels = []
    
    #### get default model
    if residual_model is None:
        residual_model = {
            'features': features_used,
            'target': error_target,
            'family': xgb(mlvars.problems.BINARY),
            'algo': {
                'max_depth': 3,
                'n_estimators': 15,
                'scale_pos_weight': 'balanced',
            },
            'preprocess': {
                'original': {
                    'features': features_used,
                    'transformer': 'clipper',
                },
            },
        }

    #### above
    residual_values = residuals[residuals['residual'] >= 0]['residual']

    error_upper_limit = np.histogram(residual_values, 10)[1][-3]

    error_lower_limit = np.histogram(residual_values, 10)[1][-9]

    error_selected = residuals['residual'] > error_upper_limit

    error_counter = (residuals['residual'] < error_lower_limit) & (residuals['residual'] > 0)

    error_classes.append(('above', error_selected, error_counter))

    #### below
    residual_values = residuals[residuals['residual'] <= 0]['residual']

    error_upper_limit = np.histogram(residual_values, 10)[1][3]

    error_lower_limit = np.histogram(residual_values, 10)[1][9]

    error_selected = residuals['residual'] < error_upper_limit

    error_counter = (residuals['residual'] > error_lower_limit) & (residuals['residual'] < 0)

    error_classes.append(('below', error_selected, error_counter))
    
    for error_class, error_selected, error_counter in error_classes:
        
        df = residuals.loc[(error_selected) | (error_counter)].copy()

        df[error_target] = error_selected * 1
        
        if df[error_target].nunique() == 1 or len(df) < 5_000:
            continue

        _, _, _, model, _ = train_score_model(residual_model, df, None, ['roc_auc'])
        
        models_labels.append((model, error_class + '_' + label))

    return models_labels

#TODO: update global var alll_features
def autores_tree(df, df_residual, error_classes, features, depth):
    
    new_features = []
    
    for error_class, error_selected, error_counter in error_classes:
        
        error_target = 'error_target'
        
        df_error = df_residual.loc[(error_selected) | (error_counter)].copy()

        df_error[error_target] = error_selected * 1

        error_tree = get_error_tree(df_error, features, error_target, depth)

        parents_map = get_parents_map(error_tree, features)

        leafs = get_best_leafs(error_tree, len(df_error), df_error[error_target].sum(), parents_map, 0)

        df, node_features = make_node_features(df, features, leafs, parents_map, error_class)

        new_features += node_features
        
    features += new_features
    
    return df, features

def make_node_features(df, features, leafs, parents_map, error_class):
    
    new_features = []
    
    for node in leafs:
        
        #### store bools to later get final feature
        bool_arrays = []
        
        #### features names
        label = error_class + '_' + str(node) + '|error_tree'
        if label in df:
            label += '_' + mlutil.get_random_string()
        
        #### drop features
        drop_features = []

        #### get each split in the path
        for feature, operator, threshold in node_path(node, parents_map):
            
            df = comparison_feature(
                df = df,
                feature_base = feature,
                comp_val = threshold,
                comp_operator = operator,
                label = None
            )
            
            split_node_feature = get_feat_label(
                label = None,
                feature_1 = feature,
                feature_2 = threshold,
                func = operator,
                func_val = None,
            )
            
            #### save it for later compute
            bool_arrays.append(df[split_node_feature])
            
            #### aux
            drop_features.append(split_node_feature)

        #### since is a tree, just make the prod to get final bool; T.op.T is faster than apply(op)
        bool_arrays = pd.concat(bool_arrays, axis = 1).T.prod().T
        df[label] = bool_arrays
        df[label] = df[label].astype(np.uint8)
        
        #### append new feature
        new_features.append(label)

        #### drop features
        df = df.drop(drop_features, axis = 1)
        
    return df, new_features

####################################
# Error tree
####################################

def get_error_tree(df, features, target, max_depth):

    error_tree = DecisionTreeClassifier(

        max_depth = max_depth,
        
        class_weight = None,
        random_state = rnd_slt,
    )
    
    error_tree = Pipeline([('filler', Filler()), ('clipper', Clipper()), ('algo', error_tree)])

    error_tree.fit(df[features], df[target])

    return error_tree['algo'].tree_

####################################
# Auxiliary functions for tree
####################################

def get_parents_map(tree, features):

    parents_map = {}

    for i in range(0, tree.node_count):
        if tree.children_left[i] != -1:
            parents_map[tree.children_left[i]] = {'parent_node': i, 'side': 'le', 'threshold': tree.threshold[i], 'parent_feature': features[tree.feature[i]]}
        if tree.children_right[i] != -1:
            parents_map[tree.children_right[i]] = {'parent_node': i, 'side': 'gt', 'threshold': tree.threshold[i], 'parent_feature': features[tree.feature[i]]}

    return parents_map

def is_node_above(node, node_to_check, parents_map):
    
    if not node in parents_map:
        return False
    
    parent = parents_map[node]['parent_node']
    
    while parent != 0:
        if parent == node_to_check:
            return True
        
        parent = parents_map[parent]['parent_node']
        
    return False

def node_path(node, parents_map):
    
    parent_node = parents_map[node]['parent_node']
    side = parents_map[node]['side']
    threshold = parents_map[node]['threshold']
    parent_feature = parents_map[node]['parent_feature']
    
    node_split = [(parent_feature, side, threshold)]
    
    if parent_node == 0:
        return node_split
    else:
        return node_split + node_path(parent_node, parents_map)
    
def get_best_leafs(tree, number_samples, number_errors, parents_map, min_sample_percentage):

    balance = number_errors / number_samples

    leafs = {}

    for node in range(0, tree.node_count):

        total_samples = tree.n_node_samples[node]
        error_samples = tree.value[node][0][1]

        leafs[node] = {
            'error_percentage_above_balance': error_samples / total_samples - balance,
            'sample_percentage': total_samples / number_samples,
        }
    
    discard_nodes = [node for node in leafs if leafs[node]['error_percentage_above_balance'] <= 0]
    
    discard_nodes += [node_to_check for node in leafs for node_to_check in leafs if is_node_above(node, node_to_check, parents_map)]
    
    discard_nodes += [node for node in leafs if leafs[node]['sample_percentage'] <= min_sample_percentage]
    
    leafs = [node for node in leafs if node not in discard_nodes + [0]]
    
    return leafs