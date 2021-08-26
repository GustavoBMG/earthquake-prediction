import pandas as pd
import numpy as np

#TODO remove *, at least use the name of the module
from zucaml.util import *
import zucaml.util as mlutil

def create_reset(df, item, time_ref, order):
    
    if order is None:
        order = [True, True]
    
    df = df.sort_values([item, time_ref], ascending = order)

    df['reset'] = (df[item].shift(1) != df[item]) * 1

    df['reset'] = df['reset'].cumsum()
    
    return df

def get_feat_label(label, feature_1, feature_2, func, func_val):
    
    str_feature_2 = str(feature_2)
    
    if label is None:
        if not feature_2 is None:
            label = feature_1 + '||' + func + '||' + str_feature_2
        else:
            label = feature_1 + '|' + func
            if not func_val is None:
                label += '#' + str(func_val)

    return label

def math_feature(df, feature_1, feature_2, func, label):
    
    label = get_feat_label(label, feature_1, feature_2, func, None)
    
    if func == 'ratio':
        df[label] = df[feature_1] / df[feature_2]
    elif func == 'diff':
        df[label] = df[feature_1] - df[feature_2]
    elif func == 'mult':
        df[label] = df[feature_1] * df[feature_2]
    elif func == 'add':
        df[label] = df[feature_1] + df[feature_2]
    elif func == 'diff.days':
        df[label] = (df[feature_1] - df[feature_2]).dt.days
    else:
        print(f'Error - math function not found: {func}')
        return

    return df

def comparison_feature(df, feature_base, comp_val, comp_operator, label):
    
    label = get_feat_label(label, feature_base, comp_val, comp_operator, None)
    
    if comp_operator == 'eq':
        df[label] = (df[feature_base] == comp_val)
    elif comp_operator == 'ne':
        df[label] = (df[feature_base] != comp_val)
    elif comp_operator == 'gt':
        df[label] = (df[feature_base] > comp_val)
    elif comp_operator == 'ge':
        df[label] = (df[feature_base] >= comp_val)
    elif comp_operator == 'lt':
        df[label] = (df[feature_base] < comp_val)
    elif comp_operator == 'le':
        df[label] = (df[feature_base] <= comp_val)
    else:
        print(f'Error - comp operator not found: {func}')
        return
    
    return df

def ts_feature(df, feature_base, func, func_val, label):
    return distribute(df = df, item = 'reset', func = single_core_ts_feature, params = locals())

def single_core_ts_feature(df, feature_base, func, func_val, label):

    label = get_feat_label(label, feature_base, None, func, func_val)

    if func.startswith('rolling'):
        df[label] = df.groupby(['reset'])[feature_base].rolling(window = func_val, min_periods = 1).agg(func.split('.')[1]).reset_index().set_index('level_1')[feature_base]
    elif func.startswith('cum'):
        df[label] = df.groupby(['reset'])[feature_base].agg(func)
    elif func == 'shift':
        df[label] = df.groupby(['reset'])[feature_base].shift(func_val)
    elif func.endswith('fill'):
        df[label] = df.groupby(['reset'])[feature_base].fillna(method = func)
    elif func.startswith('count'):
        _, count_operator, count_val = func.split('_')
        
        if '.' in count_val:
            count_val = float(count_val)
        else:
            count_val = int(count_val)
            
        df = comparison_feature(
            df = df,
            feature_base = feature_base,
            comp_val = count_val,
            comp_operator = count_operator,
            label = label,
        )

        df[label] = df.groupby(['reset'])[label].rolling(window = func_val, min_periods = 1).agg('sum').reset_index().set_index('level_1')[label]
    else:
        print(f'Error - ts function not found: {func}')
        return

    return df

def track_feature(df, feature_base, condition, track_window, track_function, label):
    
    #### features names
    if label is None:
        label = 'track.' + feature_base + '.cond.' + track_function + '#' + str(track_window)
        if label in df:
            label += mlutil.get_random_string()
            
    feature_condition = feature_base + '_cond_' + mlutil.get_random_string(10)
    feature_condition_shifted = feature_condition + '_shift_' + mlutil.get_random_string(10)
    feature_condition_shifted_filled = feature_condition_shifted + '_fill_' + mlutil.get_random_string(10)
    
    #### drop features
    drop_features = []
    drop_features.append(feature_condition)
    drop_features.append(feature_condition_shifted)
    drop_features.append(feature_condition_shifted_filled)
    
    #### calculate the condition
    df[feature_condition] = np.nan
    df.loc[condition, feature_condition] = df.loc[condition, feature_base]
    
    #### shift in time
    df = ts_feature(
        df = df,
        feature_base = feature_condition,
        func = 'shift',
        func_val = track_window,
        label = feature_condition_shifted,
    )

    #### forward fill - important to leave the nan before so here it works
    df = ts_feature(
        df = df,
        feature_base = feature_condition_shifted,
        func = 'ffill',
        func_val = None,
        label = feature_condition_shifted_filled,
    )
    
    #### TODO: group before to calculate in other granularities, like week etc
    #### calculate the function
    df = math_feature(
        df = df,
        feature_1 = feature_base,
        feature_2 = feature_condition_shifted_filled,
        func = track_function,
        label = label,
    )
    
    #### aux
    df = df.drop(drop_features, axis = 1)
    
    return df