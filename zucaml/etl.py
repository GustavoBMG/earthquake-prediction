from datetime import timedelta

import pandas as pd

#TODO remove *, at least use the name of the module
from zucaml.util import *
from zucaml.feature_engineering import *
from zucaml.file_handler import *

def reindex_by_minmax(df, item, time_ref, time_freq, forwardfill_features, backfill_features, zerofill_features):
    return distribute(df = df, item = item, func = single_core_reindex_by_minmax, params = locals()).reset_index().drop('index', axis = 1)

def single_core_reindex_by_minmax(df, item, time_ref, time_freq, forwardfill_features, backfill_features, zerofill_features):

    drop_feats = []

    #### create reset
    df = create_reset(
        df = df,
        item = item,
        time_ref = time_ref,
        order = None,
    )

    #### reindex
    df = df.groupby(item).resample(time_freq, on = time_ref, closed = 'left').sum()
    df = df.reset_index()

    #### ordering to fill by pair
    df = create_reset(
        df = df,
        item = item,
        time_ref = time_ref,
        order = None,
    )
    drop_feats.append('reset')

    #### ffill
    for feat in forwardfill_features:
        df[feat] = df.groupby(['reset'])[feat].fillna(method = 'ffill')

    #### bfill
    for feat in backfill_features:
        df[feat] = df.groupby(['reset'])[feat].fillna(method = 'bfill')

    #### fill 0 doesnt matter order
    for feat in zerofill_features:
        df[feat] = df[feat].fillna(0)

    #### aux
    df = df.drop(drop_feats, axis = 1)

    return df

def get_relationship(df, group_key, label, path, group_id, group_feat):

    df_groups = pd.read_csv(path)

    dict_groups = df_groups.set_index(group_id)[group_feat].to_dict()

    df[label] = df[group_key].map(dict_groups)
    
    return df

#TODO: create one for categorical feats.... mode etc, right now is only sum
#TODO: distribute
def get_group_df(df, group, time_ref, feature, time_freq):
    
    #### sum all in same time_ref
    #TODO: create one weighted by .... like price, weighted by qty.. before mult item by weight
    df_grouped = df.groupby([group, time_ref]).agg({feature: np.sum}).reset_index()

    #### resample to reindex in time (feat already summed). Using sum, that way no need to fill
    df_grouped = df_grouped.groupby(group).resample(time_freq, on = time_ref, closed = 'left').sum()

    #### check if group exists (if was int it will be calculated as well) and drop to reindex
    if group in df_grouped:
        df_grouped = df_grouped.drop(group, axis = 1)

    #### reset index
    df_grouped = df_grouped.reset_index()

    #### order ... no need in theory
    df_grouped = df_grouped.sort_values([group, time_ref], ascending = [True, True]).reset_index().drop('index', axis = 1)

    return df_grouped

def merge_group_df(df, df_group, time_ref, group):

    df = pd.merge(
        df,
        df_group,
        how = 'left',
        on = [group, time_ref],
        suffixes = ['_repeated_feat_original', '_repeated_feat_merged_' + group]
    )

    for feat in df:
        if '_repeated_feat_' in feat:
            print('Repeated feature: ' + feat)
    
    return df

def pad(df, item, time_ref, all_items, boundry, zero_fill, other_fill):

    if boundry == 'min':
        boundry_time = df[time_ref].min()
    elif boundry == 'max':
        boundry_time = df[time_ref].max()
    else:
        print(f'\n Unknown boundry: {str(boundry)} \n')
    
    boundry_items = df[df[time_ref] == boundry_time][item].unique()
    items_without_boundry = [i for i in all_items if i not in boundry_items]
    
    features_without_fill = []
    dict_to_add = {}
    for feature in df:
        if feature == item:
            dict_to_add[item] = items_without_boundry
        elif feature == time_ref:
            dict_to_add[time_ref] = boundry_time
        else:
            if feature in zero_fill:
                dict_to_add[feature] = 0
            elif feature in other_fill:
                dict_to_add[feature] = other_fill[feature]
            else:
                dict_to_add[feature] = np.nan
                features_without_fill.append(feature)

    df_to_add = pd.DataFrame(dict_to_add)
    
    assert(df_to_add[[feature for feature in df_to_add if feature not in features_without_fill]].isna().sum().sum() == 0)
    
    df = pd.concat([df_to_add, df])
    
    df = df.sort_values([item, time_ref], ascending = [True, True]).reset_index().drop('index', axis = 1)
    
    return df