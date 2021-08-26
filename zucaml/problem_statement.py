#TODO remove *, at least use the name of the module
from zucaml.feature_engineering import *

def set_target(df, item, time_ref, target, early_warning, range_warning, drop_na_target):

    new_feat = 'target'
    drop_feats = []

    df = create_reset(df = df, item = item, time_ref = time_ref, order = None)
    drop_feats.append('reset')
    
    if range_warning > 1:
        df = ts_feature(
            df = df,
            feature_base = target,
            func = 'rolling.max',
            func_val = range_warning,
            label = new_feat,
        )
    else:
        range_warning = 1
        df[new_feat] = df[target]

    df = ts_feature(
        df = df,
        feature_base = new_feat,
        func = 'shift',
        func_val = -(range_warning - 1 + early_warning),
        label = new_feat,
    )

    na_feats = df.isna().sum()

    na_feats = list(na_feats[na_feats > 0].index.values)

    for feat in na_feats:
        if feat != new_feat:
            print(f'\nWarning: Feature {feat} with na values\n')

    if df[new_feat].isna().sum() > df[item].nunique() * abs(range_warning - 1 + early_warning):
        print(f'\nWarning: Target with more na values\n')
        
    if drop_na_target:
        df = df.loc[~df[new_feat].isna()].copy()

    df = df.sort_values([item, time_ref]).reset_index().drop('index', axis = 1)
    
    df = df.drop(drop_feats, axis = 1)

    return df