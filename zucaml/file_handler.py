import pandas as pd
import json
import datetime

import zucaml.util as mlutil

date_time_format = '%Y-%m-%d %H:%M:%S'

def get_csv(path, label, only_fields):
    
    with open(path + label + '_feats.txt') as json_file:
        types_desc = json.load(json_file)

    if len(only_fields) > 0:
        to_discard = []
        for feat in types_desc:
            if feat not in only_fields:
                print(f'Discarded:\t{feat}')
                to_discard.append(feat)
        
        for feat in to_discard:
            types_desc.pop(feat)
        
        for feat in only_fields:
            if feat not in types_desc:
                print(f'Warning:\t{feat} not found')
                
        only_fields = [feat for feat in only_fields if feat in types_desc]

    dts = [feat for feat in types_desc if types_desc[feat].startswith('datetime')]
    
    for feat in dts:
        types_desc.pop(feat)
    
    df = pd.read_csv(path + label + '.csv', dtype = types_desc, usecols = only_fields if len(only_fields) > 0 else None)
    
    if len(only_fields) > 0:
        df = df[only_fields]

    for dt in dts:
        df[dt] = pd.to_datetime(df[dt], format = date_time_format, exact = True)

    return df

def downcast_csv(df, unsigned_feats, integer_feats, float_feats):
    
    before = mlutil.get_memory(df)

    for feat in unsigned_feats:
        if feat in df:
            df[feat] = pd.to_numeric(df[feat], downcast = 'unsigned')
        else:
            print(f'Unsigned feature not found\t{feat}')

    for feat in integer_feats:
        if feat in df:
            df[feat] = pd.to_numeric(df[feat], downcast = 'integer')
        else:
            print(f'Integer feature not found\t{feat}')

    for feat in float_feats:
        if feat in df:
            df[feat] = pd.to_numeric(df[feat], downcast = 'float')
        else:
            print(f'Float feature not found  \t{feat}')

    after = mlutil.get_memory(df)
    
    print('Downcast: {0:.0%}'.format(after / before - 1))
    
    return df
    
def print_csv(df, path, label, print_index = False):

    feat_dict = {}

    for feat in df:
        feat_dict[feat] = str(df[feat].dtype)

    with open(path + label + '_feats.txt', 'w') as outfile:
        json.dump(feat_dict, outfile)

    df.to_csv(path + label + '.csv', index = print_index)

    mlutil.print_memory(df)
    
def print_results(df, path, label, model_config, feats_used, results, n_rows):
    
    file = str(label) + '__' + str(int(results['Score'] * 100)) + '.csv'
    
    df.to_csv(path + 'submit_csv/' + file, index = False)

    general_stats = {}
    general_stats['results'] = results
    general_stats['n_feats'] = len(feats_used)
    general_stats['feats'] = feats_used
    general_stats['date'] = datetime.datetime.now().strftime('%H:%M %d/%m')
    general_stats['n_rows'] = n_rows
    general_stats['model'] = mariml.util.get_string(model_config)
    
    with open(path + label + '.txt', 'w') as outfile:
        json.dump(general_stats, outfile)