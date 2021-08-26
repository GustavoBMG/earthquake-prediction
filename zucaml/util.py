import multiprocessing
import hashlib
import copy

import pandas as pd
import numpy as np

n_cores = multiprocessing.cpu_count()

def get_memory(df):
    return df.memory_usage().sum() / (1024 * 1024)

def print_memory(df):
    print('Memory usage: {0:.2f}'.format(get_memory(df)) + ' MB')
    print('{0:,d} x {1:,d}'.format(df.shape[0], df.shape[1]))

#TODO: rename to get string form model.
#If used for tracking, create one to get the json, but string all values (funcions cannot be serialize in json dump)
def get_string(config):
    
    hashed_string = ''
    
    for section in config:
        hashed_string += '|' + section + '|'
        if isinstance(config[section], dict):
            for param in config[section]:
                hashed_string += '|' + param + '_' + str(config[section][param]) + '|'
        else:
            hashed_string += '|' + str(config[section]) + '|'
            
    return hashed_string
    
def hash_string(label, num_char = None):

    label_encoded = label.encode('utf-8')
    
    hash_object = hashlib.md5(label_encoded)
    
    hash_string = hash_object.hexdigest()
    
    if not num_char is None:
        hash_string = hash_string[:num_char]
    
    return hash_string

def get_random_string(num_char = 5):
    
    return hash_string(str(np.random.randint(0, 10000)), num_char)

def distribute_dfs(df, partition):

    items_count = df[partition].value_counts()

    buckets = {}

    for bucket in range(0, n_cores):
        buckets[bucket] = []

    count_order = 0
    for k, v in sorted(items_count.items(), key = lambda item: item[1], reverse = True):

        buckets[count_order % n_cores].append(k)

        count_order += 1
        
    return buckets
    
def distribute(df, item, func, params):

    buckets = distribute_dfs(df = df, partition = item)

    pool_params = []

    for bucket in buckets:

        df_sliced = df.loc[df[item].isin(buckets[bucket])].copy()

        params_sliced = copy.deepcopy(params)

        params_sliced['df'] = df_sliced

        params_sliced = tuple(params_sliced.values())

        pool_params.append(params_sliced)
    
    with multiprocessing.Pool(n_cores) as pool:
        dfs = pool.starmap(func, pool_params)

    return pd.concat(dfs)