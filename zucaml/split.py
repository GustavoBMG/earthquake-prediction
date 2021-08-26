import pandas as pd
import numpy as np

import zucaml.global_vars as mlvars

def split_by_time_ref(df, perc, target, time_ref, problem, print_details):

    min_all = df[time_ref].min()
    max_all = df[time_ref].max()

    ##limit
    if np.issubdtype(type(df.loc[0, time_ref]), np.integer):
        train_floor = min_all + 0

        train_ceil = int((max_all - train_floor) * perc) + train_floor

        test_ceil = max_all + 1
    else:
        train_floor = min_all + pd.to_timedelta(0, unit = 'D')

        train_ceil = (max_all - train_floor) * perc + train_floor

        test_ceil = max_all + pd.to_timedelta(1, unit = 'D')

    ##df
    df_train = df.loc[
        (df[time_ref] >= train_floor) &
        (df[time_ref] < train_ceil)
    ].copy().reset_index().drop('index', axis = 1)

    df_test = df.loc[
        (df[time_ref] >= train_ceil) &
        (df[time_ref] < test_ceil)
    ].copy().reset_index().drop('index', axis = 1)

    assert(df_train[time_ref].max() < df_test[time_ref].min())
    
    if print_details:
        print_balances([df_train, df_test], len(df), target, problem)
        
    return df_train, df_test

def print_balances(dfs, len_df, target, problem):
    
    if problem == mlvars.problems.REGRESSION:
        
        header = ''
        header += str.rjust('Total', 10) + '\t'
        header += str.rjust('Total perc', 10) + '\t'
        header += str.rjust('Median', 10) + '\t'
        header += str.rjust('Mean', 10) + '\t'
        header += str.rjust('Std', 10) + '\t'
        header += str.rjust('Min', 10) + '\t'
        header += str.rjust('Max', 10) + '\t'
        header += str.rjust('Zeros', 10) + '\t'
        print(header)

        for df in dfs:

            line = ''
            line += f'{len(df):10,}\t'
            line += f'{len(df) / len_df: 10.0%}\t'
            line += f'{df[target].median(): 10.2f}\t'
            line += f'{df[target].mean(): 10.2f}\t'
            line += f'{df[target].std(): 10.2f}\t'
            line += f'{df[target].min(): 10.2f}\t'
            line += f'{df[target].max(): 10.2f}\t'
            line += f'{sum(df[target] == 0) / len(df): 10.0%}\t'

            print(line)
        
    elif problem == mlvars.problems.BINARY:
        
        header = ''
        header += str.rjust('Total', 10) + '\t'
        header += str.rjust('Total perc', 10) + '\t'
        header += str.rjust('Balance', 10) + '\t'
        header += str.rjust('Events', 10) + '\t'
        print(header)

        for df in dfs:

            line = ''
            line += f'{len(df):10,}\t'
            line += f'{len(df) / len_df: 10.0%}\t'
            line += f'{df[target].sum() / len(df): 10.2%}\t'
            line += f'{df[target].sum(): 10,d}\t'

            print(line)
            
    else:
        print('\n Unknown objective \n')
        return