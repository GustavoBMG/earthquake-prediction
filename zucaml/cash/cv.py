import zucaml.split as mlsplit
import zucaml.global_vars as mlvars

def get_folds(train, target, time_ref, problem, cv_strategy, k_fold, percentage_test):

    df_folds = []
    
    if cv_strategy == mlvars.cv_strategies.TIME:

        percentage_train_fold = 1.0 / k_fold * (1.0 - percentage_test)

        number_records_test = int(len(train) * percentage_test)

        for fold in range(k_fold):
            train_fold, not_train = mlsplit.split_by_time_ref(train, percentage_train_fold * (fold + 1), target, time_ref, problem, False)
            test_fold, _ = mlsplit.split_by_time_ref(not_train, number_records_test / len(not_train), target, time_ref, problem, False)
            df_folds.append((train_fold, test_fold))
    else:
        print(f'\n Unknown cv strategy: {str(cv_strategy)} \n')
        return
        
    return df_folds