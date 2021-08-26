import copy

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import shap

import zucaml.cash as mlcash

def get_group_feature(feature, group):

    feature_without_group = feature.replace(group, '')

    possible_features = feature_without_group.split('_')

    to_remove = '_' + str(possible_features[len(possible_features) - 1])

    return '_'.join(possible_features).replace(to_remove, '')

def plot_features_importances(model, features, groups, normalize, as_plot):
    
    #TODO: different types of agg by groups ....mean, median
    #TODO: change to one function for getting importances and other to plot
    #TODO: return only some features: e.g. only features in x|onehot_encod
    #TODO: custom groups
    
    #### algo
    algo = model.model['algo']
    
    if isinstance(algo, mlcash.xgboost.XGBClassifier) or isinstance(algo, mlcash.xgboost.XGBRegressor):
        raw_importances = algo.get_booster().get_score(importance_type = 'gain')
        number_raw_features = algo.get_booster().num_features()
    elif isinstance(algo, mlcash.random_forest.RandomForestClassifier) or isinstance(algo, mlcash.random_forest.RandomForestRegressor):
        raw_importances = {'f' + str(i):importance for i, importance in enumerate(algo.feature_importances_)}
        number_raw_features = algo.n_features_
    else:
        print(f'\n Unknown algo: {str(algo)} \n')
        return
    
    #### check
    if number_raw_features != len(features):
        print(f'\n Mismatch number features: raw - {number_raw_features} features - {len(features)} \n')
        return
    
    #### map raw
    total_importance = 0
    importances = {feature: 0 for feature in features if feature not in raw_importances}
    
    for feature, importance in raw_importances.items():
        
        feature_index = int(feature.replace('f', ''))
        
        importances[features[feature_index]] = importance
        
        total_importance += importance
    
    #### map groups
    group_features = {}
    for group in groups:
        
        group_features[group] = {}
        
        for feature, importance in importances.items():
            if feature.endswith(group):
                group_features[group][feature] = get_group_feature(feature, group)

    for group in group_features:
        unique_features = list(set(list(group_features[group].values())))
        print(f'Group: {group} - features: {len(unique_features)} - raw features: {len(group_features[group])}')

    #### final importance
    final_importances = {}
    for feature, importance in importances.items():
        
        final_feature = feature
        
        for group in group_features:
            if feature in group_features[group]:
                final_feature = group_features[group][feature] + group
        
        if final_feature not in final_importances:
            final_importances[final_feature] = 0
        
        final_importances[final_feature] += importance if not normalize else importance / total_importance

    #### plot
    if as_plot:
        plot_x = []
        plot_y = []
        for x, y in sorted(final_importances.items(), key = lambda item: item[1]):
            plot_x.append(x)
            plot_y.append(y)

        plt.barh(plot_x, plot_y)
    else:
        for x, y in sorted(final_importances.items(), key = lambda item: item[1], reverse = True):
            if normalize:
                print(f'{y:.0%} \t {x:s}')
            else:
                print(f'{y:.0f} \t {x:s}') 

def get_shap_values(X, model, values_X):
    
    transformed_X = Pipeline(model.model.steps[:-1]).transform(X)

    explainer = shap.Explainer(model.model['algo'], transformed_X)
    
    if values_X is None:
        shap_values = explainer(transformed_X)
    else:
        transformed_values_X = Pipeline(model.model.steps[:-1]).transform(values_X)
        shap_values = explainer(transformed_values_X)

    #TODO: not working as expected, wait for following versions
#     shap_values.feature_names = features
    
    return shap_values

#TODO: abstract logic from function plot_features_importances to group features
def plot_waterfall(shap_value, number_features, features):
    
    #TODO: not working as expected, wait for following versions
    #just call the function
    
    named_shap_values = {}
    index_actual_values = {}

    for i, feature in enumerate(features):
        named_shap_values[feature] = shap_value.values[i]
        index_actual_values[feature] = i

    ordered_shap_values = dict(sorted(named_shap_values.items(), key = lambda item: abs(item[1]), reverse = False))
    filtered_shap_values = {feature: ordered_shap_values[feature] for feature in list(ordered_shap_values)[-number_features:]}

    used_values = [filtered_shap_values[feature] for feature in filtered_shap_values]

    y_ticks = [feature + ' = ' + str(round(shap_value.data[index_actual_values[feature]], 2)) for feature in filtered_shap_values]
    colors = ['blue' if value < 0 else 'red' for value in used_values]

    total_shap_value = round(shap_value.values.sum() + shap_value.base_values, 2)
    
    print(f'Total shap value: {total_shap_value} \nBase shap value: {round(shap_value.base_values, 2)}')

    plt.barh(*zip(*filtered_shap_values.items()), color = colors)
    locs, labels = plt.yticks()
    plt.yticks(range(number_features), y_ticks);
    
def plot_beeswarm(shap_values, features):

    #TODO: not working as expected, wait for following versions
    shap_values_copy = copy.deepcopy(shap_values)
    
    shap_values_copy.feature_names = features

    shap.plots.beeswarm(shap_values_copy)