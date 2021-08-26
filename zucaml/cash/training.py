from datetime import datetime
import multiprocessing

from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

import zucaml.util as mlutil
import zucaml.results as mlresults
from zucaml.transformers import transformers

n_cores = multiprocessing.cpu_count()
rnd_slt = 123

def get_label(params, feats, n_rows):

    hashed_string = mlutil.get_string(params)

    hashed_string += ' feats: '
    for feat in feats:
        hashed_string += feat + ' :: '
        
    hashed_string += ' n_rows: ' + str(n_rows)

    return mlutil.hash_string(hashed_string)

def get_param_value(param_value):
    if isinstance(param_value, str) or isinstance(param_value, int) or isinstance(param_value, float):
        return str(param_value)
    elif isinstance(param_value, list):
        return param_value
    elif hasattr(param_value, '__name__'):
        return param_value.__name__
    elif hasattr(param_value, '__class__'):
        if hasattr(param_value.__class__, '__name__'):
            return param_value.__class__.__name__
        else:
            return param_value
    else:
        return param_value

def train_score_model(params, train, test, metrics):

    register = {}

    #### check
    for config_section in params:
        if config_section not in ('family', 'algo', 'target_transform', 'preprocess', 'resampler', 'features', 'target'):
            print(f'\n Unknown config section: {config_section} \n')
    
    #### features and target
    features_used = params['features']
    target = params['target']

    #### family
    family = params['family']
    register['family'] = family.name

    #### config algo
    algo, X, y = family.get_config(params['algo'], train, features_used, target)
    for param in params['algo']:
        register[param] = get_param_value(params['algo'][param])
    
    #### config target algo
    if 'target_transform' in params:
        target_steps = []
        for target_step in params['target_transform']:
            target_steps.append(
                (target_step, params['target_transform'][target_step]['transformer'])
            )
        if len(target_steps) > 0:
            algo = TransformedTargetRegressor(regressor = algo, transformer = Pipeline(target_steps))
            
        for step in params['target_transform']:
            register['target_' + step] = params['target_transform'][step]['transformer']

    #### config preprocess
    transformer_steps = []
    register_steps = []
    for pre_step in params['preprocess']:

        # get config values
        step_features = params['preprocess'][pre_step]['features']
        step_transformer_string = params['preprocess'][pre_step]['transformer']
        step_transformer = transformers.get_transformer(step_transformer_string)

        # append to final transformer
        transformer_steps.append(
            (
                pre_step,
                step_transformer,
                step_features
            )
        )

        # register
        if isinstance(step_transformer_string, list):
            register_steps.append(pre_step + ' ' + ' > '.join(get_param_value(transformer) for transformer in step_transformer_string))
        else:
            register_steps.append(get_param_value(pre_step))
        register['preprocess'] = ' | '.join(register_steps)
    pre = ColumnTransformer(n_jobs = n_cores, transformers = transformer_steps, remainder = 'drop')

    #### config resampler
    resampler = FunctionTransformer(func = None, inverse_func = None)
    if 'resampler' in params:
        if len(params['resampler']) > 0:
            resampler = params['resampler']['method']
            resampler.n_jobs = n_cores
            resampler.random_state = rnd_slt
            
        for step in params['resampler']:
            register['resampler_' + step] = params['resampler'][step]['method']

    #### set Pipeline
    model = Pipeline([('preprocess', pre), ('resampler', resampler), ('algo', algo)])

    #### train
    start_training_time = datetime.now()
    model.fit(X = X, y = y)
    register['time'] = int((datetime.now() - start_training_time).total_seconds() / 60.0)
    
    #### get list of features after transform
    features_final = []
    for preprocess_step in pre.transformers_:

        step_features = preprocess_step[2]
        step_transformer = preprocess_step[1]
        
        if step_transformer != 'drop':
            features_final += transformers.get_features_names(step_transformer, step_features)

    #### get label
    label = get_label(params, features_used, len(X))

    #### score
    main_metric = metrics[0]
    train_score, _ = family.score_predictions(model, train[features_used], train[target], metrics, None, None)
    test_score = None
    if not test is None:
        test_score, model_output = family.score_predictions(model, test[features_used], test[target], metrics, train_score['threshold'], train_score[main_metric])

    #### store results
    results = mlresults.get_results(train_score, test_score, main_metric)

    return label, results, register, family, features_final