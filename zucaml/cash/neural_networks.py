import random as python_random

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv3D, Conv2D
from tensorflow.keras.layers import MaxPooling3D, MaxPooling2D
from tensorflow.keras.layers import Flatten
import tensorflow as tf
import tensorflow.keras.backend as kb

import zucaml.global_vars as mlvars
import zucaml.metrics as mlmetrics

rnd_slt = 123

np.random.seed(rnd_slt)
python_random.seed(rnd_slt)
tf.random.set_seed(rnd_slt)

#############################################################################
# issue 11735
# https://github.com/keras-team/keras/issues/11735
# after solved:
#  - use class_weight on binary_conv
#  - last layer sigmoid
#  - reshape function dont need class weight
#  - loss used should be binary_crros_entropy or other binary
#############################################################################

#############################################################################
# issue 9643
# https://github.com/ray-project/ray/issues/9643
# after solved:
#  - remove kerasbackend setting
#############################################################################

class nn():
    
    def __init__(self, problem):
        self.problem = problem
        self.name = self.__class__.__name__ + ' ' + problem.name

    def get_config(self, params, train, features_used, target):
        
        #### algo params
        algo_params = {}
        algo_params['callbacks'] = EarlyStopping(
            monitor = 'val_loss',
            mode = 'auto',
            min_delta = 0,
            patience = params['stop_rounds'],
            restore_best_weights = True,
        )
        algo_params['batch_size'] = params['batch']
        algo_params['epochs'] = params['epochs']
        algo_params['validation_split'] = 0.1 if not 'validation_split' in params else params['validation_split']
        if self.problem == mlvars.problems.BINARY: # issue 11735
            algo_params['class_weight'] = {0: 1.0, 1: 1.0} if not 'class_weight' in params else {0: 1.0, 1: params['class_weight']}

        #### sets and algo
        if self.problem == mlvars.problems.BINARY_AUTENC:
            algo = nn_seq_aut(algo_params)
            X = train.loc[train[target] == 0, features_used].copy()
            y = None
        elif self.problem == mlvars.problems.BINARY_CONV:
            
            record_shape = get_shape(train, features_used, params['coordinates'])
            coordinates_nunique_prod = np.prod([record_shape[i] for i in range(len(record_shape) - 1)])
            class_weight = 1.0 if not 'class_weight' in params else params['class_weight'] # issue 11735
            
            algo = nn_seq_conv(algo_params, record_shape, coordinates_nunique_prod, class_weight)
            X = train[features_used].copy()
            y = train[target].copy()
        else:
            algo = nn_seq(algo_params, self.problem)
            X = train[features_used].copy()
            y = train[target].copy()

        #### layers
        if self.problem == mlvars.problems.BINARY_CONV:
            
            convs = params['convs']
            activation = params['activation']
            
            for conv in convs:
                if len(record_shape) == 4:
                    algo.add(Conv3D(conv['number_filters'], kernel_size = conv['kernel_size'], activation = activation, kernel_initializer = 'he_uniform', input_shape = record_shape))
                elif len(record_shape) == 3:
                    algo.add(Conv2D(conv['number_filters'], kernel_size = conv['kernel_size'], activation = activation, kernel_initializer = 'he_uniform', input_shape = record_shape))
            
            if 'maxpooling' in params:
                if len(record_shape) == 4:
                    algo.add(MaxPooling3D(pool_size = params['maxpooling']))
                elif len(record_shape) == 3:
                    algo.add(MaxPooling2D(pool_size = params['maxpooling']))

            if 'dropout' in params:
                algo.add(Dropout(params['dropout']))

            algo.add(Flatten())
            algo.add(Dense(coordinates_nunique_prod, activation = 'linear', bias_initializer = tf.keras.initializers.Constant(np.log(class_weight)))) # issue 11735
            
        else:
            
            hidden = params['hidden']
            activation = params['activation']

            algo.add(Dense(hidden[0], input_dim = params['first_layer'], activation = activation))
            for layer in range(1, len(hidden)):
                algo.add(Dense(hidden[layer], activation = activation))

            #### last layer
            if self.problem == mlvars.problems.BINARY:
                algo.add(Dense(1, activation = 'sigmoid'))
            elif self.problem == mlvars.problems.REGRESSION:
                algo.add(Dense(1, activation = 'linear'))
            elif self.problem == mlvars.problems.BINARY_AUTENC:
                algo.add(Dense(params['last_layer'], activation = 'linear'))
            else:
                print('\n Unknown objective for neural networks \n')
                return

        #### compile nn
        monitor_metrics = []
        if 'metrics' in params:
            monitor_metrics = [metric if not isinstance(metric, str) else get_custom_metric(metric) for metric in params['metrics']]

        algo.compile(
            loss = params['loss'],
            optimizer = params['optimizer'],
            metrics = monitor_metrics,
        )

        return algo, X, y
    
    def get_predictions(self, X):
        
        return self.model.predict(X)

    def score_predictions(self, model, X, y, metrics, threshold_to_use, score_to_compare):
        
        self.model = model

        model_output = self.get_predictions(X)

        #### scaling
        if self.problem == mlvars.problems.BINARY_AUTENC or self.problem == mlvars.problems.BINARY_CONV:
            model_output_scaling = 'bin'
        else:
            model_output_scaling = None

        metrics_with_scores = mlmetrics.get_metrics(y.copy(), model_output, metrics, threshold_to_use, model_output_scaling)

        if not score_to_compare is None:
            if self.problem == mlvars.problems.BINARY or self.problem == mlvars.problems.BINARY_AUTENC or self.problem == mlvars.problems.BINARY_CONV:
                metrics_with_scores['overfit'] = score_to_compare - metrics_with_scores[metrics[0]]
            elif self.problem == mlvars.problems.REGRESSION:
                metrics_with_scores['overfit'] = metrics_with_scores[metrics[0]] - score_to_compare
            else:
                print(f'\n Unknown objective for neural networks: {str(self.problem)} \n')
                return

        return metrics_with_scores, model_output

class nn_seq(Sequential):

    def __init__(self, params, problem):
        
        tf.keras.backend.set_floatx('float64') # issue 9643

        self.params = params
        
        self.problem = problem

        super().__init__()

    def fit(self, X, y):
        
        tf.keras.backend.set_floatx('float64') # issue 9643

        super().fit(
            x = X,
            y = y,
            **self.params,
        )
        
    def predict(self, X):
        
        tf.keras.backend.set_floatx('float64') # issue 9643
        
        if self.problem == mlvars.problems.BINARY:
            return super().predict(X)[:, 0]
        elif self.problem == mlvars.problems.REGRESSION:
            return super().predict(X)
        else:
            print(f'\n Unknown objective for neural networks: {str(self.problem)} \n')
            return
        
class nn_seq_aut(Sequential):

    def __init__(self, params):

        self.params = params

        super().__init__()

    def fit(self, X, y):
        
        super().fit(
            x = X,
            y = X,
            **self.params,
        )
        
    def predict(self, X):
        
        X_encoded = super().predict(X)
            
        return tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)(X, X_encoded).numpy()
        
class nn_seq_conv(Sequential):

    def __init__(self, params, record_shape, coordinates_nunique_prod, class_weight):
        
        tf.keras.backend.set_floatx('float32') # issue 9643

        self.params = params
        
        self.record_shape = record_shape
        self.coordinates_nunique_prod = coordinates_nunique_prod
        self.class_weight = class_weight

        super().__init__()

    def fit(self, X, y):
        
        tf.keras.backend.set_floatx('float32') # issue 9643
        
        X_reshaped, y_reshape = reshape_data(X, y, self.record_shape, self.coordinates_nunique_prod, self.class_weight)

        super().fit(
            x = X_reshaped,
            y = y_reshape,
            **self.params,
        )
        
    def predict(self, X):
        
        tf.keras.backend.set_floatx('float32') # issue 9643
        
        X_reshaped, _ = reshape_data(X, None, self.record_shape, self.coordinates_nunique_prod, None)

        return np.asarray(super().predict(X_reshaped)).reshape(len(X))
    
#############################################################################
# Metrics
#############################################################################
def get_custom_metric(metric):

    if metric == 'soft_f1':
        return soft_f1_probabilities
    else:
        print(f'\n Unknown metric for neural networks: {str(metric)} \n')
        return

# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
def soft_f1_probabilities(actual, model_output):

    y = tf.cast(actual, tf.float64)

    tp = kb.sum(y * model_output)
    fp = kb.sum((1.0 - y) * model_output)
    fn = kb.sum(y * (1.0 - model_output))

    precision = tp / (tp + fp + kb.epsilon())
    recall = tp / (tp + fn + kb.epsilon())

    return 2.0 * (precision * recall) / (precision + recall + kb.epsilon())

def soft_f1_logits(actual, model_output):

    p = tf.math.sigmoid(model_output)
    y = tf.cast(actual, tf.float64)

    tp = kb.sum(y * p)
    fp = kb.sum((1.0 - y) * p)
    fn = kb.sum(y * (1.0 - p))

    precision = tp / (tp + fp + kb.epsilon())
    recall = tp / (tp + fn + kb.epsilon())

    return 2.0 * (precision * recall) / (precision + recall + kb.epsilon())

#############################################################################
# Convolutional
#############################################################################
def get_shape(df, features, coordinates):
    
    dimensions_lenght = {coordinate:df[coordinate].nunique() for coordinate in coordinates}
    
    if len(coordinates) == 3:
        return (dimensions_lenght[coordinates[0]], dimensions_lenght[coordinates[1]], dimensions_lenght[coordinates[2]], len(features))
    elif len(coordinates) == 2:
        return (dimensions_lenght[coordinates[0]], dimensions_lenght[coordinates[1]], len(features))
    else:
        print(f'\n Unknown len of coordinates for conv neural networks: {str(len(coordinates))} \n')

def reshape_data(X, y, record_shape, coordinates_nunique_prod, class_weight): # issue 11735
    
    X_reshaped = []
    y_reshaped = []
    
    #### check if is right shape
    if len(X) % coordinates_nunique_prod != 0:
        error_message = '\nDataFrame / Array in wrong shape. Not multiple of unique coordinates.\n'
        error_message += f'Array: {str(len(X))} \t Coord: {str(coordinates_nunique_prod)}\n'
        error_message += 'Also confirms that is ordered by time then coordinates\n'
        print(error_message)
        return

    #### loop through time to append
    for i in range(int(len(X) / coordinates_nunique_prod)):
        
        lower_limit = i * coordinates_nunique_prod
        upper_limit = (i + 1) * coordinates_nunique_prod
        
        slice_features = X[lower_limit:upper_limit].copy()
        array_features = np.asarray(slice_features).reshape(record_shape)
        X_reshaped.append(array_features)

        if not y is None:
            slice_target = y[lower_limit:upper_limit].copy() * class_weight # issue 11735
            array_target = np.asarray(slice_target)
            y_reshaped.append(array_target)

    X_reshaped = np.asarray(X_reshaped)
    y_reshaped = np.asarray(y_reshaped)
    
    assert(len(X_reshaped) == int(len(X) / coordinates_nunique_prod))
    if not y is None:
        assert(len(y_reshaped) == int(len(X) / coordinates_nunique_prod))
    
    return X_reshaped, y_reshaped