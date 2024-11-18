# -*- coding: utf-8 -*-

"""
Created on Aug 17 10:00:00 2024
Modified on Oct 18 17:00 2024
Modified on Oct 23 15:00 2024
Modified on Nov 07 15:35 2024

@author: Yuriy Izotov
@author: Andrei Velichko
@user: izotov93
"""

import numbers
import time
import os
import numpy as np
import joblib
from LogNNet.pso_method import PSO
from LogNNet.mlp_evaluation import lognnet_evaluate_by_params
import warnings
from sklearn.utils import check_array, check_X_y

from . import __version__

warnings.filterwarnings("ignore", message="Stochastic Optimizer: Maximum iterations")


def validate_param(param: (tuple, int, float, str, bool, None),
                   expected_type: type, name_param: str, valid_options=(),
                   check_limits=False) -> (tuple, int, float, str, bool, None):
    """
    Checks and validates the parameter depending on its type and additional criteria.

    The parameter can be an integer, a floating point number, a string, a Boolean value, or a tuple.
    If check_limits is specified, the parameter must be a tuple or a number.

        :param param: (int, float, str, bool, tuple): Parameter for verification and validation.
        :param expected_type: (type): The expected type of the parameter.
        :param valid_options: (tuple, optional): Valid values for a variable string.
                            By default, an empty tuple for a string mismatch.
        :param check_limits: (bool, optional): Specifies whether the function should check limits
                            for a tuple or a number. If True, the parameter must be a number or a tuple
                            of two numbers. By default, False.
        :param name_param: (str): The name of the validation parameter
        :return: (int, float, str, bool, tuple, None): A verified and validated parameter.
    """

    if check_limits:
        if isinstance(param, (int, float)):
            param = (param, param)

        elif isinstance(param, tuple) and len(param) == 2 and all(isinstance(x, numbers.Number) for x in param):
            param = (param[1], param[0]) if not (param[0] < param[1]) else param

        else:
            raise ValueError(f'The parameter "{name_param}" must be of type tuple or integer or float.'
                             f'If you use tuple then the length must be 2 and each '
                             f'element must have an integer or float. Value {param} is not supported')

        if any(value < 0 for value in param):
            raise ValueError(f'The parameter "{name_param}" contains invalid negative values. '
                             f'Value {param} is not supported')

        return param

    if isinstance(param, str) and not (len(param) != 0 and (param in valid_options)):
        raise ValueError(f'The parameter "{name_param}" must be in the list {valid_options}')

    elif not isinstance(param, expected_type):
        raise ValueError(f'The parameter "{name_param}" must be of type {expected_type}')

    elif isinstance(param, (int, float)) and param <= 0 and not isinstance(param, bool):
        raise ValueError(f'The parameter "{name_param}" must be positive number. Value {param} is not supported')

    elif isinstance(param, dict):
        default_params = {
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'power_t': 0.5,
            'shuffle': True,
            'random_state': 42,
            'tol': 1e-04,
            'verbose': False,
            'warm_start': False,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'early_stopping': False,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-08,
            'n_iter_no_change': 10,
            'max_fun': 15000
        }
        validated_params = {'random_state': 42}

        for key, default_value in default_params.items():
            if key in param and param[key] != default_value:
                validated_params[key] = param[key]

        return validated_params

    else:
        return param


def validate_limit_hidden_layers(limit_hidden_layers: (tuple, int)) -> tuple:
    """
    This function checks if 'limit_hidden_layers' is a tuple that contains either:
    - integers, in which case it converts them into tuples of identical limits,
    - or tuples of integers, where each inner tuple must contain 1 or 2 integer elements.

        :param limit_hidden_layers: (tuple): The input to be validated.
        :return: tuple: A tuple containing the validated limits.
    """

    if isinstance(limit_hidden_layers, tuple):
        validated_limits = []
        for item in limit_hidden_layers:
            if isinstance(item, int):
                validated_limits.append((item, item))
            elif isinstance(item, tuple):
                if len(item) > 2 or not all(isinstance(x, int) for x in item):
                    raise ValueError("Invalid format: each element must be a tuple "
                                     "consisting of 1 or 2 integer values.")
                validated_limits.append(item)
            else:
                raise ValueError("Invalid format: elements must be integers or tuples.")
        return tuple(validated_limits)

    elif isinstance(limit_hidden_layers, int) and limit_hidden_layers > 0:
        return ((limit_hidden_layers, limit_hidden_layers), )

    raise ValueError("Invalid format: it must be a tuple of integers or a tuple of tuples.")


class BaseLogNNet(object):
    LogNNet_version = __version__

    def __init__(self,
                 num_rows_W: (tuple, int),
                 limit_hidden_layers: (tuple, int),
                 learning_rate_init: (tuple, float),
                 n_epochs: (tuple, int),
                 n_f: (tuple, int),
                 selected_metric: str,
                 selected_metric_class: (None, int),
                 num_folds: int,
                 num_particles: int,
                 num_threads: int,
                 num_iterations: int,
                 **kwargs):

        self.mlp_model = None
        self.input_layer_data = None
        self.LogNNet_best_params = {}

        # additional parameters
        use_reservoir = validate_param(kwargs.get('use_reservoir', True), bool,
                                       name_param='use_reservoir')
        self._use_debug_mode = validate_param(kwargs.get('use_debug_mode', False), bool,
                                       name_param='use_debug_mode')
        self.mlp_params = validate_param(kwargs, dict, name_param='mlp_params')

        self._param_ranges = {
            'num_rows_W': validate_param(num_rows_W, int, check_limits=True,
                                         name_param='input_layer_neurons') if use_reservoir else (0, 0),
            'Zn0': (-499, 499) if use_reservoir else (0, 0),
            'Cint': (-499, 499) if use_reservoir else (0, 0),
            'Bint': (-499, 499) if use_reservoir else (0, 0),
            'Lint': (100, 10000) if use_reservoir else (0, 0),
            'learning_rate_init': validate_param(learning_rate_init, float,
                                                 check_limits=True, name_param='learning_rate_init'),
            'epochs': validate_param(n_epochs, int, check_limits=True, name_param='n_epochs'),
            'prizn': (0, 1),
            'n_f': n_f,
            'ngen': (1, 500),
        }

        limit_hidden_layers = validate_limit_hidden_layers(limit_hidden_layers)
        for i, limits in enumerate(limit_hidden_layers, start=1):
            self._param_ranges[f'hidden_layers_{i}'] = limits

        self.basic_params = {
            'X': None,
            'y': None,
            'num_folds': validate_param(num_folds, int, name_param='num_folds'),
            'param_ranges': self._param_ranges,
            'selected_metric': selected_metric,
            'selected_metric_class': selected_metric_class,
            'num_particles': validate_param(num_particles, int, name_param='num_particles'),
            'num_threads': validate_param(num_threads, int, name_param='num_threads'),
            'num_iterations': validate_param(num_iterations, int, name_param='num_iterations'),
            'target': None,
            'static_features': None,
            'use_reservoir': use_reservoir,
            'use_debug_mode': self._use_debug_mode,
            'mlp_params': self.mlp_params,
            'test_size_in_fold': validate_param(kwargs.get('test_size_in_fold', 0.2), float,
                                        name_param='test_size_in_fold'),
        }

        if self.basic_params['test_size_in_fold'] > 0.9:
            raise ValueError("Invalid value: 'test_size_in_fold' must be between 0 and 0.9")

    def get_version(self) -> str:
        """
        Getting package version

            :return: (str): LogNNet package version
        """
        return self.LogNNet_version

    def fit(self, X, y) -> object:
        """
        Fit the model to data matrix X and targets y.

            :param X: (array-like): The input data.
            :param y: (array-like): The target values (class labels in classification, real numbers in regression).
            :return: (object): Returns a trained (MLPRegressor or MLPClassifier) model.
        """

        self.basic_params['X'], self.basic_params['y'] = check_X_y(X, y)
        self.__validate_LogNNet_params_before_fit()

        best_position, _ = PSO(**self.basic_params)

        self.LogNNet_best_params = {
            'num_rows_W': int(best_position[0]),
            'Zn0': best_position[1],
            'Cint': best_position[2],
            'Bint': best_position[3],
            'Lint': best_position[4],
            'hidden_layer_sizes': tuple(int(x) for x in best_position[10:]),
            'learning_rate_init': float(best_position[5]),
            'epochs': int(best_position[6]),
            'prizn': int(best_position[7]),
            'n_f': int(best_position[8]),
            'ngen': int(best_position[9]),
        }

        params = {
            'hidden_layer_sizes': self.LogNNet_best_params['hidden_layer_sizes'],
            'learning_rate_init': self.LogNNet_best_params['learning_rate_init'],
            'max_iter': self.LogNNet_best_params['epochs'],
        }

        if self.mlp_params is not None:
            params.update(self.mlp_params)

        res_metric, self.mlp_model, self.input_layer_data = lognnet_evaluate_by_params(
            X=X,
            y=y,
            mlp_params=params,
            lognnet_params=self.LogNNet_best_params,
            selected_metric=self.basic_params['selected_metric'],
            selected_metric_class=self.basic_params['selected_metric_class'],
            static_features=self.basic_params['static_features'],
            target=self.basic_params['target'])

        print(f"Metric '{self.basic_params['selected_metric']}' = {round(res_metric, 6)} (Train set)")

        keys_to_remove = ['X', 'y', 'use_debug_mode']
        self.basic_params = {key: value for key, value in self.basic_params.items() if key not in keys_to_remove}

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict using the LogNNet model.

            :param X: (np.ndarray): The input data.
            :return: (np.ndarray): The predicted classes.
        """
        X = check_array(X)

        if self.input_layer_data is None or self.mlp_model is None or not self.LogNNet_best_params:
            raise Exception("The LogNNet neural network model is not trained. "
                            "Use the 'fit' function before using the 'predict' function.")

        for i in range(X.shape[1]):
            if self.input_layer_data['prizn_binary'][i] == '0':
                X[:, i] = 0

        denominator = np.array(self.input_layer_data['X_train_max']) - np.array(self.input_layer_data['X_train_min'])
        denominator[denominator == 0] = 1

        if self.input_layer_data['W'] is not None:
            X_test_normalized = (X - np.array(self.input_layer_data['X_train_min'])) / denominator
            W = self.input_layer_data['W']
            X_new_test = np.dot(X_test_normalized, W.T)

            denominator_Sh = np.array(self.input_layer_data['Shmax']) - np.array(self.input_layer_data['Shmin'])
            denominator_Sh[denominator_Sh == 0] = 1
            X_new_test_Sh = (X_new_test - np.array(self.input_layer_data['Shmin'])) / denominator_Sh - 0.5
        else:
            X_new_test_Sh = (X - np.array(self.input_layer_data['X_train_min'])) / denominator

        return self.mlp_model.predict(X_new_test_Sh)

    def __validate_LogNNet_params_before_fit(self):
        self._param_ranges['prizn'] = (*self._param_ranges['prizn'][:1], 2 ** self.basic_params['X'].shape[1] - 1)

        if isinstance(self._param_ranges['n_f'], int):
            if self._param_ranges['n_f'] == -1:
                self._param_ranges['n_f'] = (self.basic_params['X'].shape[1], self.basic_params['X'].shape[1])
            elif self._param_ranges['n_f'] > 0:
                self._param_ranges['n_f'] = (int(self._param_ranges['n_f']), int(self._param_ranges['n_f']))
            else:
                raise ValueError("Invalid value for 'n_f'. Allowed values are -1 (all features) or positive numbers")

            self.basic_params['static_features'] = None
            self._param_ranges['ngen'] = (1, 1)

        elif isinstance(self._param_ranges['n_f'], tuple) and len(self._param_ranges['n_f']) == 2:
            self._param_ranges['n_f'] = (max(1, self._param_ranges['n_f'][0]),
                                         min(self._param_ranges['n_f'][1], self.basic_params['X'].shape[1]))
            self.basic_params['static_features'] = None

        elif isinstance(self._param_ranges['n_f'], list):
            self.basic_params['static_features'] = self._param_ranges['n_f']
            self._param_ranges['n_f'] = (0, 0)
            self._param_ranges['ngen'] = (0, 0)

        else:
            raise ValueError("Invalid value for 'n_f'. Support types: int, tuple or list")

        if (self.basic_params['selected_metric_class'] is not None and
                self.basic_params['target'] == 'Classifier' and
                (self.basic_params['selected_metric_class'] > int(np.max(self.basic_params['y'], axis=0)) or
                 self.basic_params['selected_metric_class'] < 0)):
            raise ValueError(f"Wrong param 'selected_metric_class'. "
                             f"Validate limits - (0, {int(np.max(self.basic_params['y'], axis=0))})")

        self.basic_params['param_ranges'] = self._param_ranges

    def export_model(self, file_name=None, **kwargs):
        """
        Save the trained LogNNet model and its parameters to a file.

        This method serializes and saves the best model and its associated
        parameters based on the specified type.

            :param file_name: (str or None): file name of the saved model.
                If none, the name is generated automatically with a prefix in the form of a timestamp.
            :return: (None) A model with the name - file_name
                or in the format will be generated '{timestamp}_LogNNet_model.joblib'
        """

        file_name = file_name if file_name is not None else f'{int(time.time())}_LogNNet_model.joblib'
        if not isinstance(file_name, str) or not file_name.endswith('.joblib'):
            raise TypeError('Parameter "file_name" must be a string and end with "*.joblib"')

        if self.input_layer_data is None or self.mlp_model is None or not self.LogNNet_best_params:
            raise Exception("The LogNNet neural network model is not trained. "
                            "Use the 'fit' function before using the 'export_model' function.")

        type_of_model = kwargs.get('type', 'max')

        if type_of_model == 'max':
            value = {'model': self.mlp_model,
                     'model_params': self.LogNNet_best_params,
                     'input_layer_data': self.input_layer_data,
                     'basic_params': self.basic_params,
                     'version': self.LogNNet_version
                     }

        elif type_of_model == 'min':
            params_reservoir = {
                'num_rows_W': self.LogNNet_best_params['num_rows_W'],
                'Zn0': self.LogNNet_best_params['Zn0'],
                'Cint': self.LogNNet_best_params['Cint'],
                'Bint': self.LogNNet_best_params['Bint'],
                'Lint': self.LogNNet_best_params['Lint'],
                'prizn_binary': self.input_layer_data['prizn_binary'],
                'Shmax': self.input_layer_data['Shmax'],
                'Shmin': self.input_layer_data['Shmin'],
                'X_train_max': self.input_layer_data['X_train_max'],
                'X_train_min': self.input_layer_data['X_train_min']
            }
            value = {'coefs': self.mlp_model.coefs_,
                     'bias': self.mlp_model.intercepts_,
                     'input_layers_params': params_reservoir}

        else:
            raise ValueError('Param "type_of_model" is not correct. Valid options: "min" or "max"')

        joblib.dump(value=value, filename=file_name)

        if self._use_debug_mode:
            print(f'Model successfully saved under file name {file_name}')

    def import_model(self, file_name: str) -> object:
        """
        Import a trained LogNNet model from a specified file.

        This method loads a serialized LogNNet model and its associated parameters
        from the provided file path. It checks for the existence of the file before
        attempting to load the model. Depending on the contents of the loaded
        data, it initializes the model and its parameters for further use.

            :param file_name: (str): The path to the file containing the serialized model.
            :return: (object) Fills an example of a class with data.
        """

        if not isinstance(file_name, str):
            raise TypeError('Parameter "file_name" must be a string.')
        if not os.path.isfile(file_name):
            raise ValueError(f'File "{file_name}" not found. Check the file path and try again.')

        model_data = joblib.load(file_name)

        if model_data['model'] is not None:
            self.mlp_model = model_data['model']
            self.LogNNet_best_params = model_data['model_params']
            self.input_layer_data = model_data['input_layer_data']
            self.basic_params = model_data['basic_params']

            version_of_model = model_data.get('version', None)
            if version_of_model is None and version_of_model != self.LogNNet_version:
                print(f'WARNING. The version of the LogNNet package {self.LogNNet_version} does not match '
                      f'the version imported from the model - {version_of_model}. '
                      f'The functionality of the package is not guaranteed.')

            if self.LogNNet_best_params['num_rows_W'] == 0:
                self.basic_params['use_reservoir'] = False

        elif model_data['coefs'] is not None and model_data['bias'] is not None:
            raise Exception(f"The file {file_name} contains minimalistic data for LogNNet to work. "
                            f"Use a special script to unload data from this model")

        if self._use_debug_mode:
            print(f'Data from {file_name} was successfully loaded')

        return self

    def fit_MLP(self, X, y):
        """
        Fit the model MLP to data matrix X and targets y.
            :param X: (array-like): The input data.
            :param y: (array-like): The target values (class labels in classification, real numbers in regression).
            :return: (object): Returns a trained (MLPRegressor or MLPClassifier) model.
        """

        res_metric, self.mlp_model, self.input_layer_data = lognnet_evaluate_by_params(
            X=X,
            y=y,
            mlp_params=self.mlp_model.get_params(),
            lognnet_params=self.LogNNet_best_params,
            selected_metric=self.basic_params['selected_metric'],
            selected_metric_class=self.basic_params['selected_metric_class'],
            static_features=self.basic_params['static_features'],
            target=self.basic_params['target'])

        if self.basic_params.get('use_debug_mode', False):
            print(f"Metric '{self.basic_params['selected_metric']}' = {round(res_metric, 6)} (Fit MLP)")

        keys_to_remove = ['X', 'y', 'use_debug_mode']
        self.basic_params = {key: value for key, value in self.basic_params.items() if key not in keys_to_remove}

        return self.mlp_model

    def get_mask_feature(self) -> str:
        """
        Returns the mask of the used features of the input vector
            :return: (str): Mask feature
        """
        if self.input_layer_data is None:
            raise Exception("The LogNNet neural network model is not trained. "
                            "Use the 'fit' function or import the LogNNet model")

        return self.input_layer_data['prizn_binary']

    def get_LogNNet_params(self) -> dict:
        """
        Returns the best parameters LogNNet was trained with.
            :return: (dict): Parameters of the LogNNet model.
        """

        if len(self.LogNNet_best_params) == 0:
            raise Exception("The LogNNet neural network model is not trained. "
                            "Use the 'fit' function or import the LogNNet model")

        return self.LogNNet_best_params

class LogNNetRegressor(BaseLogNNet):
    def __init__(self,
                 num_rows_W=(10, 150),
                 limit_hidden_layers=((1, 60), (1, 35)),
                 learning_rate_init=(0.001, 0.1),
                 n_epochs=(5, 550),
                 n_f=-1,
                 selected_metric='r2',
                 num_folds=1,
                 num_particles=10,
                 num_threads=10,
                 num_iterations=10,
                 **kwargs):
        """
        Model LogNNet Regression.

            :param num_rows_W: (array-like of int or singular int value, optional): The element represents
                the number of rows in the reservoir. Default value to (10, 150).
            :param limit_hidden_layers: (array-like of int or singular int value, optional): The element represents
                the number of neurons in the hidden layer. Default value to ((1, 60), (1, 35)).
            :param learning_rate_init: (array-like of float or singular float value, optional):
                The range of learning rate values that the optimizer will use to adjust the model's parameters.
                Default value to (0.001, 0.1).
            :param n_epochs: (array-like of int or singular int value, optional): The range of the number of epochs
                for which the model will be trained. Default value to (5, 550).
            :param n_f: (array-like of int or singular int value, optional): This parameter defines the conditions
                for selecting features in the input vector. It supports three types of input:
                    1. A list of specific feature indices (e.g., [1, 2, 10] means only features at
                        indices 1, 2, and 10 will be used).
                    2. A range of feature indices as a tuple (e.g., (1, 100) means the PSO method will
                        determine the best features from index 1 to 100).
                    3. A single integer indicating the number of features to be used (e.g., 20 means the
                        PSO method will select the best combination of 20 features). If set to -1,
                        all features from the input vector will be used.
                Default value -1.
            :param selected_metric: (str, optional): The selected metric for evaluating the model's performance.
                Support metrics:
                    1. 'r2': R-squared score indicating the proportion of variance explained by the model.
                    2. 'pearson_corr': Pearson correlation coefficient between the true and predicted values.
                    3. 'mse': Mean Squared Error indicating the average squared difference between the true and
                        predicted values.
                    4. 'mae': Mean Absolute Error indicating the average absolute difference between the true and
                        predicted values.
                    5. 'rmse': Root Mean Squared Error indicating the square root of the average squared differences.
                Default value to 'r2'.
            :param num_folds: (int, optional): The number of folds for cross-validation of the model.
                Default value to 1.
            :param num_particles: (int, optional): The number of particles in the Particle Swarm Optimization (PSO)
                method, used for parameter optimization. Default value to 10.
            :param num_threads: (int, optional): The number of threads to be used during model training for
                parallel data processing. Default value to 10.
            :param num_iterations: (int, optional): The number of iterations of the optimization algorithm.
                Default value to 10.
        """

        self.kwargs = kwargs
        valid_options = ["r2", "pearson_corr", "mse", "mae", "rmse"]

        selected_metric = validate_param(selected_metric,
                                         expected_type=str,
                                         valid_options=valid_options,
                                         name_param='selected_metric') if selected_metric != '' else valid_options[0]

        super().__init__(
            num_rows_W=num_rows_W,
            limit_hidden_layers=limit_hidden_layers,
            learning_rate_init=learning_rate_init,
            n_epochs=n_epochs,
            n_f=n_f,
            selected_metric=selected_metric,
            selected_metric_class=None,
            num_folds=num_folds,
            num_particles=num_particles,
            num_threads=num_threads,
            num_iterations=num_iterations,
            **kwargs)

        self.basic_params['target'] = 'Regressor'

class LogNNetClassifier(BaseLogNNet):
    def __init__(self,
                 num_rows_W=(10, 150),
                 limit_hidden_layers=((1, 60), (1, 35)),
                 learning_rate_init=(0.001, 0.1),
                 n_epochs=(5, 550),
                 n_f=-1,
                 selected_metric='accuracy',
                 selected_metric_class=None,
                 num_folds=1,
                 num_particles=10,
                 num_threads=10,
                 num_iterations=10,
                 **kwargs):
        """
        LogNNet classification class

            :param num_rows_W: (array-like of int or singular int value, optional): The element represents
                the number of rows in the reservoir. Default value to (10, 150).
            :param limit_hidden_layers: (array-like of int or singular int value, optional): The element represents
                the number of neurons in the hidden layer. Default value to ((1, 60), (1, 35)).
            :param learning_rate_init: (array-like of float or singular float value, optional): The range of
                learning rate values that the optimizer will use to adjust the model's parameters.
                Default value to (0.001, 0.1).
            :param n_epochs: (array-like of int or singular int value, optional): The range of the number of epochs
                for which the model will be trained. Default value to (5, 550).
            :param n_f: (array-like of int or singular int value, optional): This parameter defines the conditions
                for selecting features in the input vector. It supports three types of input:
                    1. A list of specific feature indices (e.g., [1, 2, 10] means only features at
                        indices 1, 2, and 10 will be used).
                    2. A range of feature indices as a tuple (e.g., (1, 100) means the PSO method will
                        determine the best features from index 1 to 100).
                    3. A single integer indicating the number of features to be used (e.g., 20 means the
                        PSO method will select the best combination of 20 features). If set to -1,
                        all features from the input vector will be used.
                Default value to -1.
            :param selected_metric: (str, optional): The selected metric for evaluating the model's performance.
                Support metrics:
                1. 'mcc': Matthews Correlation Coefficient indicating classification quality.
                2. 'precision': Precision score.
                3. 'recall': Recall score.
                4. 'f1': F1 score.
                5. 'accuracy': Accuracy score of the classifier.
                Default value to 'accuracy'.
            :param selected_metric_class: (int or None, optional): Select a class for training model.
                Default is None.
            :param num_folds: (int, optional): The number of folds for cross-validation of the model.
                Default value to 1.
            :param num_particles: (int, optional): The number of particles in the Particle Swarm Optimization (PSO)
                method, used for parameter optimization. Default value to 10.
            :param num_threads: (int, optional): The number of threads to be used during model training for
                parallel data processing. Default value to 10.
            :param num_iterations: (int, optional): The number of iterations of the optimization algorithm.
                Default value to 10.
        """

        self.kwargs = kwargs

        valid_options = ["accuracy", "mcc", "precision", "recall", "f1"]

        selected_metric = validate_param(selected_metric,
                                         expected_type=str,
                                         valid_options=valid_options,
                                         name_param='selected_metric') if selected_metric != '' else valid_options[0]

        if (any(keyword in selected_metric for keyword in ["precision", "recall", "f1"])
                and selected_metric_class is None):
            selected_metric_class = 1
        elif any(keyword in selected_metric for keyword in ["mcc", "accuracy"]):
            selected_metric_class = None

        super().__init__(
            num_rows_W=num_rows_W,
            limit_hidden_layers=limit_hidden_layers,
            learning_rate_init=learning_rate_init,
            n_epochs=n_epochs,
            n_f=n_f,
            selected_metric=selected_metric,
            selected_metric_class=selected_metric_class,
            num_folds=num_folds,
            num_particles=num_particles,
            num_threads=num_threads,
            num_iterations=num_iterations,
            **kwargs
        )

        self.basic_params['target'] = 'Classifier'


def main():
    pass


if __name__ == "__main__":
    main()
