# -*- coding: utf-8 -*-

"""
Created on Thu Aug 17 10:00:00 2024

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
from LogNNet.mlp_evaluation import testing_model_on_all_data
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings

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

    elif isinstance(param, (int, float)) and param <= 0 and name_param != 'noise':
        raise ValueError(f'The parameter "{name_param}" must be positive number. Value {param} is not supported')

    elif isinstance(param, (int, float)) and param < 0 and name_param == 'noise':
        raise ValueError(f'The parameter "{name_param}" must be non-negative number. Value {param} is not supported')

    else:
        return param


def validate_dict_mlp_param(input_dict: dict) -> (dict, None):
    if not isinstance(input_dict, dict):
        return None
    else:
        default_params = {
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'power_t': 0.5,
            'shuffle': True,
            'random_state': None,
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
        validated_params = {}

        for key, default_value in default_params.items():
            if key in input_dict and input_dict[key] != default_value:
                validated_params[key] = input_dict[key]

        return validated_params


class BaseLogNNet(object):
    lib_version = '1.4'

    def __init__(self,
                 input_layer_neurons: (tuple, int),
                 first_layer_neurons: (tuple, int),
                 hidden_layer_neurons: (tuple, int),
                 learning_rate_init: (tuple, float),
                 n_epochs: (tuple, int),
                 n_f: (tuple, int),
                 ngen: (tuple, int),
                 selected_metric: str,
                 selected_metric_class: (None, int),
                 num_folds: int,
                 num_particles: int,
                 num_threads: int,
                 num_iterations: int,
                 **kwargs):

        self.input_layer_data = None
        self._LogNNet_global_best_position = None
        self._LogNNet_global_best_fitness = None
        self.mlp_model = None
        self.LogNNet_best_params = {}

        self.mlp_params = validate_dict_mlp_param(kwargs)

        self._param_ranges = {
            'num_rows_W': validate_param(input_layer_neurons, int,
                                         check_limits=True, name_param='input_layer_neurons'),
            'Zn0': (-499, 499),
            'Cint': (-499, 499),
            'Bint': (-499, 499),
            'Lint': (100, 10000),
            'first_layer_neurons': validate_param(first_layer_neurons, int,
                                                  check_limits=True, name_param='first_layer_neurons'),
            'hidden_layer_neurons': validate_param(hidden_layer_neurons, int,
                                                   check_limits=True, name_param='hidden_layer_neurons'),
            'learning_rate_init': validate_param(learning_rate_init, float,
                                                 check_limits=True, name_param='learning_rate_init'),
            'epochs': validate_param(n_epochs, int, check_limits=True, name_param='n_epochs'),
            'prizn': (0, 1),
            'n_f': n_f,
            'ngen': validate_param(ngen, int, check_limits=True, name_param='ngen')
        }

        self.basic_params = {
            'X': None,
            'y': None,
            'num_folds': validate_param(num_folds, int, name_param='num_folds'),
            'param_ranges': self._param_ranges,
            'selected_metric': selected_metric,
            'selected_metric_class': selected_metric_class,
            'dimensions': len(self._param_ranges),
            'num_particles': validate_param(num_particles, int, name_param='num_particles'),
            'num_threads': validate_param(num_threads, int, name_param='num_threads'),
            'num_iterations': validate_param(num_iterations, int, name_param='num_iterations'),
            'target': None,
            'static_features': None,
            'mlp_params': self.mlp_params
        }

    def __version__(self):
        return self.lib_version

    def fit(self, X: np.ndarray, y: np.ndarray) -> (MLPRegressor, MLPClassifier):
        """
        Fit the model to data matrix X and targets y.
            :param X: (ndarray): The input data.
            :param y: (ndarray): The target values (class labels in classification, real numbers in regression).
            :return: (object): Returns a trained (MLPRegressor or MLPClassifier) model.
        """

        self.basic_params['X'] = X
        self.basic_params['y'] = y

        self._param_ranges['prizn'] = (*self._param_ranges['prizn'][:1], 2 ** X.shape[1] - 1)

        if isinstance(self._param_ranges['n_f'], int):
            if self._param_ranges['n_f'] == -1:
                self._param_ranges['n_f'] = (X.shape[1], X.shape[1])
            elif self._param_ranges['n_f'] > 0:
                self._param_ranges['n_f'] = (int(self._param_ranges['n_f']), int(self._param_ranges['n_f']))
            else:
                raise ValueError("Invalid value for 'n_f'. Allowed values are -1 (all features) or positive numbers")

            self.basic_params['static_features'] = None
            self._param_ranges['ngen'] = (1, 1)

        elif isinstance(self._param_ranges['n_f'], tuple) and len(self._param_ranges['n_f']) == 2:
            self._param_ranges['n_f'] = (max(1, self._param_ranges['n_f'][0]),
                                         min(self._param_ranges['n_f'][1], X.shape[1]))
            self.basic_params['static_features'] = None

        elif isinstance(self._param_ranges['n_f'], list):
            self.basic_params['static_features'] = self._param_ranges['n_f']
            self._param_ranges['n_f'] = (0, 0)
            self._param_ranges['ngen'] = (0, 0)

        else:
            raise ValueError("Invalid value for 'n_f'. Support types: int, tuple or list")

        if (self.basic_params['selected_metric_class'] is not None and
                self.basic_params['target'] == 'Classifier' and
                (self.basic_params['selected_metric_class'] > int(np.max(y, axis=0)) or
                 self.basic_params['selected_metric_class'] < 0)):
            raise ValueError(f"Wrong param 'selected_metric_class'. "
                             f"Validate limits - (0, {int(np.max(y, axis=0))})")

        self.basic_params['param_ranges'] = self._param_ranges

        (self._LogNNet_global_best_position, self._LogNNet_global_best_fitness,
         self.mlp_model, self.input_layer_data) = PSO(**self.basic_params)

        self.LogNNet_best_params = {
            'num_rows_W': int(self._LogNNet_global_best_position[0]),
            'Zn0': self._LogNNet_global_best_position[1],
            'Cint': self._LogNNet_global_best_position[2],
            'Bint': self._LogNNet_global_best_position[3],
            'Lint': self._LogNNet_global_best_position[4],
            'first_layer_neurons': int(self._LogNNet_global_best_position[5]),
            'hidden_layer_neurons': int(self._LogNNet_global_best_position[6]),
            'learning_rate_init': float(self._LogNNet_global_best_position[7]),
            'epochs': int(self._LogNNet_global_best_position[8]),
            'prizn': int(self._LogNNet_global_best_position[9]),
            'n_f': int(self._LogNNet_global_best_position[10]),
            'ngen': int(self._LogNNet_global_best_position[11]),
        }

        params = {
            'hidden_layer_sizes': (self.LogNNet_best_params['first_layer_neurons'],
                                   self.LogNNet_best_params['hidden_layer_neurons']),
            'learning_rate_init': self.LogNNet_best_params['learning_rate_init'],
            'max_iter': self.LogNNet_best_params['epochs'],
        }
        if self.mlp_params is not None:
            params.update(self.mlp_params)

        metrics, self.mlp_model, self.input_layer_data = (
            testing_model_on_all_data(X=X, y=y, params=params,
                                      prizn_binary=self.input_layer_data['prizn_binary'],
                                      W=self.input_layer_data['W'],
                                      target=self.basic_params['target']))

        res_metric = metrics[self.basic_params['selected_metric']] if (
                self.basic_params['selected_metric_class'] is None) else (
            metrics)[self.basic_params['selected_metric']][self.basic_params['selected_metric_class']]

        print(f"Final value metric {self.basic_params['selected_metric']} = {round(res_metric, 6)}")

        return self.mlp_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the LogNNet regressor.
            :param X: (np.ndarray): The input data.
            :return: (np.ndarray): The predicted classes.
        """

        if self.input_layer_data is None or self.mlp_model is None:
            raise Exception("The LogNNet neural network model is not trained. "
                            "Use the 'FIT' function before using the 'PREDICT' function.")

        for i in range(X.shape[1]):
            if self.input_layer_data['prizn_binary'][i] == '0':
                X[:, i] = 0

        denominator = np.array(self.input_layer_data['X_train_max']) - np.array(self.input_layer_data['X_train_min'])
        denominator[denominator == 0] = 1
        X_test_normalized = (X - np.array(self.input_layer_data['X_train_min'])) / denominator
        W = self.input_layer_data['W']
        X_new_test = np.dot(X_test_normalized, W.T)

        denominator_Sh = np.array(self.input_layer_data['Shmax']) - np.array(self.input_layer_data['Shmin'])
        denominator_Sh[denominator_Sh == 0] = 1
        X_new_test_Sh = (X_new_test - np.array(self.input_layer_data['Shmin'])) / denominator_Sh - 0.5

        return self.mlp_model.predict(X_new_test_Sh)

    def export_model(self, type_of_model='max'):
        """
        Save the trained LogNNet model and its parameters to a file.

        This method serializes and saves the best model and its associated
        parameters based on the specified type. The user can specify
        either 'max' to save the model with maximum performance or 'min'
        to save it with minimized parameters.

            :param type_of_model: (str): The type of model to save.
                - 'max': Saves the complete model and its parameters.
                - 'min': Saves only the model coefficients, biases, and input layer parameters.
            :return: (None) The saved model will be written to a file named with a timestamp
                    to ensure uniqueness, formatted as '{timestamp}_LogNNet_model_{type_of_model}.joblib'
        """
        if type_of_model == 'max':
            value = {'model': self.mlp_model,
                     'model_params': self._LogNNet_global_best_position,
                     'input_layer_data': self.input_layer_data}
        elif type_of_model == 'min':
            params_reservoir = {
                'num_rows_W': self._LogNNet_global_best_position['num_rows_W'],
                'Zn0': self._LogNNet_global_best_position['Zn0'],
                'Cint': self._LogNNet_global_best_position['Cint'],
                'Bint': self._LogNNet_global_best_position['Bint'],
                'Lint': self._LogNNet_global_best_position['Lint'],
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

        joblib.dump(value=value, filename=f'{int(time.time())}_LogNNet_model_{type_of_model}.joblib')

    def import_model(self, file_model_name: str):
        """
        Import a trained LogNNet model from a specified file.

        This method loads a serialized LogNNet model and its associated parameters
        from the provided file path. It checks for the existence of the file before
        attempting to load the model. Depending on the contents of the loaded
        data, it initializes the model and its parameters for further use.

            :param file_model_name: (str): The path to the file containing the serialized model.
            :return: (None) Fills an example of a class with data.
        """

        if not os.path.isfile(file_model_name):
            raise ValueError(f'File "{file_model_name}" not found. Check the file path and try again.')

        model_data = joblib.load(file_model_name)

        if model_data['model'] is not None:
            self.mlp_model = model_data['model']
            self._LogNNet_global_best_position = model_data['model_params']
            self.input_layer_data = model_data['input_layer_data']
        elif model_data['coefs'] is not None and model_data['bias'] is not None:
            raise Exception(f"The file {file_model_name} contains minimalistic data for LogNNet to work. "
                            f"Use a special script to unload data from this model")


class LogNNetRegressor(BaseLogNNet):
    def __init__(self,
                 input_layer_neurons=(10, 150),
                 first_layer_neurons=(1, 60),
                 hidden_layer_neurons=(1, 35),
                 learning_rate_init=(0.001, 0.01),
                 n_epochs=(5, 550),
                 n_f=-1,
                 ngen=(1, 500),
                 selected_metric='r2',
                 num_folds=1,
                 num_particles=10,
                 num_threads=10,
                 num_iterations=10,
                 **kwargs):
        """
        Model LogNNet Regression.

            :param input_layer_neurons: (array-like of int or singular int value, optional): The element represents
                the number of rows in the reservoir. Default value to (10, 150).
            :param first_layer_neurons: (array-like of int or singular int value, optional): The element represents
                the number of neurons in the first hidden layer. Default value to (1, 60).
            :param hidden_layer_neurons: (array-like of int or singular int value, optional): The element represents
                the number of neurons in the hidden layer. Default value to (1, 35).
            :param learning_rate_init: (array-like of float or singular float value, optional):
                The range of learning rate values that the optimizer will use to adjust the model's parameters.
                Default value to (0.001, 0.01).
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
            :param ngen: (array-like of int or singular int value, optional): The range of generations for
                the optimization algorithm that will be used to find the optimal model parameters.
                Default value to (1, 500).
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
            input_layer_neurons=input_layer_neurons,
            first_layer_neurons=first_layer_neurons,
            hidden_layer_neurons=hidden_layer_neurons,
            learning_rate_init=learning_rate_init,
            n_epochs=n_epochs,
            n_f=n_f,
            ngen=ngen,
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
                 input_layer_neurons=(10, 150),
                 first_layer_neurons=(1, 60),
                 hidden_layer_neurons=(1, 35),
                 learning_rate_init=(0.001, 0.01),
                 n_epochs=(5, 550),
                 n_f=-1,
                 ngen=(1, 500),
                 selected_metric='accuracy',
                 selected_metric_class=None,
                 num_folds=1,
                 num_particles=10,
                 num_threads=10,
                 num_iterations=10,
                 **kwargs):
        """
        LogNNet classification class

            :param input_layer_neurons: (array-like of int or singular int value, optional): The element represents
                the number of rows in the reservoir. Default value to (10, 150).
            :param first_layer_neurons: (array-like of int or singular int value, optional): The element represents
                the number of neurons in the first hidden layer. Default value to (1, 60).
            :param hidden_layer_neurons: (array-like of int or singular int value, optional): The element represents
                the number of neurons in the hidden layer. Default value to (1, 35).
            :param learning_rate_init: (array-like of float or singular float value, optional): The range of
                learning rate values that the optimizer will use to adjust the model's parameters.
                Default value to (0.001, 0.01).
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
            :param ngen: (array-like of int or singular int value, optional): The range of generations for the
                optimization algorithm that will be used to find the optimal model parameters.
                Default value to (1, 500).
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

        valid_options = ["accuracy", "mcc", "precision", "recall", "f1",
                         "avg_precision", "avg_recall", "avg_f1", "avg_accuracy", "avg_mcc"]

        selected_metric = validate_param(selected_metric,
                                         expected_type=str,
                                         valid_options=valid_options,
                                         name_param='selected_metric') if selected_metric != '' else valid_options[0]

        if (any(keyword in selected_metric for keyword in ["precision", "recall", "f1"])
                and selected_metric_class is None):
            selected_metric_class = 0
        elif any(keyword in selected_metric for keyword in ["mcc", "accuracy"]):
            selected_metric_class = None

        super().__init__(
            input_layer_neurons=input_layer_neurons,
            first_layer_neurons=first_layer_neurons,
            hidden_layer_neurons=hidden_layer_neurons,
            learning_rate_init=learning_rate_init,
            n_epochs=n_epochs,
            n_f=n_f,
            ngen=ngen,
            selected_metric=selected_metric,
            selected_metric_class=selected_metric_class,
            num_folds=num_folds,
            num_particles=num_particles,
            num_threads=num_threads,
            num_iterations=num_iterations,
            **kwargs
        )

        self.basic_params['target'] = 'Classifier'


if __name__ == "__main__":
    pass
