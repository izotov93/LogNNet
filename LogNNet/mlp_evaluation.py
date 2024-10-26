# -*- coding: utf-8 -*-

"""
Created on Aug 17 10:00 2024
Modified on Oct 18 17:00 2024
Modified on Oct 23 15:00 2024

@author: Yuriy Izotov
@author: Andrei Velichko
@user: izotov93
"""

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import pearsonr
from LogNNet import utility


def evaluate_mlp_mod(X: np.ndarray, y: np.ndarray, mlp_params: dict, selected_metric:str, selected_metric_class=None,
                     num_folds=2, num_rows_W=10, Zn0=100, Cint=45, Bint=-43, Lint=3025, prizn=123, n_f=100,
                     ngen=100, target='Regressor', static_features=None) -> (float, object):
    """
    Evaluates Multi-Layer Perceptron (MLP) models using cross-validation.

        :param X: (np.ndarray): Input features for training, shaped as (n_samples, n_features).
        :param y: (np.ndarray): Target variables for the input features, shaped as (n_samples,).
        :param mlp_params: (dict): Parameters for training the mlp model
        :param selected_metric: (str, optional): The selected metric for evaluating the model's performance.
        :param selected_metric_class: (int or None, optional): Select a class for training model.
                Default is None.
        :param num_folds: (int, optional): Number of folds for cross-validation. Default value to 5.
        :param num_rows_W: (int, optional): Number of rows in the weight matrix W. Default value to 10.
        :param Zn0: (float, optional): Initial value for weights W. Default value to 100.
        :param Cint: (float, optional): Constant parameter for weight initialization. Default value to 45.
        :param Bint: (float, optional): Constant parameter for weight initialization. Default value to -43.
        :param Lint: (float, optional): Long-term parameter for weight initialization. Default value to 3025.
        :param prizn: (int, optional): Feature to be transformed into a binary representation. Default value to 123.
        :param n_f: (int, optional): Parameter for modifying the binary feature string. Default value to 100.
        :param ngen: (int, optional): Number of generations for modifying the binary string. Default value to 100.
        :param target: (str, optional): The type of prediction task: 'Regressor' for regression or
            'Classifier' for classification. Default value 'Regressor'.
        :param static_features: (list or None, optional): List of input vector features to be used. Default is None.
        :return: (tuple): Tuple containing the params:
                - metrics: (dict) Performance metrics of the model, varying depending on whether the
            task is regression or classification.
                - model: (object) The trained MLP model.
                - input_layers_data: (dict) Data related to the input layers, including weights W and
            other normalization parameters.
    """

    input_dim = X.shape[1]
    all_y_true, all_y_pred = [], []

    if num_folds == 1:
        # np.random.permutation(len(X))
        list_array = np.arange(len(X))
        kf = [(list_array, list_array)]
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42).split(X)

    gray_prizn = utility.decimal_to_gray(prizn)
    prizn_binary = utility.binary_representation(gray_prizn, input_dim)
    prizn_binary = utility.modify_binary_string(binary_string=prizn_binary,
                                                N=n_f, NG=ngen,
                                                static_features=static_features)

    for i in range(input_dim):
        if prizn_binary[i] == '0':
            X[:, i] = 0

    use_reservoir = True
    if num_rows_W != 0:
        W = utility.initialize_W(num_rows_W=num_rows_W,
                                 input_dim=input_dim,
                                 Zn0=Zn0,
                                 Cint=Cint,
                                 Bint=Bint,
                                 Lint=Lint)
    else:
        use_reservoir = False

    mlp_model = None
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_min, X_train_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
        denominator = X_train_max - X_train_min
        denominator[denominator == 0] = 1

        X_train, X_test = utility.normalize_data2(X_train=X_train,
                                                  X_test=X_test,
                                                  X_train_min=X_train_min,
                                                  denominator=denominator)

        if use_reservoir:
            X_new_train = np.dot(X_train, W.T)
            X_new_test = np.dot(X_test, W.T)

            Shmax, Shmin = np.max(X_new_train, axis=0), np.min(X_new_train, axis=0)
            d = Shmax - Shmin
            Shmax = Shmax + d * 0.25
            Shmin = Shmin - d * 0.25
            denominator_Sh = Shmax - Shmin
            denominator_Sh[denominator_Sh == 0] = 1

            X_new_train_Sh = (X_new_train - Shmin) / denominator_Sh - 0.5
            X_new_test_Sh = (X_new_test - Shmin) / denominator_Sh - 0.5
        else:
            X_new_train_Sh, X_new_test_Sh = X_train, X_test

        if target == 'Regressor':
            mlp_model = MLPRegressor(**mlp_params)
        elif target == 'Classifier':
            mlp_model = MLPClassifier(**mlp_params)

        mlp_model.fit(X_new_train_Sh, y_train)
        y_pred = mlp_model.predict(X_new_test_Sh)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    metric_value = calculation_metric_value(all_y_true=all_y_true, all_y_pred=all_y_pred,
                                    target=target, selected_metric=selected_metric,
                                    selected_metric_class=selected_metric_class)

    return metric_value, mlp_model


def lognnet_evaluate_by_params(X: np.ndarray, y: np.ndarray, mlp_params: dict,
                               lognnet_params: dict, selected_metric:str, selected_metric_class=None,
                               static_features=None, target='Regressor') -> (float, object, dict):
    """
    Test the trained MLP model on the entire dataset.

    This function takes the trained MLP model, input features, target variables, and other
    relevant parameters, and evaluates the model's performance on the entire dataset.
        :param X: (np.ndarray): Input features for training, shaped as (n_samples, n_features).
        :param y: (np.ndarray): Target variables for the input features, shaped as (n_samples,).
        :param mlp_params: (dict): Parameters for mlp model.
        :param lognnet_params: (dict): Parameters for lognnet model.
        :param selected_metric: (str, optional): The selected metric for evaluating the model's performance.
        :param selected_metric_class: (int or None, optional): Select a class for training model.
        :param static_features: (list or None, optional): List of input vector features to be used. Default is None.
        :param target: (str, optional): The type of prediction task: 'Regressor' for regression or
            'Classifier' for classification. Default value 'Regressor'.
        :return: (tuple): Tuple containing the params:
                - metrics: (dict) Performance metrics of the model, varying depending on whether the
            task is regression or classification.
                - model: (object) The trained MLP model.
                - input_layers_data: (dict) Data related to the input layers, including weights W and
            other normalization parameters.
    """
    W, Shmin, Shmax = None, None, None
    mlp_model, metrics = None, None

    gray_prizn = utility.decimal_to_gray(lognnet_params['prizn'])
    prizn_binary = utility.binary_representation(gray_prizn, X.shape[1])
    prizn_binary = utility.modify_binary_string(binary_string=prizn_binary,
                                                N=lognnet_params['n_f'],
                                                NG=lognnet_params['ngen'],
                                                static_features=static_features)

    for i in range(X.shape[1]):
        if prizn_binary[i] == '0':
            X[:, i] = 0

    X_train_min, X_train_max = np.min(X, axis=0), np.max(X, axis=0)
    denominator = X_train_max - X_train_min
    denominator[denominator == 0] = 1

    if lognnet_params['num_rows_W'] != 0:
        W = utility.initialize_W(num_rows_W=lognnet_params['num_rows_W'],
                                 input_dim=X.shape[1],
                                 Zn0=lognnet_params['Zn0'],
                                 Cint=lognnet_params['Cint'],
                                 Bint=lognnet_params['Bint'],
                                 Lint=lognnet_params['Lint'])

        X_new_train = np.dot(utility.normalize_data(X), W.T)
        Shmax, Shmin = np.max(X_new_train, axis=0), np.min(X_new_train, axis=0)
        d = Shmax - Shmin
        Shmax = Shmax + d * 0.25
        Shmin = Shmin - d * 0.25
        denominator_Sh = Shmax - Shmin
        denominator_Sh[denominator_Sh == 0] = 1

        X_new_train_Sh = (X_new_train - Shmin) / denominator_Sh - 0.5
    else:
        X_new_train_Sh = utility.normalize_data(X)

    if target == 'Regressor':
        mlp_model = MLPRegressor(**mlp_params)
    elif target == 'Classifier':
        mlp_model = MLPClassifier(**mlp_params)

    mlp_model.fit(X_new_train_Sh, y)
    y_pred = mlp_model.predict(X_new_train_Sh)

    metric_value = calculation_metric_value(all_y_true=y, all_y_pred=y_pred,
                                            target=target, selected_metric=selected_metric,
                                            selected_metric_class=selected_metric_class)

    input_layers_data = {
        'W': W,
        'prizn_binary': prizn_binary,
        'Shmax': Shmax,
        'Shmin': Shmin,
        'X_train_max': X_train_max,
        'X_train_min': X_train_min
    }

    return metric_value, mlp_model, input_layers_data


def calculation_metric_value(all_y_true, all_y_pred, target: str,
                             selected_metric: str, selected_metric_class=None) -> float:
    """
    Calculate the specified metric value for regression or classification tasks.

        :param all_y_true: (array-like): Ground truth target values.
        :param all_y_pred: (array-like): Estimated target values.
        :param target: (str): Type of the target ("Regressor" or "Classifier").
        :param selected_metric: (str): The metric to compute (e.g., 'r2', 'mse', 'accuracy', 'f1', etc.).
        :param selected_metric_class: (int, optional): Class index for multi-class metrics (only for classifiers).
        :return: The computed metric value.
    """

    if target == 'Regressor':
        if selected_metric == 'r2':
            return r2_score(all_y_true, all_y_pred)
        elif selected_metric == 'pearson_corr':
            if np.all(all_y_true == all_y_true[0]) or np.all(all_y_pred == all_y_pred[0]):
                return 0
            else:
                metric_value, _ = pearsonr(all_y_true, all_y_pred)
                return metric_value
        elif selected_metric in ['mse', 'rmse']:
            mse = mean_squared_error(all_y_true, all_y_pred)
            if selected_metric == 'mse':
                return mse
            else:
                return np.sqrt(mse)
        elif selected_metric == 'mae':
            return mean_absolute_error(all_y_true, all_y_pred)

    elif target == 'Classifier':
        if selected_metric == 'mcc':
            return matthews_corrcoef(all_y_true, all_y_pred)
        elif selected_metric == 'accuracy':
            return accuracy_score(all_y_true, all_y_pred)
        elif selected_metric == 'precision':
            return precision_score(all_y_true, all_y_pred,
                                           average=None, zero_division=0)[selected_metric_class]
        elif selected_metric == 'recall':
            return recall_score(all_y_true, all_y_pred,
                                        average=None, zero_division=0)[selected_metric_class]
        elif selected_metric == 'f1':
            return f1_score(all_y_true, all_y_pred,
                                    average=None, zero_division=0)[selected_metric_class]

    else:
        raise ValueError(f"Unknown target type '{target}'")

    raise ValueError(f"Unknown metric '{selected_metric}'")


def main():
    pass


if __name__ == "__main__":
    main()

