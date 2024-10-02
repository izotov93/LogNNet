# -*- coding: utf-8 -*-

"""
Created on Thu Aug 17 10:00:00 2024

@author: Yuriy Izotov
@author: Andrei Velichko
@user: izotov93
"""

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, accuracy_score
from LogNNet import utility


def evaluate_mlp_regressor(X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, params: dict) -> (np.ndarray, MLPRegressor):
    """
    Train and evaluate a Multi-layer Perceptron (MLP) regressor.

    This function initializes an MLP regressor with specified parameters,
    trains it on the provided training data, and then makes predictions on
    the test data.

        :param X_train: (np.ndarray): The input features for training the model. Shape: (n_samples, n_features).
        :param X_test: (np.ndarray): The input features for testing the model. Shape: (n_samples, n_features).
        :param y_train: (np.ndarray): The target values for training the model. Shape: (n_samples,).
        :param params: (dict): A dictionary containing configuration parameters for the MLP regressor.
        :return: tuple: A tuple containing:
            - y_pred (np.ndarray): Predicted target values for the test set.
            - model (MLPRegressor): The trained MLP model instance.
    """

    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, model


def evaluate_mlp_classifier(X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, params: dict) -> (np.ndarray, MLPClassifier):
    """
    Train and evaluate a Multi-layer Perceptron (MLP) classifier.

    This function initializes an MLP classifier with specified parameters,
    fits it to the provided training data, and makes predictions on the test data.

        :param X_train: (np.ndarray): The input features for training the model. Shape: (n_samples, n_features).
        :param X_test: (np.ndarray): The input features for testing the model. Shape: (n_samples, n_features).
        :param y_train: (np.ndarray): The target values for training the model. Shape: (n_samples,).
        :param params: (dict): A dictionary containing configuration parameters for the MLP regressor.
        :return: tuple: A tuple containing:
            - y_pred (np.ndarray): Predicted target values for the test set.
            - model (MLPClassifier): The trained MLP model instance.
    """

    model = MLPClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, model


def evaluate_mlp_mod(X: np.ndarray, y: np.ndarray, params: dict, num_folds=5, num_rows_W=10,
                     Zn0=100, Cint=45, Bint=-43, Lint=3025, prizn=123, n_f=100, ngen=100,
                     shuffle=True, random_state=42, target='Regressor',
                     static_features=None) -> (dict, object, dict):
    """
    Evaluates Multi-Layer Perceptron (MLP) models using cross-validation.

        :param X: (np.ndarray): Input features for training, shaped as (n_samples, n_features).
        :param y: (np.ndarray): Target variables for the input features, shaped as (n_samples,).
        :param params: (dict): Parameters for training the model, which may vary depending on whether
            it's regression or classification.
        :param num_folds: (int, optional): Number of folds for cross-validation. Default value to 5.
        :param num_rows_W: (int, optional): Number of rows in the weight matrix W. Default value to 10.
        :param Zn0: (float, optional): Initial value for weights W. Default value to 100.
        :param Cint: (float, optional): Constant parameter for weight initialization. Default value to 45.
        :param Bint: (float, optional): Constant parameter for weight initialization. Default value to -43.
        :param Lint: (float, optional): Long-term parameter for weight initialization. Default value to 3025.
        :param prizn: (int, optional): Feature to be transformed into a binary representation. Default value to 123.
        :param n_f: (int, optional): Parameter for modifying the binary feature string. Default value to 100.
        :param ngen: (int, optional): Number of generations for modifying the binary string. Default value to 100.
        :param shuffle: (bool, optional): A parameter indicating that the data will be shuffled before
                splitting into training and testing sets. Default is True.
        :param random_state: (int, optional): Random state for reproducibility. Default value to 42.
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

    Shmax, Shmin = None, None
    X_train_max, X_train_min = None, None
    metrics, model = None, None

    all_y_true, all_y_pred = [], []
    mcc_scores, precision_scores, recall_scores, f1_scores, accuracy_scores = [], [], [], [], []

    if num_folds == 1:
        # np.random.permutation(len(X))
        train_index, test_index = np.arange(len(X)), np.arange(len(X))

        def single_split():
            yield train_index, test_index

        kf = single_split()
    else:
        kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state).split(X)

    gray_prizn = utility.decimal_to_gray(prizn)
    prizn_binary = utility.binary_representation(gray_prizn, input_dim)
    prizn_binary = utility.modify_binary_string(binary_string=prizn_binary,
                                                N=n_f, NG=ngen,
                                                static_features=static_features)

    for i in range(input_dim):
        if prizn_binary[i] == '0':
            X[:, i] = 0

    W = utility.initialize_W(num_rows_W=num_rows_W, input_dim=input_dim, Zn0=Zn0, Cint=Cint, Bint=Bint, Lint=Lint)

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_min, X_train_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
        denominator = X_train_max - X_train_min
        denominator[denominator == 0] = 1

        X_train, X_test = utility.normalize_data2(X_train, X_test, X_train_min, denominator)
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

        if target == 'Regressor':
            y_pred, model = evaluate_mlp_regressor(X_new_train_Sh, X_new_test_Sh, y_train, params)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        elif target == 'Classifier':
            y_pred, model = evaluate_mlp_classifier(X_new_train_Sh, X_new_test_Sh, y_train, params)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

            mcc_scores.append(matthews_corrcoef(y_test, y_pred))
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_scores.append(accuracy_score(y_test, y_pred))

    if target == 'Regressor':
        metrics = utility.calculate_metrics_for_regressor(all_y_true, all_y_pred)
    elif target == 'Classifier':
        metrics = utility.calculate_metrics_for_classifier(all_y_true, all_y_pred,
                                                           mcc_scores, precision_scores,
                                                           recall_scores, f1_scores, accuracy_scores)

    input_layers_data = {
        'W': W,
        'prizn_binary': prizn_binary,
        'Shmax': Shmax,
        'Shmin': Shmin,
        'X_train_max': X_train_max,
        'X_train_min': X_train_min
    }

    return metrics, model, input_layers_data


def testing_model_on_all_data(X: np.ndarray, y: np.ndarray, params: dict,
                              prizn_binary: str, W: np.ndarray, target='Regressor'):
    """
    Test the trained MLP model on the entire dataset.

    This function takes the trained MLP model, input features, target variables, and other
    relevant parameters, and evaluates the model's performance on the entire dataset.
        :param X: (np.ndarray): Input features for training, shaped as (n_samples, n_features).
        :param y: (np.ndarray): Target variables for the input features, shaped as (n_samples,).
        :param params: (dict): Parameters for training the model, which may vary depending on whether
            it's regression or classification.
        :param prizn_binary: (str): Binary representation of the feature to be transformed.
        :param W: (np.ndarray): Weight matrix used for the input layer transformation.
        :param random_state: (int, optional): Random state for reproducibility. Default value to 42.
        :param target: (str, optional): The type of prediction task: 'Regressor' for regression or
            'Classifier' for classification. Default value 'Regressor'.
        :return: (tuple): Tuple containing the params:
                - metrics: (dict) Performance metrics of the model, varying depending on whether the
            task is regression or classification.
                - model: (object) The trained MLP model.
                - input_layers_data: (dict) Data related to the input layers, including weights W and
            other normalization parameters.
    """

    model, metrics = None, None

    for i in range(X.shape[1]):
        if prizn_binary[i] == '0':
            X[:, i] = 0

    X_train_min, X_train_max = np.min(X, axis=0), np.max(X, axis=0)
    denominator = X_train_max - X_train_min
    denominator[denominator == 0] = 1

    X_new_train = np.dot(utility.normalize_data(X), W.T)
    Shmax, Shmin = np.max(X_new_train, axis=0), np.min(X_new_train, axis=0)
    d = Shmax - Shmin
    Shmax = Shmax + d * 0.25
    Shmin = Shmin - d * 0.25
    denominator_Sh = Shmax - Shmin
    denominator_Sh[denominator_Sh == 0] = 1

    X_new_train_Sh = (X_new_train - Shmin) / denominator_Sh - 0.5

    if target == 'Regressor':
        y_pred, model = evaluate_mlp_regressor(X_new_train_Sh, X_new_train_Sh, y, params)

        metrics = utility.calculate_metrics_for_regressor(y, y_pred)

    elif target == 'Classifier':
        y_pred, model = evaluate_mlp_classifier(X_new_train_Sh, X_new_train_Sh, y, params)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average=None,
                                                                   zero_division=0)

        metrics = utility.calculate_metrics_for_classifier(y, y_pred, matthews_corrcoef(y, y_pred),
                                                           precision, recall, f1,
                                                           accuracy_score(y, y_pred))

    input_layers_data = {
        'W': W,
        'prizn_binary': prizn_binary,
        'Shmax': Shmax,
        'Shmin': Shmin,
        'X_train_max': X_train_max,
        'X_train_min': X_train_min
    }

    return metrics, model, input_layers_data
