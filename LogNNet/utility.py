# -*- coding: utf-8 -*-

"""
Created on Thu Aug 17 10:00:00 2024

@author: Yuriy Izotov
@author: Andrei Velichko
@user: izotov93
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (matthews_corrcoef, precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score)
from scipy.stats import pearsonr


def normalize_data2(X_train: np.ndarray, X_test: np.ndarray,
                    X_train_min: np.ndarray, denominator: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Normalize the training and test data.

        :param X_train: (numpy.ndarray): The training data that needs to be normalized
        :param X_test: (numpy.ndarray): The test data that needs to be normalized
        :param X_train_min: (float or numpy.ndarray): The minimum value of the training data,
                                           which is used for normalization.
        :param denominator: (float or numpy.ndarray): The value by which the difference
                                           between the data and the minimum value is divided
        :return: tuple: A tuple containing the normalized training and test data.
                            (train_normalized: numpy.ndarray, test_normalized: numpy.ndarray)
    """
    X_train_normalized = (X_train - X_train_min) / denominator
    X_test_normalized = (X_test - X_train_min) / denominator
    return X_train_normalized, X_test_normalized


def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Function to normalize data using Min-Max Scaling method.

        :param X: (np.ndarray): Array with training data
        :return:(np.ndarray): Normalized training and testing data
    """

    X_train_min = np.min(X, axis=0)
    X_train_max = np.max(X, axis=0)
    denominator = X_train_max - X_train_min
    denominator[denominator == 0] = 1

    X_train_normalized = (X - X_train_min) / denominator

    return X_train_normalized


def initialize_W(num_rows_W: int, input_dim: int, Zn0: int,
                 Cint: int, Bint: int, Lint: int) -> np.ndarray:
    """
    Initialize a weight matrix W using a specific deterministic algorithm.

    This function creates a weight matrix W of shape (num_rows_W, input_dim)
    where each entry is computed using a linear transformation based on the input parameters.
    The transformation is defined as:
        zn = (Cint - Bint * zn) % Lint,
    starting from an initial value Zn0.

        :param num_rows_W: (int): The number of rows in the weight matrix W.
        :param input_dim: (int): The number of columns in the weight matrix W.
        :param Zn0: (int): The initial value used for computing the entries of W.
        :param Cint: (int): Constant the linear congruential generator for computing the entries of W
        :param Bint: (int): Constant the linear congruential generator for computing the entries of W
        :param Lint: (int): Constant the linear congruential generator for computing the entries of W
        :return: numpy.ndarray: A weight matrix W scaled to the range [-0.5, 0.5]

    """
    W = np.zeros((num_rows_W, input_dim))
    zn = Zn0
    for i in range(num_rows_W):
        for j in range(input_dim):
            zn = Cint - Bint * zn
            zn = zn % Lint
            W[i, j] = zn
    W = W / Lint - 0.5
    return W


def congruential_generator(seed: int, n: int) -> list:
    """
    Generate a sequence of pseudo-random numbers using a linear congruential generator.

    This function implements a linear congruential generator (LCG) with default parameters.
    The sequence is generated using the formula:
    x_{n+1} = (a * x_n + c) % m,
    where `a`, `c`, and `m` are constants, and `x` is the current value (or seed).

        :param seed: (int): The initial value used to generate the sequence of pseudo-random numbers.
        :param n: (int): The number of pseudo-random numbers to generate.
        :return: list: A list of `n` pseudo-random numbers uniformly distributed in the range [0, 100].
    """
    a = 1103515245
    c = 12345
    m = 2 ** 31
    x = seed
    numbers = []
    for _ in range(n):
        x = (a * x + c) % m
        numbers.append(x % 101)
    return numbers


def modify_binary_string(binary_string: str, N: int, NG: int, static_features=None) -> str:
    """
    The function adjusts the input binary string to have exactly N ones. If the number of ones in
    the input string matches N or N is set to 0, the input string is returned unchanged. If N exceeds
    the length of the string, all zeros are replaced with ones. If the number of ones is less than N,
    a random selection of zeros is replaced with ones based on a congruential number generator with
    the seed NG.

        :param binary_string: (str): The input binary string.
        :param N: (int): The desired number of ones in the modified string.
        :param NG: (int): The initial seed for number generation
        :param static_features: (list or None): A list of static values to add to the modified string.
        :return: str: The modified binary string based on the specified criteria
    """
    binary_list = list(binary_string)

    if static_features is None:
        current_ones = binary_list.count('1')
        if current_ones == N or N == 0:
            return binary_string
        elif N > len(binary_list):
            return '1' * len(binary_list)

        NZ = abs(current_ones - N)
        generated_numbers = congruential_generator(NG, NG + NZ)[NG:]

        i = 0
        while NZ > 0:
            pos = generated_numbers[i % len(generated_numbers)]
            idx = pos % len(binary_list)

            if current_ones > N:
                while binary_list[idx] != '1':
                    idx = (idx + 1) % len(binary_list)
                binary_list[idx] = '0'
                current_ones -= 1
            else:
                while binary_list[idx] != '0':
                    idx = (idx + 1) % len(binary_list)
                binary_list[idx] = '1'

            NZ -= 1
            i += 1
    else:
        binary_list = ['0'] * len(binary_list)
        for index in static_features:
            if index >= len(binary_string):
                index = len(binary_string) - 1
            binary_list[index] = '1'

    return ''.join(binary_list)


def decimal_to_gray(n: int) -> int:
    """
    Convert a decimal number to its Gray code equivalent

        :param n: (int): Decimal number to convert
        :return: (int) Gray code equivalent of the input decimal number
        """
    return n ^ (n >> 1)


def binary_representation(n: int, num_bits: int) -> str:
    """
    Returns the binary representation of a decimal number padded with zeros to
    a specified number of bits.

        :param n: (int): Decimal number to convert to binary
        :param num_bits: (int): Number of bits to represent the binary number
        :return: (int): Binary representation of the input decimal number with zero-padding
                to the specified number of bits.
        """
    return bin(n)[2:].zfill(num_bits)


def calculate_metrics_for_regressor(all_y_true, all_y_pred) -> dict:
    """
    Calculate evaluation metrics for a regression model.

    This function computes several common regression metrics based
    on the true and predicted values provided. The metrics calculated
    include R-squared (R²), Pearson correlation coefficient, Mean Squared
    Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

        :param all_y_true: (list): The true target values
        :param all_y_pred: (list): The predicted values from the regression model
        :return: dict: A dictionary containing the following metrics:
            - "r2": R-squared score indicating the proportion of variance explained by the model.
            - "pearson_corr": Pearson correlation coefficient between the true and predicted values.
            - "mse": Mean Squared Error indicating the average squared difference between the true and predicted
            values.
            - "mae": Mean Absolute Error indicating the average absolute difference between the true and predicted
            values.
            - "rmse": Root Mean Squared Error indicating the square root of the average squared differences.
    """

    if np.all(all_y_true == all_y_true[0]) or np.all(all_y_pred == all_y_pred[0]):
        pearson_corr = 0
    else:
        pearson_corr, _ = pearsonr(all_y_true, all_y_pred)
    mse = mean_squared_error(all_y_true, all_y_pred)

    return {
        "r2": r2_score(all_y_true, all_y_pred),
        "pearson_corr": pearson_corr,
        "mse": mse,
        "mae": mean_absolute_error(all_y_true, all_y_pred),
        "rmse": np.sqrt(mse)
    }


def calculate_metrics_for_classifier(all_y_true, all_y_pred,
                                     mcc_scores=None, precision_scores=None,
                                     recall_scores=None, f1_scores=None,
                                     accuracy_scores=None) -> dict:
    """
    Calculate evaluation metrics for a classification model.

    This function computes a range of performance metrics based on
    the true and predicted values provided for a classifier. The metrics
    include Matthews Correlation Coefficient (MCC), precision, recall,
    F1 score, accuracy, and a confusion matrix.

        :param all_y_true: (list): The true labels for the data
        :param all_y_pred: (list): The predicted labels from the classification model
        :param mcc_scores: (list, optional): List of MCC scores to compute the average. Default is None
        :param precision_scores: (list, optional): List of precision scores to compute the average.
                                    Default is None
        :param recall_scores: (list, optional): List of recall scores to compute the average. Default is None
        :param f1_scores: (list, optional): List of F1 scores to compute the average. Default is None
        :param accuracy_scores: (list, optional): List of accuracy scores to compute the average. Default is None
        :param all_labels: (list, optional): List containing possible labels in the array y. Default is None
        :return: dict: A dictionary containing the following metrics:
                - "mcc": Matthews Correlation Coefficient indicating classification quality.
                - "precision": Precision score computed with 'average=None'.
                - "recall": Recall score computed with 'average=None'.
                - "f1": F1 score computed with 'average=None'.
                - "accuracy": Accuracy score of the classifier.
                - "avg_mcc": Average MCC if `mcc_scores` is provided.
                - "avg_precision": Average precision if `precision_scores` is provided.
                - "avg_recall": Average recall if `recall_scores` is provided.
                - "avg_f1": Average F1 score if `f1_scores` is provided.
                - "avg_accuracy": Average accuracy if `accuracy_scores` is provided.
    """

    metrics = {
        "mcc": matthews_corrcoef(all_y_true, all_y_pred),
        "precision": precision_score(all_y_true, all_y_pred, average=None, zero_division=0),
        "recall": recall_score(all_y_true, all_y_pred, average=None, zero_division=0),
        "f1": f1_score(all_y_true, all_y_pred, average=None, zero_division=0),
        "accuracy": accuracy_score(all_y_true, all_y_pred),
    }

    metrics.setdefault('avg_mcc', np.mean(mcc_scores)) if mcc_scores is not None else None
    metrics.setdefault('avg_precision', np.mean(precision_scores)) if precision_scores is not None else None
    metrics.setdefault('avg_recall', np.mean(recall_scores)) if recall_scores is not None else None
    metrics.setdefault('avg_f1', np.mean(f1_scores)) if f1_scores is not None else None
    metrics.setdefault('avg_accuracy', np.mean(accuracy_scores)) if accuracy_scores is not None else None

    return metrics
