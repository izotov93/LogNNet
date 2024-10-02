# -*- coding: utf-8 -*-

"""
Created on Thu Sep 14 12:31:00 2024

@author: Izotov Yuriy
@user: izotov93
"""

import time
import os
import json
import csv
import pandas as pd
import numpy as np
from LogNNet import utility
from LogNNet.neural_network import LogNNetClassifier
from sklearn.metrics import (matthews_corrcoef, precision_score, recall_score,
                             f1_score, confusion_matrix, accuracy_score)


"""
- input_file - variable containing the name of the *.csv file in the folder "/database/"
- target_column_input_file - variable containing the name of the target column in the input file. 
If the variable is not defined, the first column in the file "input_file" will be selected.
"""
input_file = 'Veri_Statlog.csv'
target_column_input_file = 'Ischemic'

LogNNet_params = {
    'input_layer_neurons': (10, 150),
    'first_layer_neurons': (1, 60),
    'hidden_layer_neurons': (1, 35),
    'learning_rate_init': (0.001, 0.01),
    'n_epochs': (5, 550),
    'n_f': -1,
    'ngen': (1, 500),
    'selected_metric': 'accuracy',
    'selected_metric_class': None,
    'num_folds': 1,
    'num_particles': 10,
    'num_threads': 10,
    'num_iterations': 10,
}


def format_execution_time(start_time: float) -> str:
    """
    This function computes the elapsed time since the specified start_time
    and formats it into a human-readable string. It only includes hours and
    minutes if they are greater than zero. Seconds are always included.

        :param start_time: (float): The start time in seconds
        :return: (str): A formatted string representing the execution time.
    """
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_parts = []
    if hours > 0:
        time_parts.append(f"{int(hours)} h.")
    if minutes > 0:
        time_parts.append(f"{int(minutes)} m.")

    time_parts.append(f"{int(seconds)} sec.")

    return " ".join(time_parts)


def LogNNet_classification(input_data_file: str, target_column: (str, None),
                           basic_params: dict, output_dir: str) -> None:
    """
    Perform classification calculations using the LogNNet model on input data.

    This function reads data from a specified CSV file, trains a LogNNet classifier
    on the training dataset, evaluates the model on a test dataset, and computes
    various classification performance metrics. The dataset is split into an 80/20
    ratio for training and testing, respectively. Finally, it saves the results
    to the specified output directory.

        :param input_data_file: (str): The name input CSV file containing feature data and target values.
        :param target_column: (str or None): The name of the column in the input data that contains the target values.
        :param basic_params: (dict): A dictionary containing the basic parameters to initialize the
            LogNNet classification.
        :param output_dir: (str): The directory where the output results and predictions will be saved.
        :return: (None): Outputs the performance metrics to a text file in the specified output directory (output_dir).
    """

    start_time = time.time()

    input_file_name = os.path.splitext(os.path.basename(input_data_file))[0]

    X, y, feature_names = read_csv_file(input_data_file, target_column=target_column)

    cutoff = int(len(X) * 0.8)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    model = LogNNetClassifier(**basic_params)
    print(f'LogNNet library version: {model.__version__()}')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mcc": float(round(matthews_corrcoef(y_test, y_pred), 8)),
        "precision": precision_score(y_test, y_pred, average=None, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=None, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=None, zero_division=0),
        "accuracy": float(round(accuracy_score(y_test, y_pred), 8)),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    str_metric_class = '' if basic_params['selected_metric_class'] is None \
        else f" [Class {basic_params['selected_metric_class']}]"
    str_value_metric = metrics[basic_params['selected_metric']] if basic_params['selected_metric_class'] is None \
        else round(metrics[basic_params['selected_metric']][basic_params['selected_metric_class']], 8)

    print(f"Final value of the metric '{basic_params['selected_metric']}{str_metric_class}' "
          f"on the test set = {str_value_metric}")

    print(f"Computation time - {format_execution_time(start_time)}")

    print_and_save_results(
        out_dir=output_dir,
        metrics=metrics,
        best_params=model.LogNNet_best_params,
        data_filename=input_file_name,
        basic_params=model.basic_params,
        mlp_params=model.mlp_model.get_params(),
        feature_names=feature_names)

    print('Calculation classification finished')


def read_csv_file(file_name: str, target_column=None, none_value=' ') -> (np.ndarray, np.ndarray, list):
    """
    Reads a CSV file and preprocesses the data for further analysis.

    This function performs the following steps:
    - Checks if the specified file exists.
    - Determines the separator and encoding of the CSV file.
    - Reads the CSV file into a pandas DataFrame, handling missing values according to the specified `none_value`.
    - Replaces certain string patterns in the data to prepare for conversion to numeric types.
    - Normalizes the values of the target column based on its minimum value.
    - Returns the feature array (X), target array (y), and the list of feature column names.
        :param file_name: (str) The path to the CSV file to be read
        :param target_column: (str or None, optional) Name of the resulting column.
            If None then the first column is selected.
        :param none_value: (str, optional) Value for null value (default is space character)
        :return: (tuple): A tuple containing the params:
            - (np.ndarray): A numpy array of feature values (all columns except the target column).
            - (np.ndarray): A numpy array of target values (the last column).
            - (list): A list of feature column names (excluding the target column)
    """

    if not os.path.isfile(file_name):
        print(f'File {os.path.basename(file_name)} not found')
        exit(0)

    with open(file_name, 'r') as file:
        dialect = csv.Sniffer().sniff(file.readline())
        separator = str(dialect.delimiter)
        encoding = str(file.encoding)

    data = pd.read_csv(file_name, sep=separator, na_values=none_value, encoding=encoding)

    data = data.fillna(0)
    data.replace(to_replace=r',', value='.', inplace=True, regex=True)
    data.replace(to_replace=r'(?<!\d)\.', value='0.', inplace=True, regex=True)
    print(f'Classes column: {data.shape[1]} (Read {data.shape[0]} rows)')

    columns = data.columns.tolist()

    if target_column is not None:
        column_with_keyword = data.filter(like=target_column)
        if column_with_keyword.empty:
            print(f'Column with keyword "{target_column}" not found')
            exit(0)
        col_name = column_with_keyword.columns.tolist()[0]
    else:
        col_name = columns[0]

    print(f'Classes column name: {col_name}')
    columns.remove(col_name)
    columns.append(col_name)
    data = data[columns]

    y = np.array(data.iloc[:, -1]).astype(float)
    X = np.array(data.iloc[:, :-1]).astype(float)

    return X, y, data.columns.tolist()[:-1]


def print_and_save_results(out_dir: str, metrics: dict, best_params: dict,
                           data_filename: str, basic_params: dict,
                           mlp_params: dict, feature_names: list) -> int:
    """
    Prints key results of a model evaluation and saves them to a text file.

        :param out_dir: (str) The directory where the results file will be saved
        :param metrics: (dict) A dictionary containing model evaluation metrics.
        :param best_params: (dict) A dictionary containing the best hyperparameters found during model tuning.
        :param data_filename: (str) The name of the data file used in the evaluation.
        :param basic_params: (dict) A dictionary containing the basic params LogNNet model.
        :param mlp_params: (dict): A dictionary containing the MLP params LogNNet model.
        :param feature_names: (list) A list of names associated with each feature in the dataset.
        :return: (int) The Unix time (in seconds) at which the results were saved.
    """

    unix_time = int(time.time())
    filename = os.path.join(out_dir, f"{unix_time}_metrics_[{data_filename.split('.')[0]}].txt")
    output_str = ""

    input_dim = basic_params['X'].shape[1]

    gray_prizn = utility.decimal_to_gray(best_params['prizn'])
    prizn_binary = utility.binary_representation(gray_prizn, input_dim)
    prizn_binary = utility.modify_binary_string(prizn_binary, best_params['n_f'], best_params['ngen'],
                                                basic_params['static_features'])

    output_str += (f"Data filename: {data_filename}\n\n"
                   f"num_particles: {basic_params['num_particles']}\nnum_threads: {basic_params['num_threads']}\n"
                   f"num_iterations: {basic_params['num_iterations']}\ndimensions: {basic_params['dimensions']}\n"
                   f"selected_metric: {basic_params['selected_metric']}\n"
                   f"selected_metric_class: {basic_params['selected_metric_class']}\n"
                   f"num_folds: {basic_params['num_folds']}\n")

    output_str += "\nMetrics:\n"
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            if key == "conf_matrix":
                output_str += f"{key}:\nActual\\Predicted\n" + "\t".join(
                    [str(i) for i in range(value.shape[1])]) + "\n"
                for i in range(value.shape[0]):
                    output_str += f"{i}\t" + "\t".join(map(str, value[i])) + "\n"
            else:
                output_str += f"{key}:\n{value}\n"
        else:
            output_str += f"{key}: {value}\n"

    output_str += (f"\nFeature list: {prizn_binary}\nNumber of features used: {prizn_binary.count('1')}\n"
                   f"Feature status:\n")

    for i in range(input_dim):
        status = "(0)" if prizn_binary[i] == '0' else "(1)"
        output_str += f"Feature {i + 1} {status}\t{feature_names[i]}\n"

    output_str += "\nBest parameters:\n"
    output_str += json.dumps(best_params, indent=4)

    if mlp_params is not None:
        output_str += "\nMLP params:\n"
        output_str += json.dumps(mlp_params, indent=4)

    print(f"Data saved successfully to {filename}")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output_str)

    return unix_time


if __name__ == "__main__":
    output_directory = 'LogNNet_results'
    os.makedirs(output_directory, exist_ok=True)

    print("Running LogNNet classification example")

    input_file = os.path.join("database", input_file)

    LogNNet_classification(input_data_file=input_file, target_column=target_column_input_file,
                           basic_params=LogNNet_params, output_dir=output_directory)
