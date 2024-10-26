# -*- coding: utf-8 -*-

"""
Created on Thu Sep 14 12:31:00 2024

@author: Izotov Yuriy
@user: izotov93
"""

import time
import os
import csv
import pandas as pd
import numpy as np
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
    'num_rows_W': (10, 150),
    'limit_hidden_layers': ((1, 60), (1, 35)),
    'learning_rate_init': (0.001, 0.1),
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


def print_and_save_results(out_dir: str, data_filename: str, target_column: str, final_metrics: dict,
                           starting_params: dict, LogNNet_model: object, feature_column_names: list,
                           compute_time: str) -> int:
    """
    Prints key results of a model evaluation and saves them to a text file.

        :param out_dir: (str): The directory where the results file will be saved.
        :param data_filename: (str): The name of the data file used in the evaluation.
        :param target_column: (str): Variable containing the name of the target column in the input file.
        :param final_metrics: (dict): A dictionary containing model evaluation metrics.
        :param starting_params: (dict): A dictionary containing the parameters when starting up.
        :param LogNNet_model: (object): The trained LogNNet model.
        :param feature_column_names: (list) A list of names associated with each feature in the dataset.
        :param compute_time: (str): Formatted computation time.

        :return: (int) The Unix time (in seconds) at which the results were saved.
    """

    unix_time = int(time.time())
    filename = os.path.join(out_dir, f"{unix_time}_metrics_[{data_filename}].txt")
    output_str = ""

    prizn_binary = LogNNet_model.input_layer_data['prizn_binary']

    output_str += (f"Data filename: {data_filename}\ntarget_column_input_file: {target_column}\n\n"
                   f"LogNNet version - {LogNNet_model.get_version()}\nComputation time - {compute_time}\n\n"
                   f"Settings for starting:\n  ")
    output_str += '\n  '.join(f"{key}: {value}" for key, value in starting_params.items() if key != 'use_debug_mode')

    output_str += "\n\nMetrics:\n"
    for key, value in final_metrics.items():
        if isinstance(value, np.ndarray):
            if key == "conf_matrix":
                output_str += f"\nConfusion Matrix:\nActual\\Predicted\n\t" + "\t".join(
                    [str(i) for i in range(value.shape[1])]) + "\n"
                for i in range(value.shape[0]):
                    output_str += f"{i}\t" + "\t".join(map(str, value[i])) + "\n"
            else:
                output_str += f"{key}: {', '.join(str(round(v, 6)) for v in value.tolist())}\n"
        else:
            output_str += f"{key}: {value}\n"

    output_str += (f"\n\nNumber of features used: {prizn_binary.count('1')}\n"
                   f"Feature list: {prizn_binary}\nFeature status:\n")
    for i in range(len(prizn_binary)):
        status = "(0)" if prizn_binary[i] == '0' else "(1)"
        output_str += f"Feature {i + 1} {status}\t{feature_column_names[i]}\n"

    output_str += "\nBest parameters LogNNet:\n  "
    output_str += '\n  '.join(f"{key}: {value}" for key, value in LogNNet_model.LogNNet_best_params.items())

    output_str += "\n\nBest MLP params:\n  "
    output_str += '\n  '.join(f"{key}: {value}" for key, value in LogNNet_model.mlp_model.get_params().items())

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output_str)
    print(f"Report saved successfully to {filename}")

    return unix_time


if __name__ == "__main__":
    print("Running LogNNet classification example")

    output_directory = 'LogNNet_results'
    os.makedirs(output_directory, exist_ok=True)
    input_data_file = os.path.join("database", input_file)

    start_time = time.time()

    X, y, feature_names = read_csv_file(input_data_file, target_column=target_column_input_file)
    cutoff = int(len(X) * 0.8)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    model = LogNNetClassifier(**LogNNet_params)
    print(f'LogNNet library version: {model.LogNNet_version}')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mcc": float(round(matthews_corrcoef(y_test, y_pred), 6)),
        "precision": precision_score(y_test, y_pred, average=None, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=None, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=None, zero_division=0),
        "accuracy": float(round(accuracy_score(y_test, y_pred), 6)),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    str_metric_class = '' if LogNNet_params['selected_metric_class'] is None \
        else f" [Class {LogNNet_params['selected_metric_class']}]"
    str_value_metric = metrics[LogNNet_params['selected_metric']] if LogNNet_params['selected_metric_class'] is None \
        else round(metrics[LogNNet_params['selected_metric']][LogNNet_params['selected_metric_class']], 6)

    print(f"Metric '{LogNNet_params['selected_metric']}{str_metric_class}'"
          f" = {str_value_metric} (Test set)")

    print(f'Computation time - {format_execution_time(start_time)}')
    print('Calculation classification finished')

    input_file_name = os.path.basename(input_data_file)
    save_unix_time = print_and_save_results(
        out_dir=output_directory,
        data_filename=input_file_name,
        target_column=target_column_input_file,
        final_metrics=metrics,
        starting_params=LogNNet_params,
        LogNNet_model=model,
        feature_column_names=feature_names,
        compute_time=format_execution_time(start_time)
    )

    # Functionality of writing and reading the model
    model_file_name = os.path.join(output_directory, f'{save_unix_time}_LogNNet_model_[{input_file_name}].joblib')
    print(f'Trained model saved in {model_file_name}')

    model.export_model(file_name=model_file_name)

    print('Loading the trained LogNNet model')
    import_model = LogNNetClassifier().import_model(file_name=model_file_name)
    y_pred_import_model = import_model.predict(X_test)

    metrics = {
        "mcc": float(round(matthews_corrcoef(y_test, y_pred_import_model), 6)),
        "precision": precision_score(y_test, y_pred_import_model, average=None, zero_division=0),
        "recall": recall_score(y_test, y_pred_import_model, average=None, zero_division=0),
        "f1": f1_score(y_test, y_pred_import_model, average=None, zero_division=0),
        "accuracy": float(round(accuracy_score(y_test, y_pred_import_model), 6)),
        "conf_matrix": confusion_matrix(y_test, y_pred_import_model)
    }

    str_metric_class = '' if LogNNet_params['selected_metric_class'] is None \
        else f" [Class {LogNNet_params['selected_metric_class']}]"
    str_value_metric = metrics[LogNNet_params['selected_metric']] if LogNNet_params['selected_metric_class'] is None \
        else round(metrics[LogNNet_params['selected_metric']][LogNNet_params['selected_metric_class']], 6)

    print(f"Metric '{LogNNet_params['selected_metric']}{str_metric_class} = {str_value_metric} "
          f"(From imported model on test set)")
