![PyPI - Downloads](https://img.shields.io/pypi/dm/LogNNet?label=PyPI%20dowloads)
![PyPI](https://img.shields.io/pypi/v/LogNNet?color=informational&label=PyPI%20version)

# LogNNet Neural Network

LogNNet is a neural network [1,2] that can be applied to both classification and regression tasks, alongside other networks such as MLP, RNN, CNN, LSTM, Random Forest, and Gradient Boosting. One of the key advantages of LogNNet is its use of a customizable chaotic reservoir, which is represented as a weight matrix filled with chaotic mappings. In this version, a congruent generator is used, producing a sequence of chaotically changing numbers. The matrix transformation, based on optimized sequences of chaotic weights, enhances the identification of patterns in data. While LogNNet follows the structure of a traditional feedforward neural network, its chaotic transformation of the input space grants it properties similar to reservoir networks

<h4 align="center">
<img src="https://github.com/izotov93/LogNNet/raw/master/images/Struct LogNNet.png" width="800">
<p>Figure 1. Structure of the LogNNet Neural Network in Classification Mode</p>
</h4>

(1) Input dataset, (2) Normalization stage of the input vector Y with a dimension of (dn), (3) Matrix reservoir with (nr) rows, (4) Chaotic series filling the reservoir, (5) Multiplication of the reservoir matrix W by the input vector Y (resulting in product S), (6) Sh - normalized vector S, (7) Output classifier, (8) Training stage, (9) Testing stage.

LogNNet is particularly efficient for resource-constrained devices, and early versions demonstrated high performance on platforms such as Arduino [3] and IoT technologies [4,5]. This efficiency is achieved through the congruent generator, which can generate a large number of weight coefficients from only four parameters. This approach allows for the discovery of optimal weight sequences tailored to specific tasks. Moreover, the weights can be computed "on the fly," significantly reducing memory usage by eliminating the need to store all the weights in advance. Versions of LogNNet designed for low-power devices will be introduced in other GitHub projects.

LogNNet is also used for calculating neural network entropy (NNetEn) [6,7,8,9].

The Python function calls in LogNNet are structured similarly to other neural networks, such as MLP and RNN, utilizing standard stages for training and testing. The optimization of the congruent generator’s parameters is performed using Particle Swarm Optimization (PSO), and training is conducted with k-fold cross-validation. All network parameters are specified in the documentation. Additionally, LogNNet features a multi-threaded search and parameter optimization function, which enhances performance by allowing parallel searches for optimal values, resulting in faster network tuning for specific tasks.
This version is designed for research purposes and can be used for tasks such as classification, regression (including applications in medicine [10,11]), time series prediction, signal processing, recognition of small images, text analysis, anomaly detection, financial data analysis, and more.


## Installation

Installation is done from pypi using the following command

```shell
pip install LogNNet
```
To update installed package to their latest versions, use the ```--upgrade``` option with ```pip install```
```shell
pip install --upgrade LogNNet
```
We recommend using a virtual environment.

## Parameters

1. `num_rows_W` (tuple of int or singular int value, optional), default=(10, 150)

This element represents the number of rows in the reservoir. [PSO]

2. `limit_hidden_layers` (tuple of int, tuple of tuple or singular int value, optional), optional, default=((1, 60), (1, 35))

This element represents the number of neurons in the hidden layers. [PSO]

3. `learning_rate` (tuple of float or singular float value, optional), default=(0.001, 0.1)

The range of learning rate values that the optimizer will use to adjust the model's parameters. [PSO]

4. `n_epochs` (tuple of int or singular int value, optional), default=(5, 550)

The range of the number of epochs (complete passes through the training dataset) for which the model will be trained. [PSO]

5. `n_f` (array-like of int or singular int value, optional), default=-1

This parameter defines the conditions for selecting features in the input vector. It supports three types of input: 
* A list of specific feature indices (e.g., [1, 2, 10] means only features at indices 1, 2, and 10 will be used).
* A range of feature indices as a tuple (e.g., (1, 100) means the PSO method will determine the best features from index 1 to 100).
* A single integer indicating the number of features to be used (e.g., 20 means the PSO method will select the best combination of 20 features). If set to -1, all features from the input vector will be used.

6. `ngen` (tuple of int or singular int value, optional), default=(1, 500)

The range of generations for the optimization algorithm that will be used to find the optimal model parameters. [PSO]

7. `selected_metric` (str value) 

The selected metric for evaluating the model's performance.

For regression (LogNNetRegressor model), input of the following metrics is supported:
* 'r2': R-squared score indicating the proportion of variance explained by the model. (default)
* 'pearson_corr': Pearson correlation coefficient between the true and predicted values.
* 'mse': Mean Squared Error indicating the average squared difference between the true and predicted values.
* 'mae': Mean Absolute Error indicating the average absolute difference between the true and predicted values.
* 'rmse': Root Mean Squared Error indicating the square root of the average squared differences.

For classification (LogNNetClassifier model), input of the following metrics is supported:
* 'mcc': Matthews Correlation Coefficient indicating classification quality.
* 'precision': Precision score.
* 'recall': Recall score.
* 'f1': F1 score.
* 'accuracy': Accuracy score of the classifier. (default)

8. `selected_metric_class` (int or None, optional) Default is None

Select a class metric for training model. Supports input of the following metrics precision, recall and f1 for the LogNNetClassifier class.
**When using LogNNetRegressor model is not used.**

9. `num_folds` (int value, optional), default=1

The number of folds for cross-validation of the model.

10. `num_particles` (int value, optional), default=10

The number of particles in the Particle Swarm Optimization (PSO) method, used for parameter optimization.

11. `num_threads` (int value, optional), default=10

The number of threads to be used during model training for parallel data processing.

12. `num_iterations` (int value, optional), default=10

The number of iterations of the optimization algorithm.

### Additional configuration options.

1. `use_reservoir` (bool value, optional), default=True

The parameter responsible for the use of a chaotic reservoir in calculations. If the value is "False" the LogNNet module operates in the MLP-model mode with the selection of optimal parameters through PSO.

2. `use_debug_mode` (bool value, optional), default=False

The parameter responsible for additional output of service information during the LogNNet library operation.

3. `**`  MLP-model parameters from the scikit-learn library. 

The LogNNetRegressor object additionally supports [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) object parameters.

The LogNNetClassifier object additionally supports [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) object parameters.


## Usage

### LogNNetRegressor 

Multi-layer LogNNet Regression

```python
from LogNNet.neural_network import LogNNetRegressor

...

model = LogNNetRegressor(
    num_rows_W=(10, 150),
    limit_hidden_layers=((1, 60), (1, 35)),
    learning_rate=(0.001, 0.1),
    n_epochs=(5, 550),
    n_f=-1,
    ngen=(1, 500),
    selected_metric='r2',
    num_folds=1,
    num_particles=10,
    num_threads=10,
    num_iterations=10)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

....
```

### LogNNetClassifier ### 

Multi-layer LogNNet Classification

```python
from LogNNet.neural_network import LogNNetClassifier

...

model = LogNNetClassifier(
    num_rows_W=(10, 150),
    limit_hidden_layers=((1, 60), (1, 35)),
    learning_rate=(0.001, 0.1),
    n_epochs=(5, 550),
    n_f=-1,
    ngen=(1, 500),
    selected_metric='accuracy',
    selected_metric_class=None,
    num_folds=1,
    num_particles=10,
    num_threads=10,
    num_iterations=10)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

...
```

### Import/export of trained LogNNet model

The functionality of import/export of the LogNNet model after executing the "fit" function has been implemented. For the LogNNetRegressor and LogNNetClassifier implementations, the functionality is similar.
For the example, we will use the LogNNetRegressor model.

Code for saving model to file:
```python
from LogNNet.neural_network import LogNNetRegressor

params = {} # Some parameters

model = LogNNetRegressor(**params)
model.fit(X_train, y_train)

model.export_model(file_name='LogNNet_model.joblib')
....
```
If the "file_name" parameter is not specified or is None, a model with the name will be created 'timestamp_LogNNet_model.joblib'.

Code to load a trained model from a file:
```python
from LogNNet.neural_network import LogNNetRegressor

model = LogNNetRegressor().import_model(file_name='LogNNet_model.joblib')
....
```

## How to use example files

1. Install the package LogNNet [*](https://github.com/izotov93/LogNNet?tab=readme-ov-file#installation)
2. Download file [example_LogNNet_classification.py](https://github.com/izotov93/LogNNet/blob/master/example_LogNNet_classification.py) or / and [example_LogNNet_regression.py](https://github.com/izotov93/LogNNet/blob/master/example_LogNNet_regression.py)
3. In the directory where the example script is located, create a folder named `database`
4. Place your CSV database file into the `database` folder

The project structure should be as follows:
```
   your_project_folder/
   ├── example_LogNNet_classification.py
   ├── example_LogNNet_regression.py
   └── database/
       └── your_database.csv
```   
5. Configure Parameters 

In both example files there are variables "input_file", "target_column_input_file" and "LogNNet_params".
- input_file - variable containing the name of the *.csv file in the folder "/database/"
- target_column_input_file - variable containing the name of the target column in the input file. 
If the variable is not defined, the first column in the file "input_file" will be selected.
- LogNNet_params - dictionary containing the parameters of the LogNNet neural network [*](https://github.com/izotov93/LogNNet?tab=readme-ov-file#parameters).

6. If changes have been made, you should save the file. Run the example files
7. Once executed, a new directory "LogNNet_results" will be created, which contains a report file named `{unix_time}_metrics_{database_name}.txt` and a file named `{unix_time}_LogNNet_model_{database_name}.joblib` containing the trained model
8. If a regression task was performed, an additional file will be created with the predicted data, file named `{unix_time}_data_{database_name}.txt`

## Authors

This library is developed and maintained by Yuriy Izotov (<izotov93@yandex.ru>) and Andrei Velichko (<velichkogf@gmail.com>).

## License

The source code is licensed under the [MIT License](https://github.com/izotov93/LogNNet/blob/master/LICENSE).

## References
1.	NNetEn Entropy | Encyclopedia MDPI Available online: https://encyclopedia.pub/entry/18173 (accessed on 15 February 2024).
2. 	Velichko, A. Neural Network for Low-Memory IoT Devices and MNIST Image Recognition Using Kernels Based on Logistic Map. Electronics (Basel) 2020, 9, 1432, doi:10.3390/electronics9091432.
3. 	Izotov, Y.A.; Velichko, A.A.; Boriskov, P.P. Method for Fast Classification of MNIST Digits on Arduino UNO Board Using LogNNet and Linear Congruential Generator. J Phys Conf Ser 2021, 2094, 32055, doi:10.1088/1742-6596/2094/3/032055.
4. 	Velichko, А. Artificial Intelligence for Low-Memory IoT Devices. LogNNet . Reservoir Computing. - YouTube Available online: https://www.youtube.com/watch?v=htr08x_RyN8 (accessed on 31 October 2020).
5. 	Heidari, H.; Velichko, A.A. An Improved LogNNet Classifier for IoT Applications. J Phys Conf Ser 2021, 2094, 032015, doi:10.1088/1742-6596/2094/3/032015.
6. 	Conejero, J.A.; Velichko, A.; Garibo-i-Orts, Ò.; Izotov, Y.; Pham, V.-T. Exploring the Entropy-Based Classification of Time Series Using Visibility Graphs from Chaotic Maps. Mathematics 2024, 12, 938, doi:10.3390/math12070938.
7. 	NNetEn Entropy | Encyclopedia MDPI Available online: https://encyclopedia.pub/entry/18173.
8. 	Velichko, A.; Wagner, M.P.; Taravat, A.; Hobbs, B.; Ord, A. NNetEn2D: Two-Dimensional Neural Network Entropy in Remote Sensing Imagery and Geophysical Mapping. Remote Sensing 2022, 14.
9. 	Velichko, A.; Belyaev, M.; Izotov, Y.; Murugappan, M.; Heidari, H. Neural Network Entropy (NNetEn): Entropy-Based EEG Signal and Chaotic Time Series Classification, Python Package for NNetEn Calculation. Algorithms 2023, 16, 255, doi:10.3390/a16050255.
10. Heidari, H.; Velichko, A. An Improved LogNNet Classifier for IoT Application. 2021.
11. Huyut, M.T.; Velichko, A. Diagnosis and Prognosis of COVID-19 Disease Using Routine Blood Values and LogNNet Neural Network. Sensors 2022, 22, 4820, doi:10.3390/s22134820.
 
