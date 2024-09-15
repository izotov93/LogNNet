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

## Parameters

1. `input_layer_neurons` (array-like of int or singular int value, optional), default=(10, 70)

This element represents the number of neurons (nr) in the input layer. It can be specified as a range for optimization in the PSO method (e.g., (10, 70)) or as a specific number.

2. `first_layer_neurons` (array-like of int or singular int value, optional), optional, default=(1, 40)

This element represents the number of neurons in the first hidden layer. It can be specified as a range for optimization in the PSO method (e.g., (1, 40)) or as a specific number.

3. `hidden_layer_neuron` (array-like of int or singular int value, optional), default=(1, 15)

The element represents the number of neurons in the second hidden layer. It can be specified as a range for optimization in the PSO method (e.g., (1, 15)) or as a specific number.

4. `learning_rate` (array-like of float or singular float value, optional), default=(0.05, 0.5)

The range of learning rate values that the optimizer will use to adjust the model's parameters.

5. `n_epochs` (array-like of int or singular int value, optional), default=(5, 150)

The range of the number of epochs (complete passes through the training dataset) for which the model will be trained.

6. `n_f` (array-like of int or singular int value, optional), default=-1

This parameter defines the conditions for selecting features in the input vector. It supports three types of input:
* A list of specific feature indices (e.g., [1, 2, 10] means only features at indices 1, 2, and 10 will be used).
* A range of feature indices as a tuple (e.g., (1, 100) means the PSO method will determine the best features from index 1 to 100).
* A single integer indicating the number of features to be used (e.g., 20 means the PSO method will select the best combination of 20 features). If set to -1, all features from the input vector will be used.

7. `ngen` (array-like of int or singular int value, optional), default=(1, 100)

The range of generations for the optimization algorithm that will be used to find the optimal model parameters.

8. `selected_metric` (str value) 

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

9.`selected_metric_class` (int or None, optional) Default is None

Select a class metric for training model. Supports input of the following metrics precision, recall and f1 for the LogNNetClassifier class.
When using LogNNetRegressor model is not used.

10. `num_folds` (int value, optional), default=5

The number of folds for cross-validation of the model.

11. `num_particles` (int value, optional), default=10

The number of particles in the Particle Swarm Optimization (PSO) method, used for parameter optimization.

12. `num_threads` (int value, optional), default=10

The number of threads to be used during model training for parallel data processing.

13. `num_iterations` (int value, optional), default=10

The number of iterations of the optimization algorithm.

14. `random_state` (int value, optional), default=42

A fixed seed for the random number generator, ensuring the reproducibility of results.

15. `shuffle` (bool value, optional), default=True

A parameter indicating that the data will be shuffled before splitting into training and testing sets.

## Usage

### LogNNetRegressor ### 

Multi-layer LogNNet Regression

```python
from LogNNet.neural_network import LogNNetRegressor

...

model = LogNNetRegressor(
                input_layer_neurons=(10, 70),
                first_layer_neurons=(1, 40),
                hidden_layer_neurons=(1, 15),
                learning_rate=(0.05, 0.5),
                n_epochs=(5, 150),
                n_f=-1,
                ngen=(1, 100),
                selected_metric='r2',
                selected_metric_class=None,
                num_folds=5, 
                num_particles=10,
                num_threads=10,
                num_iterations=10, 
                random_state=42,
                shuffle=True)
                
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
                input_layer_neurons=(10, 70),
                first_layer_neurons=(1, 40),
                hidden_layer_neurons=(1, 15),
                learning_rate=(0.05, 0.5),
                n_epochs=(5, 150),
                n_f=-1,
                ngen=(1, 100),
                selected_metric='accuracy',
                selected_metric_class=None,
                num_folds=5, 
                num_particles=10,
                num_threads=10,
                num_iterations=10, 
                random_state=42,
                shuffle=True)
                
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

...
```

## Authors

This library is developed and maintained by Yuriy Izotov (<izotov93@yandex.ru>) and Andrei Velichko (<velichkogf@gmail.com>).

## License

The source code is licensed under the [MIT License](LICENSE).

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
 