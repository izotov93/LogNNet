# -*- coding: utf-8 -*-

"""
Created on Thu Aug 17 10:00:00 2024

@author: Yuriy Izotov
@author: Andrei Velichko
@user: izotov93
"""

import numpy as np
import signal
from LogNNet.mlp_evaluation import evaluate_mlp_mod
from multiprocessing import cpu_count, Pool

stop_flag = False


def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
    print("Optimization stopping...")


signal.signal(signal.SIGINT, signal_handler)


def init_position(param_range):
    if isinstance(param_range, tuple):
        return np.random.uniform(param_range[0], param_range[1])
    return param_range


class Particle:
    def __init__(self, param_ranges):
        self.position = [init_position(param_ranges[key]) for key in param_ranges]
        self.dimensions = len(param_ranges)
        self.velocity = np.random.rand(self.dimensions) - 0.5
        self.best_position = self.position.copy()
        self.fitness = float('-inf')
        self.best_fitness = None
        self.best_model = None
        self.input_layers_data = None

    def update_velocity(self, global_best_position):
        inertia = 0.5
        cognitive_component = 2 * np.random.rand(self.dimensions) * (
                np.array(self.best_position, dtype=float) - np.array(self.position, dtype=float))
        social_component = 2 * np.random.rand(self.dimensions) * (
                np.array(global_best_position, dtype=float) - np.array(self.position, dtype=float))
        self.velocity = inertia * self.velocity + cognitive_component + social_component

    def update_position(self, param_ranges):
        self.position = np.array(self.position, dtype=float) + self.velocity
        for i, (key, param_range) in enumerate(param_ranges.items()):
            if isinstance(param_range, tuple):
                self.position[i] = np.clip(self.position[i], param_range[0], param_range[1])


def fitness_function(particle_position: list, X: np.ndarray, y: np.ndarray,
                     num_folds: int, random_state: int, shuffle: bool,
                     selected_metric: str, selected_metric_class: int,
                     target: str, static_features=None) -> (float, object, dict):
    """
    Evaluate the fitness of a particle based on a machine learning model's performance.

        :param particle_position: (list or array-like): A list or array that contains the
                    hyperparameters of the model. The following indices are expected:
            - [0] : Number of rows in matrix W
            - [1] : Zn0 parameter
            - [2] : Cint parameter
            - [3] : Bint parameter
            - [4] : Lint parameter
            - [5] : Number of neurons in the first layer
            - [6] : Number of neurons in the hidden layer
            - [7] : Learning rate
            - [8] : Number of epochs
            - [9] : Prizn parameter
            - [10] : n_f parameter
            - [11] : ngen parameter
        :param X: (np.ndarray): The input features dataset.
        :param y: (np.ndarray): The target values (class labels in classification, real numbers in regression).
        :param num_folds: (int): The number of folds to use for cross-validation.
        :param random_state: (int): Controls the randomness of the model evaluation.
        :param shuffle: (bool): Whether to shuffle the data before splitting into batches.
        :param selected_metric: (str): The metric to be used for evaluating the model performance.
        :param selected_metric_class: (int): Identifies the target class for multi-class classification metrics.
        :param target: (str): The type of  task: 'Regressor' for regression or 'Classifier' for classification.
        :param static_features: (list or None, optional): The static features to be used in the model
        :return: (tuple): A tuple containing the params:
                - res_metric : (float) The value of the selected metric indicating the fitness of
                                the model configuration.
                - model: (object) The trained MLP model.
                - input_layers_data: (dict) Data related to the input layers,
                        including weights W and other normalization parameters.
    """
    params = {
        'first_layer_neurons': int(particle_position[5]),
        'hidden_layer_neurons': int(particle_position[6]),
        'activation': 'relu',
        'learning_rate': float(particle_position[7]),
        'epochs': int(particle_position[8]),
    }

    metrics, model, input_layers_data = evaluate_mlp_mod(X, y, params, num_folds=num_folds,
                                                         num_rows_W=int(particle_position[0]),
                                                         Zn0=particle_position[1],
                                                         Cint=particle_position[2],
                                                         Bint=particle_position[3],
                                                         Lint=particle_position[4],
                                                         shuffle=shuffle,
                                                         random_state=random_state,
                                                         prizn=int(particle_position[9]),
                                                         n_f=int(particle_position[10]),
                                                         ngen=int(particle_position[11]),
                                                         target=target,
                                                         static_features=static_features)

    res_metric = metrics[selected_metric] if selected_metric_class is None \
        else metrics[selected_metric][selected_metric_class]

    return res_metric, model, input_layers_data


def optimize_particle(args) -> Particle:
    """
    Optimizes a single particle's position in the particle swarm based on its fitness score.

        :param args: args.
        :return: (Particle): The updated particle instance with potentially improved fitness,
                    best position, and related model information.
    """

    (particle, global_best_position, param_ranges, X, y, num_folds, random_state, shuffle,
     selected_metric, selected_metric_class, target, static_features) = args

    particle.update_velocity(global_best_position)
    particle.update_position(param_ranges)
    particle.fitness, model, input_layers_data = fitness_function(particle.position, X, y, num_folds,
                                                                  random_state, shuffle, selected_metric,
                                                                  selected_metric_class, target, static_features)

    if selected_metric in ['mse', 'mae', 'rmse']:
        is_better = (particle.best_fitness is None or particle.fitness < particle.best_fitness)
    else:
        is_better = (particle.best_fitness is None or particle.fitness > particle.best_fitness)

    if is_better:
        particle.best_fitness = particle.fitness
        particle.best_position = particle.position.copy()
        particle.best_model = model
        particle.input_layers_data = input_layers_data

    return particle


def PSO(X: np.ndarray, y: np.ndarray, num_folds: int, param_ranges: dict,
        selected_metric: str, selected_metric_class: (int, None), dimensions: int,
        num_particles: int, num_iterations: int, num_threads=cpu_count(),
        random_state=42, shuffle=True, target='Regressor',
        static_features=(list, None)) -> (np.ndarray, float, object, dict):
    """
    Performs Particle Swarm Optimization (PSO) for hyperparameter tuning of LogNNet models.

        :param X: (np.ndarray): The input features of the dataset, where rows represent samples
            and columns represent features.
        :param y: (np.ndarray): The target values corresponding to the input features.
        :param num_folds: (int): The number of folds to use for cross-validation during the
            evaluation of particle fitness.
        :param param_ranges: (dict): A dictionary defining the ranges for the hyperparameters to
            optimize for each model.
        :param selected_metric: (str): A string representing the metric to be used for evaluating the
            fitness of particles.
        :param selected_metric_class: (int, None): For classification tasks, this defines the class to optimize.
        :param dimensions: (int): The number of hyperparameters to optimize, corresponding to the number of
            dimensions in the particle space.
        :param num_particles: (int): The number of particles in the swarm that will explore the hyperparameter space.
        :param num_iterations: (int): The number of iterations for the optimization process.
        :param num_threads: (int, optional): he number of threads to use for parallel execution.
            Default is the number of CPU cores.
        :param random_state: (int, optional): Seed for random number generation to ensure reproducibility.
            Default is 42.
        :param shuffle: (bool, optional):  Whether to shuffle the data before splitting into folds
            for cross-validation. Default is True.
        :param target: (str, optional): The type of  task: 'Regressor' for regression or
            'Classifier' for classification. Default is 'Regressor'.
        :param static_features: (None or list, optional): -
        :return: (tuple): A tuple containing the params:
            - global_best_position: (np.ndarray): The best set of hyperparameters found during optimization.
            - global_best_fitness: (float): The fitness value of the best hyperparameter set.
            - global_best_model: (object) The best-trained model corresponding to the best hyperparameter set.
            - input_layers_data : (dict): Data related to the input layers,
                including weights W and other normalization parameters.
    """

    particles = [Particle(param_ranges) for _ in range(num_particles)]
    global_best_position = np.random.rand(dimensions)
    global_best_fitness = None
    global_best_model, input_layers_data = None, None

    with Pool(num_threads) as pool:

        for iteration in range(num_iterations):
            if stop_flag:
                print("Stopping optimization ...")
                pool.close()
                pool.join()
                break

            args_list = [(particle, global_best_position, param_ranges, X, y, num_folds,
                          random_state, shuffle, selected_metric, selected_metric_class,
                          target, static_features) for particle in particles]

            results = pool.map(optimize_particle, args_list)

            for particle in results:
                if selected_metric in ["mse", "mae", "rmse"]:
                    is_better = (global_best_fitness is None or particle.fitness < global_best_fitness)
                else:
                    is_better = (global_best_fitness is None or particle.fitness > global_best_fitness)

                if is_better:
                    global_best_fitness = particle.fitness
                    global_best_position = particle.position.copy()
                    global_best_model = particle.best_model
                    input_layers_data = particle.input_layers_data

            print(f"Iteration {iteration + 1}/{num_iterations}, Best Fitness in fold: {round(global_best_fitness, 4)}")

        pool.close()
        pool.join()

    return global_best_position, global_best_fitness, global_best_model, input_layers_data


def main():
    pass


if __name__ == "__main__":
    main()
