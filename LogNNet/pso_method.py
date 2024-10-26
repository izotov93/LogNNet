# -*- coding: utf-8 -*-

"""
Created on Aug 17 10:00:00 2024
Modified on Oct 18 17:00 2024
Modified on Oct 23 15:00 2024

@author: Yuriy Izotov
@author: Andrei Velichko
@user: izotov93
"""

import numpy as np
import signal
from LogNNet.mlp_evaluation import evaluate_mlp_mod
from multiprocessing import cpu_count, Pool, Manager

stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True


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

        self.is_random_velocity = False

    def update_velocity(self, global_best_position):
        if self.is_random_velocity:
            self.velocity = np.random.rand(self.dimensions) - 0.5
        else:
            inertia = 0.5
            cognitive_component = 2 * np.random.rand(self.dimensions) * (
                    np.array(self.best_position, dtype=float) - np.array(self.position, dtype=float))
            social_component = 2 * np.random.rand(self.dimensions) * (
                    np.array(global_best_position, dtype=float) - np.array(self.position, dtype=float))
            self.velocity = inertia * self.velocity + cognitive_component + social_component
        self.is_random_velocity = False

    def update_position(self, param_ranges):
        self.position = np.array(self.position, dtype=float) + self.velocity
        for i, (key, param_range) in enumerate(param_ranges.items()):
            if isinstance(param_range, tuple):
                self.position[i] = np.clip(self.position[i], param_range[0], param_range[1])


def find_best_parameters(selected_metric: str, best_fitness: float, particle: Particle):
    """
    Updates the best fitness and position based on the provided metric and the particle's performance.

        :param selected_metric: (str): The metric used to evaluate fitness. It supports metrics that indicate
            either minimization (like "mse", "mae", "rmse") or maximization (any other metric).
        :param best_fitness: (float): The current best fitness value found.
        :param particle: (object): The particle object.
        :return: A tuple containing:
            - best_position (array-like): The updated best position.
            - best_fitness (float): The updated the best fitness value.
    """
    best_position, best_model = None, None

    if selected_metric in ["mse", "mae", "rmse"]:
        is_better = (best_fitness is None or particle.fitness < best_fitness)
    else:
        is_better = (best_fitness is None or particle.fitness > best_fitness)

    if is_better:
        best_fitness = particle.fitness
        best_position = particle.position.copy()
        best_model = particle.best_model.copy()

    return best_position, best_fitness, best_model


def optimize_particle(args) -> Particle:
    """
    Optimizes a single particle's position in the particle swarm based on its fitness score.

        :param args: args.
        :return: (Particle): The updated particle instance with potentially improved fitness,
                    best position, and related model information.
    """

    (particle, global_best_position, param_ranges, X, y, num_folds,
     selected_metric, selected_metric_class, target, mlp_params, static_features) = args

    particle.update_velocity(global_best_position)
    particle.update_position(param_ranges)

    params = {
        'hidden_layer_sizes': tuple(int(x) for x in particle.position[10:]),
        'learning_rate_init': float(particle.position[5]),
        'max_iter': int(particle.position[6])
    }

    if mlp_params is not None:
        params.update(mlp_params)

    particle.fitness, mlp_model = evaluate_mlp_mod(X=X, y=y,
                                                   mlp_params=params,
                                                   num_folds=num_folds,
                                                   num_rows_W=int(particle.position[0]),
                                                   Zn0=particle.position[1],
                                                   Cint=particle.position[2],
                                                   Bint=particle.position[3],
                                                   Lint=particle.position[4],
                                                   prizn=int(particle.position[7]),
                                                   n_f=int(particle.position[8]),
                                                   ngen=int(particle.position[9]),
                                                   selected_metric=selected_metric,
                                                   selected_metric_class=selected_metric_class,
                                                   target=target,
                                                   static_features=static_features)

    if selected_metric in ['mse', 'mae', 'rmse']:
        is_better = (particle.best_fitness is None or particle.fitness < particle.best_fitness)
    else:
        is_better = (particle.best_fitness is None or particle.fitness > particle.best_fitness)

    if is_better:
        particle.best_fitness = particle.fitness
        particle.best_position = particle.position.copy()
        particle.best_model = mlp_model

    return particle


def PSO(X: np.ndarray, y: np.ndarray, num_folds: int, param_ranges: dict,
        selected_metric: str, selected_metric_class: (int, None),
        num_particles: int, num_iterations: int, num_threads=cpu_count(),
        target='Regressor', mlp_params=None, static_features=(list, None), **kwargs) -> (np.ndarray, float):
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
        :param num_particles: (int): The number of particles in the swarm that will explore the hyperparameter space.
        :param num_iterations: (int): The number of iterations for the optimization process.
        :param num_threads: (int, optional): he number of threads to use for parallel execution.
            Default is the number of CPU cores.
        :param target: (str, optional): The type of  task: 'Regressor' for regression or
            'Classifier' for classification. Default is 'Regressor'.
        :param mlp_params: (dict): A dictionary containing the mlp parameters.
        :param static_features: (None or list, optional): Parameter containing a list of features of the input
            vector used in the PSO method. If the value None means that all features are used.
        :return: A tuple containing the params:
            - global_best_position: (np.ndarray): The best set of hyperparameters found during optimization.
            - global_best_fitness: (float): The fitness value of the best hyperparameter set.

    """

    use_debug_mode = kwargs.get('use_debug_mode', False)

    rand_particles = int(0.2 * num_particles)
    particles = [Particle(param_ranges) for _ in range(num_particles)]
    global_best_position = np.random.rand(len(param_ranges))
    global_best_fitness = None

    lock = Manager().Lock()

    with Pool(num_threads) as pool:

        for iteration in range(num_iterations):
            if stop_flag:
                print("Stopping optimization ...")
                pool.close()
                pool.join()
                break

            args_list = [(particle, global_best_position, param_ranges, X, y, num_folds,
                          selected_metric, selected_metric_class, target,
                          mlp_params, static_features) for particle in particles]

            results = pool.map(optimize_particle, args_list)

            for i in np.random.choice(len(results), rand_particles, replace=False):
                results[i].is_random_velocity = True

            with lock:
                for particle in results:
                    if selected_metric in ["mse", "mae", "rmse"]:
                        is_better = (global_best_fitness is None or particle.fitness < global_best_fitness)
                    else:
                        is_better = (global_best_fitness is None or particle.fitness > global_best_fitness)

                    if is_better:
                        global_best_fitness = particle.fitness
                        global_best_position = particle.position.copy()
                        global_best_model = particle.best_model

            print(f"Iteration {iteration + 1}/{num_iterations}, Best Fitness: {round(global_best_fitness, 6)}")

            if use_debug_mode:
                print(f'Param best model after interation {global_best_model.get_params()}')

        pool.close()
        pool.join()

    return global_best_position, global_best_fitness


def main():
    pass


if __name__ == "__main__":
    main()
