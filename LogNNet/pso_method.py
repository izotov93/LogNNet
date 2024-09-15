# -*- coding: utf-8 -*-

import numpy as np
import signal
import os
import time
from LogNNet.mlp_evaluation import evaluate_mlp_mod
from multiprocessing import cpu_count, Pool, current_process

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
        self.best_fitness = float('-inf')
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
                                                         target=target, static_features=static_features)

    res_metric = metrics[selected_metric] if selected_metric_class is None \
        else metrics[selected_metric][selected_metric_class]

    return res_metric, model, input_layers_data


def optimize_particle_batch(particles_batch):
    results = []
    for particle_args in particles_batch:
        (particle, global_best_position, param_ranges, X, y,
         num_folds, random_state, shuffle, selected_metric,
         selected_metric_class, target, static_features) = particle_args

        particle.update_velocity(global_best_position)
        particle.update_position(param_ranges)
        particle.fitness, model, input_layers_data = (
            fitness_function(particle.position, X, y, num_folds, random_state, shuffle, selected_metric,
                             selected_metric_class, target, static_features))

        if particle.fitness > particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = particle.position.copy()
            particle.best_model = model
            particle.input_layers_data = input_layers_data

        results.append(particle)

    return results


def split_into_batches(data, num_batches):
    avg = len(data) // num_batches
    remainder = len(data) % num_batches
    batches = []
    start = 0
    for i in range(num_batches):
        batch_size = avg + (1 if i < remainder else 0)
        batches.append(data[start:start + batch_size])
        start += batch_size
    return batches


def PSO(X: np.ndarray, y: np.ndarray, num_folds: int, param_ranges: dict,
        selected_metric: str, selected_metric_class: (int, None), dimensions: int,
        num_particles: int, num_iterations: int, num_threads=cpu_count(),
        random_state=42, shuffle=True, target='Regressor',
        static_features=(list, None)) -> (np.ndarray, float, object, dict):

    particles = [Particle(param_ranges) for _ in range(num_particles)]
    global_best_position = np.random.rand(dimensions)
    global_best_fitness = float('-inf')
    global_best_model, input_layers_data = None, None

    particles_batch = split_into_batches(
        [(particle, global_best_position, param_ranges, X, y, num_folds, random_state, shuffle,
          selected_metric, selected_metric_class, target, static_features) for particle in particles],
        num_threads)

    with Pool(num_threads) as pool:
        for iteration in range(num_iterations):
            if stop_flag:
                print("Stopping optimization ...")
                break

            try:
                results = pool.map(optimize_particle_batch, particles_batch)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                print("Optimization interrupted.")
                break

            for batch in results:
                for particle in batch:
                    if particle.fitness > global_best_fitness:
                        global_best_fitness = particle.fitness
                        global_best_position = particle.position.copy()
                        global_best_model = particle.best_model
                        input_layers_data = particle.input_layers_data

            print(f"Iteration {iteration + 1}/{num_iterations}, Best Fitness: {round(global_best_fitness, 4)}")

        pool.close()
        pool.join()

    print(f"Global best position: {[round(float(i), 3) for i in global_best_position]}, "
          f"Global best result: {round(global_best_fitness, 4)}")

    return global_best_position, global_best_fitness, global_best_model, input_layers_data


def main():
    pass


if __name__ == "__main__":
    main()
