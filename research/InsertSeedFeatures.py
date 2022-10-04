import os

import numpy as np
import pandas as pd
from research.GenerateSeeds import SubgraphSeedGenerator

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression import ExplicitTrainingData, ExplicitRegression


def get_best_optimized_model(model: AGraph, clo, repeats=5):
    best_constants = None
    best_fitness = float("inf")

    for repeat in range(repeats):
        model.needs_opt = True
        fitness = clo(model)
        if fitness < best_fitness:
            best_fitness = fitness
            best_constants = model.constants

    model.set_local_optimization_params(best_constants)
    return model


def get_seed_feature_column(seed_str, true_X, true_y):
    training_data = ExplicitTrainingData(true_X, true_y)
    fitness = ExplicitRegression(training_data)
    clo = LocalOptFitnessFunction(fitness, ScipyOptimizer(fitness, method="lm"))

    seed_eq = AGraph(equation=seed_str)
    seed_eq = get_best_optimized_model(seed_eq, clo, repeats=10)

    return seed_eq.evaluate_equation_at(true_X)


def get_seed_X(row):
    approx_eq = row["approx_eq"]
    X, y = row["true_X"], row["true_y"]
    seed_X = np.copy(X)

    seeds = SubgraphSeedGenerator.get_seed_strs(approx_eq.command_array)
    for seed in seeds:
        seed_col = get_seed_feature_column(seed, X, y)
        seed_X = np.hstack((seed_X, seed_col))

    if seed_X.shape[1] - X.shape[1] != len(seeds):
        print("error in getting seed X")

    return seed_X


if __name__ == '__main__':
    data_dir = r"C:/Users/dlranda2/Desktop/GPSR/bingo/research/data/"
    if not os.path.exists(data_dir):
        raise RuntimeError("data directory doesn't exist, "
                           "most likely Linux/Windows issue")

    data_path = data_dir + "1000_points_100_eq_16_stack"

    df = pd.read_pickle(data_path + ".pkl")

    df["seed_X"] = df.apply(get_seed_X, axis=1)

    # IMPORTANT: SAVE TO PICKLE FOR RESUSE, CSV FOR EASIER READING
    df.to_pickle(data_path + "_w_seed.pkl")
    df.to_csv(data_path + "_w_seed.csv")
