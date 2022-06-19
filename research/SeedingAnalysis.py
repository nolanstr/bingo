import json
import logging
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from bingo.evolutionary_optimizers.fitness_predictor_island import \
    FitnessPredictorIsland
from bingo.evolutionary_algorithms.deterministic_crowding import \
    DeterministicCrowdingEA
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import \
    ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.continuous_local_opt import \
    ContinuousLocalOptimization
from bingo.symbolic_regression.explicit_regression import ExplicitTrainingData, \
    ExplicitRegression


def setup_component_gen(hyperparams, n_xs):
    equation_prob = hyperparams.get("EQUATION_PROB", 0.0)
    seeds = hyperparams.get("SEEDS", [])
    if len(seeds) == 0:
        equation_prob = 0
    component_gen = ComponentGenerator(n_xs,
                                       terminal_probability=hyperparams.get(
                                           "TERMINAL_PROB", 0.1),
                                       operator_probability=hyperparams.get(
                                           "OPERATOR_PROB", 0.9),
                                       equation_probability=equation_prob)
    for operator in hyperparams["OPERATORS"]:
        component_gen.add_operator(operator)
    for seed_eq in seeds:
        component_gen.add_equation(seed_eq)
    return component_gen


def get_evaluation(X, y):
    fitness = ExplicitRegression(ExplicitTrainingData(X, y), metric="mse")
    clo = ContinuousLocalOptimization(fitness, algorithm="BFGS",
                                      param_init_bounds=[-10, 10])
    return Evaluation(clo)


def get_evo_opt(hyperparams, X, y):
    component_generator = setup_component_gen(hyperparams, X.shape[1])
    generator = AGraphGenerator(hyperparams["STACK_SIZE"],
                                component_generator,
                                hyperparams["USE_SIMPLIFICATION"])

    evaluator = get_evaluation(X, y)
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    evo_alg = DeterministicCrowdingEA(evaluator, crossover, mutation,
                                      hyperparams.get("CROSSOVER_PROB", 0.4),
                                      hyperparams.get("MUTATION_PROB", 0.1))

    return FitnessPredictorIsland(evo_alg, generator,
                                  hyperparams["POPULATION_SIZE"])


def run_trial(hyperparams, log_dir, X, y):
    evo_opt = get_evo_opt(hyperparams, X, y)
    evo_opt.evolve_until_convergence(
        max_generations=hyperparams.get("MAX_GENERATIONS", 1e6),
        fitness_threshold=hyperparams.get("FITNESS_THRESHOLD", 1e-6),
        convergence_check_frequency=10,
        checkpoint_base_name=log_dir + "/checkpoint")


def setup_logging(method_name, hyperparam_dict, trial_n):
    directory_name = f"output/{method_name}/trial_{trial_n}"

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    with open(directory_name + "/hyperparams.json", "w") as f:
        json.dump(hyperparam_dict, f)

    logging.basicConfig(filename=f"{directory_name}/trial_{trial_n}.log",
                        filemode="w+", level=logging.INFO)

    return directory_name


if __name__ == '__main__':
    HYPERPARAMS = {
        "STACK_SIZE": 64,
        "POPULATION_SIZE": 100,

        "MAX_GENERATIONS": int(1e8),
        "FITNESS_THRESHOLD": 1e-6,

        "USE_SIMPLIFICATION": True,
        "CROSSOVER_PROB": 0.4,
        "MUTATION_PROB": 0.1,

        "TERMINAL_PROB": 0.1,
        "OPERATOR_PROB": 0.7,
        "EQUATION_PROB": 0.2,
        "OPERATORS": ["+", "-", "*", "/", "sin", "cos"],
        "SEEDS": [],

        "TRAIN_PERCENT": 0.75
    }
    trial_number = 0

    log_dir = setup_logging("no_seeding/", HYPERPARAMS,
                            trial_number)

    df = pd.read_pickle("data/1000_points_100_eq_40_stack.pkl")
    row = df.iloc[0]
    X, y = row["true_X"], row["true_y"]
    train_X, test_X, train_y, test_y = \
        train_test_split(X, y, train_size=HYPERPARAMS.get("TRAIN_PERCENT", 0.75))
    run_trial(HYPERPARAMS, log_dir, train_X, train_y)
