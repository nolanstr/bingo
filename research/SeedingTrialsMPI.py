import json
import logging
import numpy as np
import os
import pandas as pd
import sys
sys.path.append("/u/dlranda2/gpsr/src/bingo")

from sklearn.model_selection import train_test_split

from bingo.evolutionary_optimizers.fitness_predictor_island import \
    FitnessPredictorIsland
from bingo.evolutionary_optimizers.parallel_archipelago import \
    ParallelArchipelago
from bingo.evolutionary_algorithms.deterministic_crowding import \
    DeterministicCrowdingEA
from bingo.stats.hall_of_fame import HallOfFame
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator import \
    ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.local_opt_fitness import \
    LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.explicit_regression import ExplicitTrainingData, \
    ExplicitRegression
from research.GenerateSeeds import SubgraphSeedGenerator

from mpi4py import MPI
communicator = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

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
    clo = LocalOptFitnessFunction(fitness,
                                  ScipyOptimizer(fitness,
                                                 method="lm",
                                                 param_init_bounds=[-10, 10]))
    return Evaluation(clo)


def get_evo_opt(hyperparams, X, y):
    component_generator = setup_component_gen(hyperparams, X.shape[1])
    generator = AGraphGenerator(hyperparams["STACK_SIZE"],
                                component_generator,
                                hyperparams["USE_SIMPLIFICATION"])

    evaluator = get_evaluation(X, y)
    crossover = AGraphCrossover()
    eq_prob = hyperparams.get("EQUATION_MUT_PROB", 0.5)
    other_prob = (1.0 - eq_prob) / 5.0
    mutation = AGraphMutation(component_generator,
        command_probability=other_prob, node_probability=other_prob,
        parameter_probability=other_prob, prune_probability=other_prob,
        fork_probability=other_prob, equation_probability=eq_prob)

    evo_alg = DeterministicCrowdingEA(evaluator, crossover, mutation,
                                      hyperparams.get("CROSSOVER_PROB", 0.4),
                                      hyperparams.get("MUTATION_PROB", 0.1))

    hof = HallOfFame(10)

    island = FitnessPredictorIsland(evo_alg, generator, hyperparams["POPULATION_SIZE"])

    return ParallelArchipelago(island, hall_of_fame=hof)


def run_trial(hyperparams, checkpoint_dir, X, y):
    evo_opt = get_evo_opt(hyperparams, X, y)
    evo_opt.evolve_until_convergence(
        max_generations=hyperparams.get("MAX_GENERATIONS", 1e6),
        fitness_threshold=hyperparams.get("FITNESS_THRESHOLD", 1e-6),
        convergence_check_frequency=hyperparams["CONVERGENCE_CHECK_FREQ"],
        num_checkpoints=hyperparams["NUM_CKPT"],
        checkpoint_base_name=checkpoint_dir + "/checkpoint")


def setup_logging(log_directory, hyperparam_dict, dataset_i, sample_i=0):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

        with open(log_directory + "/hyperparams.json", "w") as f:
            json.dump(hyperparam_dict, f)

    log_file = f"{log_directory}/sample_{sample_i}.log"

    if sample_i == 0:
        logging.basicConfig(filename=log_file,
                            filemode="w+",
                            level=logging.INFO)
    else:
        file_handler = logging.FileHandler(log_file, "w+")
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(file_handler)


def add_seeds(hyperparam_dict, approx_eq):
    seeds = SubgraphSeedGenerator.get_seed_strs(approx_eq.command_array)
    hyperparam_dict["SEEDS"] = list(seeds)


if __name__ == '__main__':
    with open(sys.argv[1], "r") as json_f:
        HYPERPARAMS = json.load(json_f)

    output_dir = HYPERPARAMS["OUTPUT_DIR"]
    if not os.path.exists(output_dir):
        raise RuntimeError("output directory doesn't exist, "
                           "most likely Linux/Windows issue")

    df = pd.read_pickle(HYPERPARAMS["DATA_PATH"])

    method_name = HYPERPARAMS["METHOD_NAME"]
    sample_size = HYPERPARAMS.get("SAMPLE_SIZE", 10)

    if rank == 0:
        print("method:", method_name, end="\n\n")
    for i, row in df.iterrows():
        if rank == 0:
            print("dataset:", i)
            train_test_seed = np.random.randint(1000)
        train_test_seed = communicator.bcast(train_test_seed, root=0)
        HYPERPARAMS["train_test_split_seed"] = train_test_seed

        if HYPERPARAMS["GEN_SEEDS"]:
            add_seeds(HYPERPARAMS, row["approx_eq"])

        if HYPERPARAMS["USE_SEED_FEATURES"]:
            X, y = row["seed_X"], row["true_y"]
        else:
            X, y = row["true_X"], row["true_y"]

        for sample_i in range(sample_size):
            if rank == 0:
                print("\t sample:", sample_i)
            train_X, test_X, train_y, test_y = \
                train_test_split(X, y,
                                 train_size=HYPERPARAMS.get("TRAIN_PERCENT", 0.75),
                                 random_state=HYPERPARAMS["train_test_split_seed"])
            log_dir = output_dir + f"/{method_name}/dataset_{i}/sample_{sample_i}"
            if rank == 0:
                setup_logging(log_dir, HYPERPARAMS, i, sample_i)
            run_trial(HYPERPARAMS, log_dir, train_X, train_y)
