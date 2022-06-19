import numpy as np
import pandas as pd
import os

from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.explicit_regression import ExplicitTrainingData, ExplicitRegression


def default_component_generator(n_xs):
    components = {"+", "-", "/", "*"}

    component_generator = ComponentGenerator(n_xs)
    for component in components:
        component_generator.add_operator(component)
    return component_generator


def get_random_equation(equation_size, component_generator):
    generator = AGraphGenerator(equation_size, component_generator)
    random_eq = generator()

    n_const = random_eq.get_number_local_optimization_params()
    rand_const = np.random.randn(n_const)
    random_eq.set_local_optimization_params(rand_const)

    return random_eq


def get_equation_data(true_equation, n, n_xs):
    X = np.random.randn(n, n_xs)
    y = true_equation.evaluate_equation_at(X)
    return X, y


terminals = {-1, 0, 1}


def get_utilized_idx(cmd_arr, current_idx=None):
    if current_idx is None:
        current_idx = len(cmd_arr) - 1
    utilized_idx = {current_idx}
    current_cmd = cmd_arr[current_idx]
    if current_cmd[0] not in terminals:
        utilized_idx.update(get_utilized_idx(cmd_arr, current_cmd[1]))
        utilized_idx.update(get_utilized_idx(cmd_arr, current_cmd[2]))
    return utilized_idx


def fit_approximator_equation(approximator_equation, X, y):
    training_data = ExplicitTrainingData(X, y)
    fitness = ExplicitRegression(training_data, metric="mse")
    clo = ContinuousLocalOptimization(fitness, algorithm="lm", param_init_bounds=[-10, 10])

    clo(approximator_equation)

    return approximator_equation


def get_approximator_equation(true_equation, X, y):
    cmd_arr = np.copy(true_equation.command_array)

    utilized_idx = get_utilized_idx(cmd_arr, len(cmd_arr) - 1)
    possible_prunes = list(utilized_idx - {len(cmd_arr) - 1})
    if len(possible_prunes) == 0:
        # print(f"true_eq: {true_equation}, is only 1 node")
        return true_equation

    prune_location = np.random.choice(possible_prunes, 1)

    cmd_arr[prune_location] = np.array([1, -1, -1])  # replace w/ constant

    approximator_equation = AGraph()
    approximator_equation.command_array = cmd_arr
    fit_approximator_equation(approximator_equation, X, y)

    return approximator_equation


def get_valid_equation_row(eq_size, component_generator, n_data, n_xs):
    true_eq = None
    X, y = None, None

    while true_eq is None or len(get_utilized_idx(true_eq.command_array)) == 1 or not np.isfinite(y).all():
        true_eq = get_random_equation(eq_size, component_generator)
        X, y = get_equation_data(true_eq, n_data, n_xs)

    approx_eq = get_approximator_equation(true_eq, X, y)

    fitness = ExplicitRegression(ExplicitTrainingData(X, y))

    return [true_eq, fitness(true_eq), approx_eq, fitness(approx_eq), X, y]


if __name__ == '__main__':
    STACK_SIZE = 40

    n_equations = 100
    n_datapoints = 1000
    n_xs = 2

    columns = ["true_eq", "true_eq_fit", "approx_eq", "approx_eq_fit", "true_X", "true_y"]
    df = pd.DataFrame(columns=columns)

    comp_gen = default_component_generator(n_xs)

    for i in range(n_equations):
        df.loc[i] = get_valid_equation_row(STACK_SIZE, comp_gen,
                                           n_datapoints, n_xs)

    print("columns:", df.columns)
    print("len of df:", len(df))
    print("df: ", df)

    directory_name = "data"
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    # IMPORTANT: SAVE TO PICKLE FOR RESUSE, CSV FOR EASIER READING
    file_name = f"{n_datapoints}_points_{n_equations}_eq_{STACK_SIZE}_stack"
    df.to_csv(directory_name + "/" + file_name + ".csv")
    df.to_pickle(directory_name + "/" + file_name + ".pkl")

