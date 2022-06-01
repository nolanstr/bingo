# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import random
import numpy as np
import sympy

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization

from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator

from bingo.symbolic_regression import AGraphGenerator, \
    AGraphCrossover, \
    AGraphMutation, \
    ExplicitRegression, \
    ExplicitTrainingData
POP_SIZE = 100
STACK_SIZE = 10


def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])


def equation_eval(x):
    # like 4.0 * X^2 + X but X is X_0 - X_1
    return 4.0 * (x[:, 0] - x[:, 1]) ** 2 + (x[:, 0] - x[:, 1])


def execute_generational_steps():
    np.random.seed(6)
    random.seed(6)

    x = np.random.randn(100, 2)
    y = equation_eval(x)
    training_data = ExplicitTrainingData(x, y)

    # you'll probably have to tune these probabilities/load statements
    # to you particular problem
    component_generator = ComponentGenerator(x.shape[1],
                                             terminal_probability=0.1,
                                             operator_probability=0.7,
                                             equation_probability=0.2,
                                             num_initial_load_statements=1)
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_equation("X_0 - X_1")

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=False)
    test_agraph = agraph_generator()
    print("Example of agraph generated with equation as component:", test_agraph)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-4)
    if opt_result.success:
        sympy_str = archipelago.get_best_individual().get_formatted_string("sympy")
        print("Found:", sympy_str)
        print("sympy simplified:", sympy.simplify(sympy_str))
    else:
        print("Failed.")

    print(opt_result.ea_diagnostics)


if __name__ == '__main__':
    execute_generational_steps()
