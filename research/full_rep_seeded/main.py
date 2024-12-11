import numpy as np

from bingo.symbolic_regression.explicit_regression import ExplicitTrainingData
from bingo.symbolic_regression.agraph.full_seeded_component_generator import (
    ComponentGenerator,
)
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.full_seeded_generator import AGraphGenerator
from bingo.symbolic_regression.explicit_regression import ExplicitRegression
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.generalized_crowding import \
                                            GeneralizedCrowdingEA
from bingo.selection.deterministic_crowding import \
                                            DeterministicCrowding
from bingo.evolutionary_optimizers.island import Island
from bingo.stats.pareto_front import ParetoFront
from bingo.symbolic_regression.agraph.agraph import AGraph


Y_TAG = "parallelgnddensity"
X_TAGS = [
        "pcs",
        "accumulatedslip",
        "truestrain"]

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == "__main__":
    
    seeded_model = "X_0**2 + 2*X_1"
    x = np.random.normal(size=(10,3))
    y = (x[:,0].reshape((-1,1)) ** 2.0) + (2.0 * x[:,1].reshape((-1,1))) 
    training_data = ExplicitTrainingData(x, y)
    component_generator = ComponentGenerator(
            input_x_dimension=x.shape[1]-2,
            operator_probability=0.35,
            equation_probability=0.2,
            terminal_probability=0.35)
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("sinh")
    component_generator.add_operator("cosh")
    component_generator.add_equation(seeded_model)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(
        component_generator,
    )

    STACK_SIZE = 64

    agraph_generator = AGraphGenerator(
        STACK_SIZE,
        component_generator,
        use_simplification=False,
        use_pytorch=True,
    )

    while True:
        print(str(agraph_generator()))
        import pdb;pdb.set_trace()

