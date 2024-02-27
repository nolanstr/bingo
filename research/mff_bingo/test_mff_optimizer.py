import numpy as np

from bingo.symbolic_regression.mff_explicit_regression import (
    MFFExplicitTrainingData,
    MFFExplicitRegression,
)
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization

from bingo.symbolic_regression.agraph.mff_agraph import MFFAGraph

if __name__ == "__main__":
    X = np.arange(0, 100, 0.5).reshape((-1, 1))
    Z = np.random.normal(loc=0, scale=1, size=(X.shape[0], 3))

    CONSTANTS = np.random.normal(loc=0, scale=1, size=Z.shape[1]) 
    
    general_model = np.hstack(
        [X**2 + CONSTANTS[i] for i in range(Z.shape[1])]
    )
    summation_terms = general_model * Z
    Y = summation_terms.sum(axis=1).reshape((-1, 1))
    

    training_data = MFFExplicitTrainingData(x=X, y=Y, z=Z)
    fitness = MFFExplicitRegression(training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, 
                                        algorithm='lm')

    model = MFFAGraph(equation="X_0 ** 2 + 1.0", z_dims=Z.shape[1])
    y = model.evaluate_equation_at(training_data.x, training_data.z)
    print(str(model))
    
    local_opt_fitness(model)

    import pdb

    pdb.set_trace()
