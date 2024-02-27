import numpy as np

from bingo.symbolic_regression import ComponentGenerator
from bingo.symbolic_regression.mff_explicit_regression import (
    MFFExplicitTrainingData,
    MFFExplicitRegression,
)

from bingo.symbolic_regression.agraph.mff_generator import MFFAGraphGenerator

if __name__ == "__main__":
    X = np.arange(0, 5.0, 0.5).reshape((-1, 2))
    ONES_LIKE = np.ones((X.shape[0], 1))
    Z = np.hstack((1 * ONES_LIKE, 2 * ONES_LIKE, 3 * ONES_LIKE))
    Y = np.ones_like(Z)

    training_data = MFFExplicitTrainingData(x=X, y=Y, z=Z)

    component_generator = ComponentGenerator(training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    generator = MFFAGraphGenerator(
        agraph_size=12,
        component_generator=component_generator,
        use_python=True,
        z_dims=Z.shape[1],
    )

    model = generator()
    #y = model.evaluate_equation_at(training_data.x, training_data.z)
    y, dy_dx = model.evaluate_equation_with_local_opt_gradient_at(X, Z)
    print(str(model))
    import pdb

    pdb.set_trace()
