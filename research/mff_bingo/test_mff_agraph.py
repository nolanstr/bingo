import numpy as np

from bingo.symbolic_regression.agraph.mff_agraph import MFFAGraph


if __name__ == "__main__":
    X = np.arange(0, 2.5, 0.5).reshape((-1, 1))
    Z = np.hstack((1 * np.ones_like(X), 2 * np.ones_like(X), 3 * np.ones_like(X)))

    model = MFFAGraph(equation="X_0 ** 2 + 1.0", z_dims=Z.shape[1])
    model.set_local_optimization_params(np.array([1, 2, 3]))
    y = model.evaluate_equation_at(X, Z)

    general_model = np.hstack(
        [X**2 + model._simplified_constants[i, :] for i in range(Z.shape[1])]
    )
    summation_terms = general_model * Z
    true_y = summation_terms.sum(axis=1).reshape((-1, 1))

    y, dy_dx = model.evaluate_equation_with_x_gradient_at(X, Z)
    import pdb

    pdb.set_trace()
