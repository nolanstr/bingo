"""Implicit Symbolic Regression

Explicit symbolic regression is the search for a function, f, such that
f(x) = constant.  One of the most difficult part of this task is avoiding
trivial solutions like f(x) = 0*x.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to implicit symbolic regression. Namely, these classes are
appropriate fitness evaluators, a corresponding training data container, and
two helper functions.
"""
import logging

import numpy as np

from ..evaluation.fitness_function import VectorBasedFunction
from ..evaluation.training_data import TrainingData

LOGGER = logging.getLogger(__name__)

class MLERegression(VectorBasedFunction):
    """ Implicit Regression via MLE
        Approximation technique derived by Nolan Strauss (M').
    """

    def __init__(self, training_data, iters=10, tol=1e-6):
        super().__init__(training_data)
        self.training_data = training_data
        self._iters = iters
        self._tol = tol

    def evaluate_fitness_vector(self, individual, return_dx=False):

        self.eval_count += 1
        data = self.training_data.x.copy().astype(complex)
        dx = np.zeros_like(data)
         
        for i in range(0, self._iters+1):
            _dx = self.estimate_dx(individual, data)
            dx += _dx
            data += _dx
            if np.abs(_dx).max() < self._tol:
                break
        dx = dx.real
        ssqe = np.square(np.linalg.norm(dx, axis=1)).sum(axis=0)

        if return_dx:
            return ssqe, dx
        return ssqe

    def estimate_dx(self, individual, data):
        
        vals = self._eval_model(individual, data, True)
        f, df_dx = vals
        dx = -f*df_dx/\
                (np.linalg.norm(df_dx, axis=1, ord=2).reshape((-1,1))**2)
        return dx

    def _eval_model(self, ind, data):

        f, df_dx = ind.evaluate_equation_with_x_gradient_at(data)
        return f, df_dx
        
