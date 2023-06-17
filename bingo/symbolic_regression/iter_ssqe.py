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

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evaluation.training_data import TrainingData

class MLERegression(VectorBasedFunction):
    """ Implicit Regression via MLE
        Approximation technique derived by Nolan Strauss (M').
    """

    def __init__(self, training_data, required_params=None, iters=5):
        super().__init__(training_data)
        self.training_data = training_data
        self._required_params = required_params
        self._iters = iters
    
    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        data = self.training_data.x.copy()
        dx = np.zeros_like(data)

        for i in range(0, self._iters+1):
            x_pos, x_neg = self.estimate_dx(individual, data)

            ssqe_pos = np.square(np.linalg.norm(x_pos, axis=0)).sum(axis=0)
            ssqe_neg = np.square(np.linalg.norm(x_neg, axis=0)).sum(axis=0)
            ssqe_pos[np.isnan(ssqe_pos)] = np.inf
            ssqe_neg[np.isnan(ssqe_neg)] = np.inf
            
            x_pos, x_neg = x_pos.squeeze().T, x_neg.squeeze().T
            if ssqe_pos[0]<ssqe_neg[0]:
                dx += x_pos
                data += x_pos
            else:
                dx += x_neg
                data += x_neg
        
        ssqe = np.square(np.linalg.norm(dx, axis=1)).sum(axis=0)

        return ssqe
    
    def estimate_dx(self, individual, data):
        vals = self._eval_model(individual, data)
        f, df_dx, df2_d2x = vals
        v = df_dx / (1 + 2*df2_d2x)
        
        a = np.sum(df2_d2x*np.square(v), axis=0)
        b = -np.sum(df_dx*v, axis=0)
        c = f.astype(complex) 
        """
        For complex numbers we can try only considering the real component?
        """
        l_pos = (-b + np.sqrt(np.square(b) - (4*a*c))) / (2*a)
        l_neg = (-b - np.sqrt(np.square(b) - (4*a*c))) / (2*a)
        
        x_pos = (-l_pos*v).real
        x_neg = (-l_neg*v).real
        
        #x_pos = x_pos.real + np.sqrt(np.square(x_neg.imag))
        #x_neg = x_neg.real + np.sqrt(np.square(x_pos.imag))

        return x_pos, x_neg

    def _eval_model(self, ind, data):
        vals = []
        constants = ind.constants
        if len(constants) == 0:
            constants = np.zeros((0,1))
        else:
            constants = np.array(constants).reshape((-1,1))
        ind.set_local_optimization_params(constants)
        ind._simplified_constants = np.array(constants)
        f = ind.evaluate_equation_at(data).reshape((1,-1,1))

        partials = [ind.evaluate_equation_with_x_partial_at(
                            data, [i]*2)[1] for i in \
                            range(data.shape[1])]

        df_dx = np.stack([partial[0] for partial in partials])
        df2_d2x = np.stack([partial[1] for partial in partials])
        return f, np.stack(df_dx), np.stack(df2_d2x)

