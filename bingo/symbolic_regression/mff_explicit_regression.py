"""Explicit Symbolic Regression

Explicit symbolic regression is the search for a function, f, such that
f(x) = y.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to explicit symbolic regression. Namely, these classes are an
appropriate fitness evaluator and a corresponding training data container.
"""
import numpy as np
import logging
import numpy as np
from .agraph.agraph import AGraph

from ..evaluation.fitness_function import VectorBasedFunction
from ..evaluation.gradient_mixin import VectorGradientMixin
from ..evaluation.training_data import TrainingData

LOGGER = logging.getLogger(__name__)


class MFFExplicitRegression(VectorGradientMixin, VectorBasedFunction):
    """ExplicitRegression

    The traditional fitness evaluation for symbolic regression

    Parameters
    ----------
    training_data : ExplicitTrainingData
        data that is used in fitness evaluation.
    metric : str
        String defining the measure of error to use. Available options are:
        'mean absolute error', 'mean squared error', and
        'root mean squared error'
    relative : bool
        Whether to use relative, pointwise normalization of errors. Default:
        False.
    """

    def __init__(self, training_data, metric="mae", relative=False):
        super().__init__(training_data, metric)
        self._relative = relative

    def evaluate_fitness_vector(self, individual):
        """Traditional fitness evaluation for symbolic regression

        fitness = y - f(x) where x and y are in the training_data (i.e.
        training_data.x and training_data.y) and the function f is defined by
        the input Equation individual.

        Parameters
        ----------
        individual : Equation
            individual whose fitness is evaluated on `training_data`

        Returns
        -------
        float
            the fitness of the input Equation individual
        """
        self.eval_count += 1
        f_of_x = individual.evaluate_equation_at(
            self.training_data.x, self.training_data.z
        )
        error = f_of_x - self.training_data.y
        if not self._relative:
            return np.squeeze(error)
        return np.squeeze(error / self.training_data.y)

    def get_fitness_vector_and_jacobian(self, individual):
        r"""Fitness and jacobian evaluation of individual

        fitness = y - f(x) where x and y are in the training_data (i.e.
        training_data.x and training_data.y) and the function f is defined by
        the input Equation individual.

        jacobian = [[:math:`df_1/dc_1`, :math:`df_1/dc_2`, ...],
                    [:math:`df_2/dc_1`, :math:`df_2/dc_2`, ...],
                    ...]
        where :math:`f_\#` is the fitness function corresponding with the
        #th fitness vector entry and :math:`c_\#` is the corresponding
        constant of the individual

        Parameters
        ----------
        individual : Equation
            individual whose fitness will be evaluated on `training_data`
            and whose constants will be used for evaluating the jacobian

        Returns
        -------
        fitness_vector, jacobian :
            the vectorized fitness of the individual and
            the partial derivatives of each fitness function with respect
            to the individual's constants
        """
        self.eval_count += 1
        f_of_x, df_dc = individual.evaluate_equation_with_local_opt_gradient_at(
            self.training_data.x,
            self.training_data.z
        )
        error = f_of_x - self.training_data.y
        if not self._relative:
            return np.squeeze(error), df_dc
        return np.squeeze(error / self.training_data.y), df_dc / self.training_data.y


class MFFExplicitTrainingData(TrainingData):
    """
    ExplicitTrainingData: Training data of this type contains an input array of
    data (x, z)  and an output array of data (y).  Both must be 2 dimensional
    numpy arrays

    Parameters
    ----------
    x : 2D numpy array
        independent variable
    y : 2D numpy array
        dependent variable
    z : 2D numpy array
        independent variable
    """

    def __init__(self, x, y, z):
        try:
            if x.ndim == 1:
                x = x.reshape([-1, 1])
            if z.ndim == 1:
                z = z.reshape([-1, 1])
            if x.ndim > 2:
                raise TypeError("Explicit training x should be 2 dim array")
        except AttributeError:
            pass

        if y.ndim == 1:
            # warnings.warn("Explicit training y should be 2 dim array, " +
            #               "reshaping array")
            y = y.reshape([-1, 1])
        if y.ndim > 2:
            raise TypeError("Explicit training y should be 2 dim array")

        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        """independent x data"""
        return self._x

    @property
    def y(self):
        """dependent y data"""
        return self._y

    @property
    def z(self):
        """independent z data"""
        return self._z

    def __getitem__(self, items):
        """gets a subset of the `ExplicitTrainingData`

        Parameters
        ----------
        items : list or int
            index (or indices) of the subset

        Returns
        -------
        `ExplicitTrainingData` :
            a Subset
        """
        raise NotImplementedError("__getitem__ for MFF Explicit Training Data")
        temp = ExplicitTrainingData(self._x[items, :], self._y[items, :])
        return temp

    def __len__(self):
        """gets the length of the first dimension of the data

        Returns
        -------
        int :
            index-able size
        """
        try:
            return self._x.size(1)
        except TypeError:
            return self._x.shape[0]
