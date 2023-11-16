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


class ExplicitRegression(VectorGradientMixin, VectorBasedFunction):
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
    def __init__(self, training_data, known_model_form, 
                            metric="mae", relative=False):
        """
        This code will take a "known model form" function that will take 
        the randomly generated agraph and place it into a assumed model 
        form.
        i.e : Known model --> y = X^2 * f(X), gpsr will generate f(X) and 
        the the function known_model_form will place f(X) into X^2*f(X) 
        when fitting parameters and estimating fitness. Furthermore, it 
        will store it as the "known_model_form" in the individuals agraph.
        """
        super().__init__(training_data, metric)
        self._relative = relative
        self.known_model_form = known_model_form
    
    def implement_known_model_form(self, individual):

        constants = individual.constants
        replacements = [f"C_{i}" for i in range(len(constants))]
        individual.set_local_optimization_params(replacements)
        model_string = str(individual)
        known_model_string = self.known_model_form.format(
                                model=model_string) 
        known_model = AGraph(equation=known_model_string)
        known_model.set_local_optimization_params(constants)
        individual.set_local_optimization_params(constants)
        
        return known_model

    def evaluate_fitness_vector(self, individual):
        """ Traditional fitness evaluation for symbolic regression

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
        known_model = self.implement_known_model_form(individual)
        
        f_of_x = known_model.evaluate_equation_at(self.training_data.x)
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
        known_model = self.implement_known_model_form(individual)
        f_of_x, df_dc = \
            known_model.evaluate_equation_with_local_opt_gradient_at(
                    self.training_data.x)
        error = f_of_x - self.training_data.y
        if not self._relative:
            return np.squeeze(error), df_dc
        return np.squeeze(error / self.training_data.y), \
            df_dc / self.training_data.y


class ExplicitTrainingData(TrainingData):
    """
    ExplicitTrainingData: Training data of this type contains an input array of
    data (x)  and an output array of data (y).  Both must be 2 dimensional
    numpy arrays

    Parameters
    ----------
    x : 2D numpy array
        independent variable
    y : 2D numpy array
        dependent variable
    """
    def __init__(self, x, y):
        try:
            if x.ndim == 1:
                # warnings.warn("Explicit training x should be 2 dim array, " +
                #               "reshaping array")
                x = x.reshape([-1, 1])
            if x.ndim > 2:
                raise TypeError('Explicit training x should be 2 dim array')
        except AttributeError:
            pass

        if y.ndim == 1:
            # warnings.warn("Explicit training y should be 2 dim array, " +
            #               "reshaping array")
            y = y.reshape([-1, 1])
        if y.ndim > 2:
            raise TypeError('Explicit training y should be 2 dim array')

        self._x = x
        self._y = y

    @property
    def x(self):
        """independent x data"""
        return self._x

    @property
    def y(self):
        """dependent y data"""
        return self._y

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
        temp = ExplicitTrainingData(self._x[items, :], self._y[items, :])
        return temp

    def __len__(self):
        """ gets the length of the first dimension of the data

        Returns
        -------
        int :
            index-able size
        """
        try:
            return self._x.size(1)
        except TypeError:
            return self._x.shape[0]

class SubsetExplicitTrainingData(TrainingData):
    """
    ExplicitTrainingData: Training data of this type contains an input array of
    data (x)  and an output array of data (y).  Both must be 2 dimensional
    numpy arrays

    Parameters
    ----------
    x : 2D numpy array
        independent variable
    y : 2D numpy array
        dependent variable
    """
    def __init__(self, training_data, src_num_pts, sample_subsets_sizes):
        self._x = training_data.x
        self._y = training_data.y
        
        self.src_num_pts = src_num_pts
        self.sample_subsets_sizes = sample_subsets_sizes

        self.random_sample(src_num_pts, sample_subsets_sizes)

    def get_dataset(self, subset=None):
        
        if subset is None:
            return self._x_subset_data, self._y_subset_data
        else:
            return self.get_subset(subset)

    @property
    def x(self):
        """independent x data"""
        return self._x

    @property
    def y(self):
        """dependent y data"""
        return self._y
    
    def random_sample(self, src_num_pts=None, sample_subsets_sizes=None):
        if src_num_pts is None:
            src_num_pts = self.src_num_pts
        if sample_subsets_sizes is None:
            sample_subsets_sizes = self.sample_subsets_sizes

        self._split_data_into_subsets(src_num_pts)
        
        for i, size in enumerate(sample_subsets_sizes):

            rnd_idxs = np.random.choice(np.arange(src_num_pts[i]), size,
                                                            replace=False)
            self._x_subsets[i] = self._x_subset(i)[rnd_idxs, :]
            self._y_subsets[i] = self._y_subset(i)[rnd_idxs, :]

        self._gather_data()

    def get_subset(self, subset):
        return self._x_subset(subset), self._y_subset(subset)

    def _x_subset(self, subset):
        return self._x_subsets[subset]

    def _y_subset(self, subset):
        return self._y_subsets[subset]
    
    def _split_data_into_subsets(self, src_num_pts):
        
        self._x_subsets = []
        self._y_subsets = []
        idxs = np.append(0, np.cumsum(src_num_pts))

        for subset in range(len(src_num_pts)): 
            self._x_subsets.append(self.x[idxs[subset]:idxs[subset+1], :])
            self._y_subsets.append(self.y[idxs[subset]:idxs[subset+1], :])
    
    def _gather_data(self):

        self._x_subset_data = np.vstack(self._x_subsets)
        self._y_subset_data = np.vstack(self._y_subsets)

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
        temp = ExplicitTrainingData(self._x[items, :], self._y[items, :])
        return temp

    def __len__(self):
        """ gets the length of the first dimension of the data

        Returns
        -------
        int :
            index-able size
        """
        return self._x.shape[0]
