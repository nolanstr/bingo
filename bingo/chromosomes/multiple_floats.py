"""Multiple Floats for genetic information

This file contains the several classes that are used for chromosomes
that contains a list of floats for their genetic information.
"""
from .multiple_values import MultipleValueChromosome, \
                            MultipleValueChromosomeGenerator


class MultipleFloatChromosome(MultipleValueChromosome):
    """Multiple float-value individual

    Parameters
    ----------
    values : list of floats
        The genetic information stored in an individual chromosome.
    needs_opt_list : list of ints
        (optional) The indices of the `individual_list` in a
        `chromosomes` object that are subject local optimization.
        This list may be empty
    """
    def __init__(self, values, needs_opt_list=None):
        super().__init__(values)
        if needs_opt_list is None:
            needs_opt_list = []
        self._needs_opt_list = needs_opt_list

    def needs_local_optimization(self):
        """Does the individual need local optimization

        Returns
        -------
        bool
            Individual needs optimization
        """
        if not self._needs_opt_list:
            return False
        return True

    def get_number_local_optimization_params(self):
        """Get number of parameters in local optimization

        Returns
        -------
        int
            number of parameters to be optimized
        """
        return len(self._needs_opt_list)

    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
                 Values to set the parameters
        """
        for param, index in zip(params, self._needs_opt_list):
            self.values[index] = param


class MultipleFloatChromosomeGenerator(MultipleValueChromosomeGenerator):
    """Generation of a population of Multi-Value chromosomes

    Parameters
    ----------
    random_value_function : user defined function
        A function that returns a randomly generated float value.
    values_per_chromosome : int
        The number of values that each chromosome will hold
    needs_opt_list : list of ints
        The indices of the `individual_list` in a  `chromosomes` object
        that are subject local optimization. This list may be empty
    """
    def __init__(self, random_value_function, values_per_chromosome,
                 needs_opt_list=None):
        super().__init__(random_value_function, values_per_chromosome)
        if needs_opt_list is None:
            needs_opt_list = []
        self._check_opt_list_contains_feasible_values(needs_opt_list)
        self._needs_opt_list = self._remove_duplicates(needs_opt_list)

    def __call__(self):
        """Generation of a population of size `population_size`
        of Multi-Value chromosomes with lists that contain
        `values_per_list` values.

        Returns
        -------
        list of chromosomes :
            The chromosomes which their values are generated by
            `random_value_function` with the optimization list
            `needs_opt_list`.
        """
        random_list = self._generate_list(self._values_per_chromosome)
        return MultipleFloatChromosome(random_list, self._needs_opt_list)

    def _check_opt_list_contains_feasible_values(self, list_of_indices):
        if not all(isinstance(x, int) for x in list_of_indices):
            raise ValueError("The list of optimization indices must be \
                              unsigned integers.")
        if list_of_indices and (min(list_of_indices) < 0 or
               max(list_of_indices) >= self._values_per_chromosome):
            raise ValueError("The list of optimization indices must be within \
                              the length of the list of values.")

    @staticmethod
    def _remove_duplicates(list_of_ints):
        set_of_ints = set(list_of_ints)
        return sorted(list(set_of_ints))
