"""
This module supplies code for a lexicase based island. In a lexicase island,
"cases" are realizations of the data set. In this implementation, these cases
are randomly sampled from the entire data set and no considerations are made to
ensure that the entire data set will be sampled/used.

In general, this method was developed to deal with solutions of high modality.

Reference for paper here.

** When using this island, it may be more beneficial to evaluate both the pareto
front and the entire final population when looking for useful solutions as the
realization of the train data is changing and thus the fitness of models is also
changing.
"""
import logging

import numpy as np
import copy

from .evolutionary_optimizer import EvolutionaryOptimizer
from ..util.argument_validation import argument_validation

LOGGER = logging.getLogger(__name__)
subset_txt = "Subset size must be smaller than training data size"


class MultiSourceLexicaseIsland(EvolutionaryOptimizer):
    """
    Island: a basic unit of evolutionary optimization.  It performs the
    generation and evolution of a single population using a generator and
    evolutionary algorithm, respectively.

    Parameters
    ----------
    evolution_algorithm : `EvolutionaryAlgorithm`
        The desired algorithm to use in assessing the population
    generator : `Generator`
        The generator class that returns an instance of a chromosome
    population_size : int
        The desired size of the population
    hall_of_fame : `HallOfFame`
        (optional) The hall of fame object to be used for storing best
        individuals

    Attributes
    ----------
    generational_age : int
        The number of generational steps that have been executed
    population : list of chromosomes
        The population that is evolving
    hall_of_fame: `HallOfFame`
        An object containing the best individuals seen in the optimization
    test_function: `FitnessFunction`
        (optional) A function which can judges the fitness of an individual,
        independent of the `FitnessFunction` used in evolution

    """
    @argument_validation(population_size={">=": 0})
    def __init__(self, evolution_algorithm, generator, population_size,
                 update_frequency=1, subset_ratio=0.1,
                 hall_of_fame=None, test_function=None):
        super().__init__(hall_of_fame, test_function)
        self._generator = generator
        self.population = [generator() for _ in range(population_size)]
        self._ea = evolution_algorithm
        self._population_size = population_size
        self._ea.evaluation.fitness_function._src_num_pts = tuple(
                                [int(subset_ratio*src_size+1) for src_size in \
                        self._ea.evaluation.fitness_function._all_src_num_pts])

        self._full_td = copy.deepcopy(
                            self._ea.evaluation.fitness_function.training_data)
        self._update_frequency = update_frequency
        self._subset_ratio = subset_ratio
        
        self._update_case()
        #assert self._full_td.x.shape[0] >= self._subset_size, subset_txt


    def _do_evolution(self, num_generations):
        for _ in range(num_generations):
            self._execute_generational_step()

    def _execute_generational_step(self):
        
        self.generational_age += 1
        self._update_case()
        self.population = self._ea.generational_step(self.population)
        for indv in self.population:
            indv.genetic_age += 1
    
    def _update_case(self):
        self._update_ff_dataset()
    
    def _update_ff_dataset(self):
        
        breaks = self._ea.evaluation.fitness_function._all_src_num_pts
        idxs = np.arange(sum(breaks))
        breaks = np.append([0], breaks)
        regions = np.cumsum(breaks)

        pos_idxs = [np.arange(regions[i], regions[i+1]) for i in
                                            range(regions.shape[0] - 1)]
        case_idxs = np.concatenate([np.random.choice(pos_idxs[i],
                    size=self._ea.evaluation.fitness_function._src_num_pts[i],
                                                replace=False)\
                                                for i in range(len(pos_idxs))])

        self._ea.evaluation.fitness_function.training_data._x = \
                                        self._full_td.x[case_idxs, :]
        self._ea.evaluation.fitness_function.training_data._y = \
                                        self._full_td.y[case_idxs, :]

    def evaluate_population(self):
        """Manually trigger evaluation of population"""
        self._ea.evaluation(self.population)

    def get_best_individual(self):
        """Finds the individual with the lowest fitness in a population

        Returns
        -------
        best : chromosomes
            The chromosomes with the lowest fitness value
        """
        self.evaluate_population()
        best = self.population[0]
        for indv in self.population:
            if indv.fitness < best.fitness or np.isnan(best.fitness).any():
                best = indv
        return best

    def get_best_fitness(self):
        """ finds the fitness value of the most fit individual

        Returns
        -------
         :
            Fitness of best individual
        """
        return self.get_best_individual().fitness

    def get_fitness_evaluation_count(self):
        """ Gets the number of fitness evaluations performed

        Returns
        -------
        int :
            number of fitness evaluations
        """
        return self._ea.evaluation.eval_count

    def get_ea_diagnostic_info(self):
        """ Gets diagnostic info from the evolutionary algorithm(s)

        Returns
        -------
        EaDiagnosticsSummary :
            summary of evolutionary algorithm diagnostics
        """
        return self._ea.diagnostics

    def _get_potential_hof_members(self):
        return self.population

    def dump_fraction_of_population(self, fraction):
        """Dumps a portion of the population to a list

        Parameters
        ----------
        fraction : float [0.0 - 1.0]
            The fraction of the population to dump

        Returns
        -------
        list of chromosomes :
            A portion of the population
        """
        np.random.shuffle(self.population)
        index = int(round(fraction * len(self.population)))
        dumped_population = self.population[:index]
        self.population = self.population[index:]
        return dumped_population

    def regenerate_population(self):
        """Randomly regenerates the population"""
        self.population = [self._generator()
                           for _ in range(len(self.population))]

    def reset_fitness(self, population=None):
        """
        Mark each individual in the population as needing fitness evaluation

        Parameters
        ----------
        population: list of `Chromosome`
            (Optional) Population to be reset. Default: the island's current
            population
        """
        if population is None:
            population = self.population

        for indv in population:
            indv.fit_set = False
