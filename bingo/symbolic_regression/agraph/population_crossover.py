"""Definition of crossover between two acyclic graph individuals

This module contains the implementation of single point crossover between
acyclic graph individuals.
"""
import numpy as np

from ...chromosomes.crossover import Crossover
from .agraph import AGraph


class AGraphCrossover(Crossover):
    """Crossover between acyclic graph individuals"""

    def __call__(self, population):
        """Single point crossover.

        Parameters
        ----------
        parent_1 : `AGraph`
            The first parent individual
        parent_2 : `AGraph`
            The second parent individual

        Returns
        -------
        tuple(`AGraph`, `AGraph`) :
            The two children from the crossover.
        """

        child1, child2 = self.sample_model_from_population(population)

        return child1, child2

    def sample_model_from_population(self, population):
        fitness_vals = 1.0 / self.get_population_fitness(population)
        weights = fitness_vals / fitness_vals.sum()
        # 2 children always
        children = []
        for _ in range(2):
            sampled_command_array = np.zeros_like(population[0].command_array)
            sample_idxs = np.random.choice(
                np.arange(len(population)),
                size=sampled_command_array.shape[0],
                p=weights,
                replace=True,
            )
            for i, sample_id in enumerate(sample_idxs):
                sampled_command_array[i] = population[sample_id].command_array[i, :]

            sampled_ind = AGraph()
            sampled_ind.command_array = sampled_command_array
            children.append(sampled_ind)

        return children[0], children[1]

    def get_population_fitness(self, population):
        fitness_vals = np.array([ind.fitness for ind in population])
        fit_min_abs = np.nanmin(abs(fitness_vals))

        if np.any(fitness_vals < 0.0):
            fitness_vals += fit_min_abs - np.nanmin(fitness_vals)
        fitness_vals[np.where(np.isnan(fitness_vals))[0]] = np.inf

        return fitness_vals
