from collections import namedtuple

import numpy as np

EaDiagnosticsSummary = namedtuple("EaDiagnosticsSummary",
                                  ["beneficial_crossover_rate",
                                   "detrimental_crossover_rate",
                                   "beneficial_mutation_rate",
                                   "detrimental_mutation_rate",
                                   "beneficial_crossover_mutation_rate",
                                   "detrimental_crossover_mutation_rate"])


class EaDiagnostics:
    def __init__(self):
        self._crossover_stats = np.zeros(3)
        self._mutation_stats = np.zeros(3)
        self._cross_mut_stats = np.zeros(3)

    @property
    def summary(self):
        return EaDiagnosticsSummary(
                self._crossover_stats[1] / self._crossover_stats[0],
                self._crossover_stats[2] / self._crossover_stats[0],
                self._mutation_stats[1] / self._mutation_stats[0],
                self._mutation_stats[2] / self._mutation_stats[0],
                self._cross_mut_stats[1] / self._cross_mut_stats[0],
                self._cross_mut_stats[2] / self._cross_mut_stats[0])

    def update(self, population, offspring, offspring_parents,
               offspring_crossover, offspring_mutation):
        beneficial_var = np.zeros(len(offspring), dtype=bool)
        detrimental_var = np.zeros(len(offspring), dtype=bool)
        for i, (child, parent_indices) in \
                enumerate(zip(offspring, offspring_parents)):
            if len(parent_indices) == 0:
                continue
            beneficial_var[i] = \
                all([child.fitness < population[p].fitness
                     for p in parent_indices])
            detrimental_var[i] = \
                all([child.fitness > population[p].fitness
                     for p in parent_indices])

        just_cross = offspring_crossover * ~offspring_mutation
        just_mut = ~offspring_crossover * offspring_mutation
        cross_mut = offspring_crossover * offspring_mutation
        self._crossover_stats += (sum(just_cross),
                                  sum(beneficial_var * just_cross),
                                  sum(detrimental_var * just_cross))
        self._mutation_stats += (sum(just_mut),
                                 sum(beneficial_var * just_mut),
                                 sum(detrimental_var * just_mut))
        self._cross_mut_stats += (sum(cross_mut),
                                  sum(beneficial_var * cross_mut),
                                  sum(detrimental_var * cross_mut))

    def __add__(self, other):
        sum_ = EaDiagnostics()
        sum_._crossover_stats = self._crossover_stats + other._crossover_stats
        sum_._mutation_stats = self._mutation_stats + other._mutation_stats
        sum_._cross_mut_stats = self._cross_mut_stats + other._cross_mut_stats
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)