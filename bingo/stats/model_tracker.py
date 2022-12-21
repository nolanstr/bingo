"""
The Pareto Front is an extension of hall of fame to construct a list of all the
non-dominated individuals.  An individual dominates another if all of it's keys
are not worse and at least one is better (smaller) than the other's keys.
"""
import numpy as np
from bingo.symbolic_regression.agraph.agraph import AGraph


class TrackModels():
    """Stores simplified agraphs along with every estimate

    Parameters
    ----------
    n : int
        Number of models being stored per 
    """

    def __init__(self, n=5):

        self._n = n
        self._tracked_models = []
        self._tracked_fitness = []
        self.numpy_stats = {'mean':np.nanmean,
                       'median':np.nanmedian,
                       'std':np.nanstd,
                       'min':np.nanmin,
                       'max':np.nanmax}

    def update(self, population):
        """Update the Pareto front based on the given population

        Parameters
        ----------
        population : list of `Chromosome`
            The list of individuals to be considered for induction into the
            Pareto front
        """
        fitness = np.array([ind.fitness for ind in population])
        models = [population[i] for i in fitness.argsort()[:self._n]]
        self._update_n_models(models)

    def _update_n_models(self, n_models):
        
        for model in n_models:
            
            checks = [np.array_equal(model._simplified_command_array, 
                                tracked_model._simplified_command_array) \
                    for tracked_model in self._tracked_models]
            if True in checks:
                tracked_idx = np.where(checks)[0][0]
                self._tracked_fitness[tracked_idx].append(model.fitness)
            else:
                store_model = AGraph()
                store_model.command_array = model._simplified_command_array
                
                self._tracked_models.append(store_model)
                self._tracked_fitness.append([model.fitness])

    def return_best_model(self, fitness_reduction='mean'):

        reduced_fitness = self._reduce_fitness(
                                    fitness_reduction=fitness_reduction)
        return self._tracked_models[np.nanargmin(reduced_fitness)]

    def _reduce_fitness(self, fitness_reduction='mean'):
        stat = self.numpy_stats[fitness_reduction]
        return [stat(fitness) if not np.all(np.isnan(fitness)) else \
                                    np.nan for fitness in self._tracked_fitness]

    def __len__(self):
        return len(self._tracked_models)

    def __getitem__(self, i):
        return self._tracked_models[i]

    def __iter__(self):
        return iter(self._tracked_models)

    def __reversed__(self):
        return reversed(self._tracked_models)

    def __str__(self):
        return '\n'.join(["{}\t{}\t{}".format(key, self._key_func_2(i), i)
                          for key, i in zip(self._keys, self._items)])
