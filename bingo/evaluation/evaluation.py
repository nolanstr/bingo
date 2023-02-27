"""The genetic operation of evaluation.

This module defines the a basic form of the evaluation phase of bingo
evolutionary algorithms.
"""
from multiprocessing import Pool
import numpy as np

class Evaluation:
    """Base phase for calculating fitness of a population.

    A base class for the fitness evaluation of populations of genetic
    individuals (list of chromosomes) in bingo.  All individuals in the
    population are evaluated with a fitness function unless their fitness has
    already been set.

    Parameters
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    redundant : bool
        Whether to re-evaluate individuals that have been evaluated previously.
        Default False.

    Attributes
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    eval_count : int
        the number of fitness function evaluations that have occurred
    """
    def __init__(self, fitness_function, redundant=False, multiprocess=False):
        self.fitness_function = fitness_function
        self._redundant = redundant
        self._multiprocess = multiprocess
        self._eval_success = []

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self.fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self.fitness_function.eval_count = value

    def __call__(self, population):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        population : list of chromosomes
                     population for which fitness should be calculated
        """
        try:
            if self.fitness_function._random_sample_subsets is not False:
                self.fitness_function.randomize_subsets()
        except:
            pass

        if self._multiprocess:
            self._multiprocess_eval(population)
        else:
            self._serial_eval(population)

        self.check_eval_success(population)

    def check_eval_success(self, population):
        
        success = 0

        for ind in population:
            if not np.isnan(ind.fitness):
                success += 1

        self._eval_success.append(success/len(population))
        print(f'******************/{self._eval_success[-1]}\******************')

    def _serial_eval(self, population):
        for indv in population:
            if self._redundant or not indv.fit_set:
                indv.fitness = self.fitness_function(indv)
                #if np.isnan(indv.fitness):
                #    import pdb;pdb.set_trace()

    def _multiprocess_eval(self, population):
        num_procs = self._multiprocess if isinstance(self._multiprocess, int) \
                                                                       else None
        with Pool(processes=num_procs) as pool:
            results = []
            for i, indv in enumerate(population):
                if self._redundant or not indv.fit_set:
                    results.append(
                            pool.apply_async(_fitness_job,
                                             (indv, self.fitness_function, i)))
            for res in results:
                indv, i = res.get()
                population[i] = indv

def _fitness_job(individual, fitness_function, population_index):
    individual.fitness = fitness_function(individual)
    return individual, population_index


class StoreEvaluation:
    """Base phase for calculating fitness of a population. This version stores
    multiple fitnesses for an individual that has been evaluated more than once.

    A base class for the fitness evaluation of populations of genetic
    individuals (list of chromosomes) in bingo.  All individuals in the
    population are evaluated with a fitness function unless their fitness has
    already been set.

    Parameters
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    redundant : bool
        Whether to re-evaluate individuals that have been evaluated previously.
        Default False.

    Attributes
    ----------
    fitness_function : FitnessFunction
        The function class that is used to calculate fitnesses of individuals
        in the population.
    eval_count : int
        the number of fitness function evaluations that have occurred
    """
    def __init__(self, fitness_function, redundant=False, multiprocess=False):
        self.fitness_function = fitness_function
        self._redundant = redundant
        self._multiprocess = multiprocess
        self._eval_success = []

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self.fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self.fitness_function.eval_count = value

    def __call__(self, population):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        population : list of chromosomes
                     population for which fitness should be calculated
        """
        try:
            if self.fitness_function._random_sample_subsets is not False:
                self.fitness_function.randomize_subsets()
        except:
            pass

        if self._multiprocess:
            self._multiprocess_eval(population)
        else:
            self._serial_eval(population)

        self.check_eval_success(population)

    def check_eval_success(self, population):
        
        success = 0

        for ind in population:
            if not np.isnan(ind.fitness):
                success += 1

        self._eval_success.append(success/len(population))
        print(f'******************/{self._eval_success[-1]}\******************')

    def _serial_eval(self, population):
        for indv in population:
            if self._redundant or not indv.fit_set:
                indv.fitness = self.fitness_function(indv)

    def _multiprocess_eval(self, population):
        print('Island evalution started!')
        num_procs = self._multiprocess if isinstance(self._multiprocess, int) \
                                                                       else None
        with Pool(processes=num_procs) as pool:
            results = []
            for i, indv in enumerate(population):
                if self._redundant or not indv.fit_set:
                    results.append(
                            pool.apply_async(stored_fitness_job,
                                             (indv, self.fitness_function, i)))
            for res in results:
                indv, i = res.get()
                population[i] = indv
        
        print('evaluation complete')


def stored_fitness_job(individual, fitness_function, population_index):
    individual.fitness = fitness_function(individual)
    individual.fitness_estimates.append(individual.fitness)
    return individual, population_index



