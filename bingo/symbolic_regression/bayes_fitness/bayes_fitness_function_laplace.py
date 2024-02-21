import numpy as np
import math

from copy import deepcopy

from bingo.evaluation.fitness_function import FitnessFunction
from bingo.symbolic_regression.explicit_regression import \
                        SubsetExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness.model_util import \
                        Utilities
#from bingo.symbolic_regression.bayes_fitness.model_priors import \
#                        Priors
from bingo.symbolic_regression.bayes_fitness.model_statistics import \
                        Statistics
#from bingo.symbolic_regression.bayes_fitness.random_sample import \
#                        RandomSample

BASE_SMC_HYPERPARAMS = {'num_particles':150,
                        'mcmc_steps':12,
                        'ess_threshold':0.75}

BASE_MULTISOURCE_INFO = None
#RANDOM_SAMPLE_INFO = None


class BayesFitnessFunction(FitnessFunction, Utilities, Statistics):
    """
    Currently we are only using a uniformly weighted proposal --> This can
    change in the future.
    """
    def __init__(self, continuous_local_opt, smc_hyperparams={}):

        self._cont_local_opt = continuous_local_opt
        Utilities.__init__(self)
        Statistics.__init__(self)
        #RandomSample.__init__(self, continuous_local_opt.training_data, 
        #                                multisource_info, random_sample_info)
        self._set_smc_hyperparams(smc_hyperparams)

        self._norm_phi = 1 / np.sqrt(self._cont_local_opt.training_data.x.shape[0])
        self._eval_count = 0

    def __call__(self, individual, return_nmll_only=True):
        """
        This performs the Laplace approximation of the Fractional Bayes Factor
        calculation as outlined in https://arxiv.org/abs/2304.06333.
        """
        
        try:
            constants, cov, var_ols, ssqe = self.estimate_covariance(individual)
            n = self.training_data.x.shape[0]
            p = len(constants)
            data_independent_terms = ((-n/2)*np.log(2*np.pi)) + ((-n/2)*np.log(var_ols))
            log_likelihood = data_independent_terms + ((-1/(2*var_ols)) * ssqe)
            nmll = ((1-self._norm_phi) * log_likelihood) + ((p/2) * np.log(self._norm_phi))
            return -nmll
        except:
            return np.nan

    def _set_smc_hyperparams(self, smc_hyperparams):
        
        for key in BASE_SMC_HYPERPARAMS.keys():
            if key not in smc_hyperparams.keys():
                smc_hyperparams[key] = BASE_SMC_HYPERPARAMS[key]

        self._num_particles = smc_hyperparams['num_particles']
        self._mcmc_steps = smc_hyperparams['mcmc_steps']
        self._ess_threshold = smc_hyperparams['ess_threshold']
    
    def do_local_opt(self, individual, subset):
        individual._needs_opt = True
        if subset is None:
            _ = self._cont_local_opt(individual)

    def evaluate_model(self, params, individual):
        self._eval_count += 1
        individual.set_local_optimization_params(params.T)
        return individual.evaluate_equation_at(self.x_subset).T

    @property
    def eval_count(self):
        return self._eval_count + self._cont_local_opt.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._eval_count = value - self._cont_local_opt.eval_count

    @property
    def training_data(self):
        return self._cont_local_opt.training_data

    @training_data.setter
    def training_data(self, training_data):
        self._cont_local_opt.training_data = training_data

