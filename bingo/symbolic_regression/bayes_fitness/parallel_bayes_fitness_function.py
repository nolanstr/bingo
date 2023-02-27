import numpy as np
import math

from copy import deepcopy

from bingo.evaluation.fitness_function import FitnessFunction
from bingo.symbolic_regression.explicit_regression import \
                        SubsetExplicitTrainingData
from bingo.symbolic_regression.bayes_fitness.model_util import \
                        Utilities
from bingo.symbolic_regression.bayes_fitness.model_priors import \
                        Priors
from bingo.symbolic_regression.bayes_fitness.model_statistics import \
                        Statistics
from bingo.symbolic_regression.bayes_fitness.random_sample import \
                        RandomSample

from smcpy.parallel_samplers import AdaptiveSampler
from smcpy import MultiSourceNormal
from smcpy.parallel_mcmc.parallel_vector_mcmc import ParallelVectorMCMC
from smcpy.parallel_mcmc.parallel_vector_mcmc_kernel import ParallelVectorMCMCKernel
from mpi4py import MPI


BASE_SMC_HYPERPARAMS = {'num_particles':150,
                        'mcmc_steps':12,
                        'ess_threshold':0.75}

BASE_MULTISOURCE_INFO = None
RANDOM_SAMPLE_INFO = None


class BayesFitnessFunction(FitnessFunction, Utilities, Priors, RandomSample,
        Statistics):
    """
    Currently we are only using a uniformly weighted proposal --> This can
    change in the future.
    """
    def __init__(self, continuous_local_opt, smc_hyperparams={},
                 multisource_info=None,
                 random_sample_info=None,
                 parallel=1,
                 num_multistarts=4,
                 noise_prior='ImproperUniform'):

        self._cont_local_opt = continuous_local_opt
        Priors.__init__(self, noise_prior=noise_prior)
        Utilities.__init__(self)
        Statistics.__init__(self)
        RandomSample.__init__(self, continuous_local_opt.training_data, 
                                        multisource_info, random_sample_info)
        self._set_smc_hyperparams(smc_hyperparams)

        self.parallel = parallel
        self._num_multistarts = num_multistarts
        self._norm_phi = 1 / np.sqrt(self._cont_local_opt.training_data.x.shape[0])
        self._eval_count = 0

    def __call__(self, individual, return_nmll_only=True):
        
        param_names, priors = self._create_priors(individual,
                                                  self._full_multisource_num_pts,
                                                  self._num_particles)

        try:
            proposal = self.generate_proposal_samples(individual,
                                                  self._num_particles*self.parallel,
                                                  param_names)
            params = np.empty((self.parallel, self._num_particles,
                                                    len(param_names)))
            for i, key in enumerate(proposal[0].keys()):
                params[:,:,i] = proposal[0][key].reshape((self.parallel,
                                                         self._num_particles))
            proposal = (params, proposal[1])
            if self._noise_prior == 'InverseGamma':
                priors = self.upate_priors_for_inv_gamma(individual,
                                                         range(len(self._multisource_num_pts)),
                                                         priors)

        except (ValueError, np.linalg.LinAlgError, RuntimeError, Exception) \
                                                                         as e:
            print('error with proposal creation')
            if return_nmll_only:
                return np.nan
            return np.nan, None, None
        log_like_args = [self._multisource_num_pts, 
                            tuple([None]*len(self._multisource_num_pts))]
        log_like_func = MultiSourceNormal
        vector_mcmc = ParallelVectorMCMC(lambda x: self.evaluate_model(x, individual),
                                 self.y_subset.flatten(),
                                 priors,
                                 log_like_args,
                                 log_like_func)

        mcmc_kernel = ParallelVectorMCMCKernel(vector_mcmc, param_order=param_names)
        smc = AdaptiveSampler(mcmc_kernel)
        try:
            step_list, marginal_log_likes = \
                smc.sample(self._num_particles, self._mcmc_steps,
                           param_names,
                           self._ess_threshold,
                           proposal=proposal,
                           parallel=self.parallel,
                           required_phi=self._norm_phi)

        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            import pdb;pdb.set_trace()
            if return_nmll_only:
                self._set_mean_proposal(individual, proposal)
                return np.nan
            return np.nan, None, None

        max_idx = np.unravel_index(step_list[-1].log_likes.argmax(),
                step_list[-1].log_likes.shape)
        maps = step_list[-1].params[max_idx]
        individual.set_local_optimization_params(maps[:-len(self._multisource_num_pts)])

        nmll = -1 * (marginal_log_likes[:, -1] -
                     marginal_log_likes[np.arange(self.parallel), smc.req_phi_index])
        nmll_mean = np.nanmean(nmll)
        print(nmll_mean)
        print(str(individual))
        if return_nmll_only:
            return nmll_mean
        return nmll, step_list, vector_mcmc

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

    def _set_mean_proposal(self, individual, proposal):
        n_params = individual.get_number_local_optimization_params() 
        params = proposal[0]
        new_params = params.reshape((-1, 
                    n_params+len(self._multisource_num_pts))).mean(axis=0)
        individual.set_local_optimization_params(new_params[:n_params])

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

