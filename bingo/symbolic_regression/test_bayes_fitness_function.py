import numpy as np
import math

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import truncnorm

from copy import deepcopy

from bingo.evaluation.fitness_function import FitnessFunction
from bingo.symbolic_regression.explicit_regression import \
                        SubsetExplicitTrainingData
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler
from smcpy import ImproperUniform
from smcpy import MultiSourceNormal
from mpi4py import MPI


BASE_SMC_HYPERPARAMS = {'num_particles':150,
                        'mcmc_steps':12,
                        'ess_threshold':0.75}

BASE_MULTISOURCE_INFO = None
RANDOM_SAMPLE_INFO = None

class TestBayesFitnessFunction(FitnessFunction):
    """
    Currently we are only using a uniformly weighted proposal --> This can
    change in the future.
    """
    def __init__(self, continuous_local_opt, smc_hyperparams={},
                 multisource_info=None,
                 random_sample_info=None,
                 num_multistarts=4):
        
        self._cont_local_opt = continuous_local_opt
        self._set_smc_hyperparams(smc_hyperparams)
        self._set_multisource_info(multisource_info)
        self._set_random_sample_info(random_sample_info)

        self._num_multistarts = num_multistarts
        self._norm_phi = 1 / np.sqrt(continuous_local_opt.training_data.x.shape[0])
        self._eval_count = 0

        self.subset_data = SubsetExplicitTrainingData(deepcopy(
                                                self.training_data),
                                                self._full_multisource_num_pts,
                                                self._multisource_num_pts)

    def __call__(self, individual, return_nmll_only=True):
        
        n_params = individual.get_number_local_optimization_params()
        param_names = self.get_parameter_names(individual) + \
                        [f'std_dev{i}' for i in range(len(self._multisource_num_pts))]
        priors = [ImproperUniform() for _ in range(n_params)] + \
                        [ImproperUniform(0, None)] * len(self._multisource_num_pts)

        try:
            proposal = self.generate_proposal_samples(individual,
                                                  self._num_particles,
                                                  param_names)
        except (ValueError, np.linalg.LinAlgError, RuntimeError, Exception) \
                                                                         as e:
            print('error with proposal creation')
            if return_nmll_only:
                return np.nan
            return np.nan, None, None

        log_like_args = [self._multisource_num_pts, 
                            tuple([None]*len(self._multisource_num_pts))]
        log_like_func = MultiSourceNormal
        vector_mcmc = VectorMCMC(lambda x: self.evaluate_model(x, individual),
                                 self.subset_data._y_subset_data.flatten(),
                                 priors,
                                 log_like_args,
                                 log_like_func)

        mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
        smc = AdaptiveSampler(mcmc_kernel)
        try:
            step_list, marginal_log_likes = \
                smc.sample(self._num_particles, self._mcmc_steps,
                           self._ess_threshold,
                           proposal=proposal,
                           required_phi=self._norm_phi)

        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            if return_nmll_only:
                self._set_mean_proposal(individual, proposal)
                return np.nan
            return np.nan, None, None

        max_idx = np.argmax(step_list[-1].log_likes)
        maps = step_list[-1].params[max_idx]
        individual.set_local_optimization_params(maps[:-len(self._multisource_num_pts)])

        nmll = -1 * (marginal_log_likes[-1] -
                     marginal_log_likes[smc.req_phi_index[0]])
        if return_nmll_only:
            return nmll
        return nmll, step_list, vector_mcmc

    def _set_smc_hyperparams(self, smc_hyperparams):
        
        for key in BASE_SMC_HYPERPARAMS.keys():
            if key not in smc_hyperparams.keys():
                smc_hyperparams[key] = BASE_SMC_HYPERPARAMS[key]

        self._num_particles = smc_hyperparams['num_particles']
        self._mcmc_steps = smc_hyperparams['mcmc_steps']
        self._ess_threshold = smc_hyperparams['ess_threshold']

    def _set_multisource_info(self, multisource_info):
        
        if multisource_info is None:
            multisource_num_pts = \
                    tuple([continuous_local_opt.training_data.x.shape[0]])
        
        self._multisource_num_pts = tuple(multisource_info)
        self._full_multisource_num_pts = tuple(multisource_info)

    def _update_multisource(self, random_sample_info):
        
        if not isinstance(random_sample_info, np.ndarray):
            random_sample_info = np.array(random_sample_info)

        assert(len(random_sample_info)==len(self._multisource_num_pts)), \
                'length of random sample subsets must match multisource num pts'

        if np.all(random_sample_info>=1):
            random_sample_info = np.minimum(random_sample_info, 
                                            np.array(self._full_multisource_num_pts))
            self._multisource_num_pts = tuple(random_sample_info.astype(int).tolist())
        
        elif np.all(random_sample_info<=1):
            self._multisource_num_pts = tuple([math.ceil(subset_size * subset_percent) \
                                      for subset_size, subset_percent in \
                                      zip(self._multisource_num_pts, random_sample_info)])
        else:
            raise ValueError(\
                    'random sample info needs to be all greater than 1 or all less than 1')

        assert(sum(self._multisource_num_pts) < \
               sum(self._full_multisource_num_pts), 
               'random sample subset smaller than full subsets')

    def _set_random_sample_info(self, random_sample_info):
        
        if np.any([isinstance(random_sample_info, float),
                  isinstance(random_sample_info, int)]):
            self._random_sample_subsets = random_sample_info
            self._update_multisource(
                    len(self._multisource_num_pts) * [random_sample_info])

        elif np.any([isinstance(random_sample_info, list), 
                     isinstance(random_sample_info, tuple),
                     isinstance(random_sample_info, np.ndarray)]):
            self._random_sample_subsets = random_sample_info
            self._update_multisource(random_sample_info, uneven_sampling=True)

        else:
            self._random_sample_subsets = 1.0

    @staticmethod
    def get_parameter_names(individual):
        num_params = individual.get_number_local_optimization_params()
        return [f'p{i}' for i in range(num_params)]
    
    def do_local_opt(self, individual, subset):
        individual._needs_opt = True
        #I only have the model being re-optimized on the whole dataset
        if subset is None:
            _ = self._cont_local_opt(individual)
            #print('COMMENTED OUT RE OPT OF MODEL FOR SR1 TEST')

    def generate_proposal_samples(self, individual, num_samples, param_names):

        pdf = np.ones((num_samples, 1))
        samples = np.ones((num_samples, len(param_names)))
        cov_estimates = []
        
        n_params = individual.get_number_local_optimization_params()
        if n_params > 0:

            param_dists, cov_estimates = self._get_dists(individual)

            pdf, samples[:,:n_params] = self._get_samples_and_pdf(param_dists,
                                                                    num_samples)

        for subset, len_data in enumerate(self._multisource_num_pts):
            param_dist, cov_estimates = self._get_dists(individual, 
                                                            subset)
            noise_pdf, noise_samples = \
                                self._get_added_noise_samples(cov_estimates,
                                                                len_data, 
                                                                num_samples)
            samples[:, n_params+subset] = noise_samples.flatten()
            pdf *= noise_pdf

        pdf = np.ones_like(pdf)

        samples = dict(zip(param_names, samples.T))

        return samples, pdf
    
    @staticmethod
    def _get_samples_and_pdf(distributions, num_samples):
        sub_samples = num_samples // len(distributions)
        samples = np.vstack([proposal.rvs(sub_samples).reshape(sub_samples, -1)
                             for proposal in distributions])
        if samples.shape[0] != num_samples:
            missed_samples = num_samples - samples.shape[0]
            new_samples = np.random.choice(distributions).rvs(
                                missed_samples).reshape((missed_samples, -1))
            samples = np.vstack([samples, new_samples])
        pdf = np.zeros((samples.shape[0], 1))
        for dist in distributions:
            pdf += dist.pdf(samples).reshape(-1, 1)
        pdf /= len(distributions)
        return pdf, samples

    def _get_added_noise_samples(self, cov_estimates, len_data, num_samples):
        noise_dists1 = [invgamma((0.01 + len_data) / 2,
                        scale=(0.01 * var_ols + ssqe) / 2)
                       for _, _, var_ols, ssqe in cov_estimates]
        noise_dists = [truncnorm(0, np.inf, loc=0, scale=var_ols)\
                        for _, _, var_ols, ssqe in cov_estimates]

        noise_pdf1, noise_samples1 = self._get_samples_and_pdf(noise_dists1,
                                                             num_samples)

        noise_pdf, noise_samples = self._get_samples_and_pdf(noise_dists,
                                                             num_samples)
        return noise_pdf, noise_samples

    def randomize_subsets(self):
        self.subset_data.random_sample(self._full_multisource_num_pts, 
                                                self._multisource_num_pts)

    def estimate_covariance(self, individual, subset=None):
        
        x, y = self.subset_data.get_dataset(subset=subset)

        self.do_local_opt(individual, subset)

        num_params = individual.get_number_local_optimization_params()
        f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)
        
        ssqe = np.sum((f - y) ** 2)
        #included a max here to avoid division by zero + negative multiplication
        # of the cov
        var_ols = ssqe / max(1, len(f) - num_params)
        try:
            cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))
        except:
            cov = var_ols * np.linalg.pinv(f_deriv.T.dot(f_deriv))

        return individual.constants, cov, var_ols, ssqe#, noise_std_dev

    def _get_dists(self, individual, subset=None):

        param_dists = []
        cov_estimates = []
        trigger = False

        for _ in range(self._num_multistarts*4):

            mean, cov, var_ols, ssqe = \
                            self.estimate_covariance(individual, subset)
            try:
                if cov.shape[0] > 0 and not np.any(np.isnan(cov)):
                    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                    if min_eig < 0:
                        cov += 1e-12 * np.eye(*cov.shape)
                if individual.get_number_local_optimization_params() == 0:
                    dists = None
                else:
                    dists = mvn(mean, cov, allow_singular=True)
            except ValueError as e:
                dists = mvn(np.zeros(len(mean)), 
                            np.eye(len(mean)), allow_singular=True)

            cov_estimates.append((mean, cov, var_ols, ssqe))
            param_dists.append(dists)
            if len(param_dists) == self._num_multistarts:
                break
        if not param_dists:
            raise RuntimeError('Could not generate any valid proposal '
                               'distributions')
            return None, None
        return param_dists, cov_estimates


    def _set_mean_proposal(self, individual, proposal):
        params = np.empty(0)
        for key in proposal[0].keys():
            params = np.append(params, proposal[0][key].mean())
        individual.set_local_optimization_params(params[:-1])

    def evaluate_model(self, params, individual):
        self._eval_count += 1
        individual.set_local_optimization_params(params.T)
        return individual.evaluate_equation_at(
                                    self.subset_data._x_subset_data).T

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
