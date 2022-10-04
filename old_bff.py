import numpy as np
import math

from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma
from scipy.stats import uniform
from scipy.stats import norm

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

class BayesFitnessFunction(FitnessFunction):

    def __init__(self, continuous_local_opt, src_num_pts=None,
                 num_particles=150, mcmc_steps=12,
                 ess_threshold=0.75,
                 return_nmll_only=True, num_multistarts=1,
                 uniformly_weighted_proposal=True, 
                 random_sample_subsets=1.0):

        self._num_particles = num_particles
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self.num_multistarts = num_multistarts
        self._std = None
        self._return_nmll_only = return_nmll_only
        self._num_multistarts = num_multistarts
        self._uniformly_weighted_proposal = uniformly_weighted_proposal
        self._random_sample_subsets = random_sample_subsets
        
        if src_num_pts is None:
            src_num_pts = \
                    tuple([continuous_local_opt.training_data.x.shape[0]])
        

        self._all_src_num_pts = tuple(src_num_pts)
        self._src_num_pts = tuple(src_num_pts)
        self._full_src_num_pts = self._src_num_pts

        num_observations = len(continuous_local_opt.training_data.x)
        self._norm_phi = 1 / np.sqrt(num_observations)

        self._cont_local_opt = continuous_local_opt
        self._eval_count = 0
        if random_sample_subsets != 1.0:
            self._src_num_pts = tuple(
                                [math.ceil(src_pts*random_sample_subsets)+1 \
                                              for src_pts in self._src_num_pts])
        self.subset_data = SubsetExplicitTrainingData(deepcopy(
                                                self.training_data),
                                                self._full_src_num_pts,
                                                self._src_num_pts)
        self._create_additive_noise_clos(len(src_num_pts))

    def __call__(self, individual):
        
        n_params = individual.get_number_local_optimization_params()
        param_names = self.get_parameter_names(individual) + \
                        [f'std_dev{i}' for i in range(len(self._src_num_pts))]
        priors = [ImproperUniform() for _ in range(n_params)] + \
                        [ImproperUniform(0, None)] * len(self._src_num_pts)

        try:
            proposal = self.generate_proposal_samples(individual,
                                                  self._num_particles,
                                                  param_names)
        except (ValueError, np.linalg.LinAlgError, RuntimeError, Exception) as e:
            print('error with proposal creation')
            if self._return_nmll_only:
                return np.nan
            return np.nan, None, None

        log_like_args = [self._src_num_pts, 
                            tuple([None]*len(self._src_num_pts))]
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
            if self._return_nmll_only:
                self._set_mean_proposal(individual, proposal)
                return np.nan
            return np.nan, None, None

        max_idx = np.argmax(step_list[-1].log_likes)
        maps = step_list[-1].params[max_idx]
        individual.set_local_optimization_params(maps[:-len(self._src_num_pts)])

        nmll = -1 * (marginal_log_likes[-1] -
                     marginal_log_likes[smc.req_phi_index[0]])
        if self._return_nmll_only:
            return nmll
        return nmll, step_list, vector_mcmc

    @staticmethod
    def get_parameter_names(individual):
        num_params = individual.get_number_local_optimization_params()
        return [f'p{i}' for i in range(num_params)]
    
    def _create_additive_noise_clos(self, num_subsets):
        self.randomize_subsets()
        self._additive_noise_clos = []
        for subset in range(num_subsets):
            x, y = self._get_subset_data(subset)
            clo = deepcopy(self._cont_local_opt)
            clo.training_data._x = x
            clo.training_data._y = y
            self._additive_noise_clos.append(clo)
        return None

    def do_local_opt(self, individual, subset):
        #individual._needs_opt = True
        #if subset is False:
        #    _ = self._cont_local_opt(individual)
        #else:
        #    _ = self._additive_noise_clos[subset](individual)
        return None  
        #return individual

    def generate_proposal_samples(self, individual, num_samples, param_names):

        pdf = np.ones((num_samples, 1))
        samples = np.ones((num_samples, len(param_names)))
        cov_estimates = []
        
        n_params = individual.get_number_local_optimization_params()
        if n_params > 0:
            param_dists, cov_estimates = self._get_dists(individual, 
                                                           self.num_multistarts)
            pdf, samples[:,:n_params] = self._get_samples_and_pdf(param_dists,
                                                                    num_samples)

        for subset, len_data in enumerate(self._src_num_pts):
            try:
                param_dist, cov_estimates = self._get_dists(individual, 
                                                            self.num_multistarts,
                                                                subset)
                noise_pdf, noise_samples = \
                                    self._get_added_noise_samples(cov_estimates,
                                                                    len_data, 
                                                                    num_samples)
                samples[:, n_params+subset] = noise_samples.flatten()
                pdf *= noise_pdf
            except:
                import pdb;pdb.set_trace()

        if self._uniformly_weighted_proposal:
            pdf = np.ones_like(pdf)

        samples = dict(zip(param_names, samples.T))
        self._check_for_negative_added_noise(samples)

        return samples, pdf
    
    def _get_added_noise_samples(self, cov_estimates, len_data, num_samples):
        noise_dists = [invgamma((0.01 + len_data) / 2,
                        scale=(0.01 * var_ols + ssqe) / 2)
                       for _, _, var_ols, ssqe in cov_estimates]

        #noise_std_devs = [noise_std_dev for \
        #            mean, cov, var_ols, ssqe, noise_std_dev in cov_estimates]
        #noise_dists = [norm(0, var) \
        #            for var  in noise_std_devs]
        noise_pdf, noise_samples = self._get_samples_and_pdf(noise_dists,
                                                             num_samples)
        return noise_pdf, noise_samples

    def randomize_subsets(self):
        self.subset_data.random_sample(self._full_src_num_pts, 
                                                self._src_num_pts)
    def _get_subset_data(self, subset):
        return self.subset_data.get_subset(subset)

    def estimate_covariance(self, individual, subset=False):
        
        individual = deepcopy(individual)
        if subset is False:
            x = self.subset_data._x_subset_data
            y = self.subset_data._y_subset_data
        else:
            x, y = self._get_subset_data(subset)
        self.do_local_opt(individual, subset)
        num_params = individual.get_number_local_optimization_params()
        f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)
        ssqe = np.sum((f - y) ** 2)
        if len(f) <= num_params:
            var_ols = ssqe
            #HUGE ISSUE HERE --> I THINK WE NEED TO RETHINK THE PROPOSAL PROCESS
        else:
            var_ols = ssqe / (len(f) - num_params)
        try:
            cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))
        except:
            #f_deriv += 1e-6
            cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv)+1e-6)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            import pdb;pdb.set_trace()

        return individual.constants, cov, var_ols, ssqe#, noise_std_dev

    def _get_dists(self, individual, num_multistarts, subset=False):

        param_dists = []
        cov_estimates = []
        trigger = False
        for _ in range(8*num_multistarts):
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
            if len(param_dists) == num_multistarts:
                break
        if not param_dists:
            raise RuntimeError('Could not generate any valid proposal '
                               'distributions')
            return None, None
        return param_dists, cov_estimates

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

    def _check_for_negative_added_noise(self, samples):

        for subset in range(len(self._src_num_pts)):
            key = f'std_dev{subset}'
            samples_i = samples[key]
            if True in samples_i < 0:
                print('!!!Negative values for added noise found!!!')
                break

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
