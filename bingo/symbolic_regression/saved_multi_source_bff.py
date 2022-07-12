import numpy as np

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

class MultiSourceBayesFitnessFunction(FitnessFunction):

    def __init__(self, continuous_local_opt, src_num_pts=None,
                 num_particles=150, mcmc_steps=12,
                 ess_threshold=0.75, std=None,
                 return_nmll_only=True, num_multistarts=1,
                 uniformly_weighted_proposal=True, 
                 random_sample_subsets=1.0):

        self._num_particles = num_particles
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._std = std
        self._return_nmll_only = return_nmll_only
        self._num_multistarts = num_multistarts
        self._uniformly_weighted_proposal = uniformly_weighted_proposal
        self._random_sample_subsets = random_sample_subsets
        
        if src_num_pts is None:
            raise NotImplementedError(
            'Eventually this class will be merged with the main bff class')
            src_num_pts = [continuous_local_opt.training_data.x.shape[0]]
        

        self._all_src_num_pts = tuple(src_num_pts)
        self._src_num_pts = tuple(src_num_pts)
        self._full_src_num_pts = self._src_num_pts

        num_observations = len(continuous_local_opt.training_data.x)
        self._norm_phi = 1 / np.sqrt(num_observations)

        self._cont_local_opt = continuous_local_opt
        self._eval_count = 0
        if random_sample_subsets != 1.0:
            self._src_num_pts = tuple([int(src_pts*random_sample_subsets + 0.5)\
                                              for src_pts in self._src_num_pts])
        self.subset_data = SubsetExplicitTrainingData(deepcopy(
                                                self.training_data),
                                                self._full_src_num_pts,
                                                self._src_num_pts)
        self._create_additive_noise_clos(len(src_num_pts))

    def __call__(self, individual):

        param_names = self.get_parameter_names(individual)
        try:
            proposal = self.generate_proposal_samples(individual,
                                                      self._num_particles)
            print(proposal)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            print('error with proposal creation')
            if self._return_nmll_only:
                return np.nan
            return np.nan, None, None

        priors = [ImproperUniform() for _ in range(len(param_names))]

        if self._std is None:
            for i in range(len(self._src_num_pts)):
                priors.append(ImproperUniform(0, None))
                param_names.append(f'std_dev{i}')

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
        individual._needs_opt = True
        if subset is False:
            _ = self._cont_local_opt(individual)
        else:
            _ = self._additive_noise_clos[subset](individual)
        return individual

    def generate_proposal_samples(self, individual, num_samples):

        param_names = self.get_parameter_names(individual)
        pdf = np.ones((num_samples, 1))
        samples = np.ones((num_samples, len(param_names)))

        num_multistarts = self._num_multistarts
        cov_estimates = []

        if len(param_names) > 0:

            param_dists, cov_estimates = self._get_dists(individual, 
                                                            num_multistarts)
            pdf, samples = self._get_samples_and_pdf(param_dists, num_samples)

        if self._std is None:
            for subset, len_data in enumerate(self._src_num_pts):

                param_dists, cov_estimates = self._get_dists(individual, 
                                                                num_multistarts,
                                                                subset)
                import pdb;pdb.set_trace()
                var = lambda ssqe: ssqe/len_data
                var_var = lambda ssqe, mu4: \
                  (mu4/len_data) - ((var(ssqe)**2) * (len_data-3)) /\
                  (len_data*(len_data-1))
                var_means = [np.sqrt(var(ssqe)) for _, _, _, ssqe, _  \
                                                    in cov_estimates]
                var_vars = [np.sqrt(var_var(ssqe, mu4)) for _, _, _, ssqe, mu4  \
                                                    in cov_estimates]
                noise_dists = [norm(var_mean, var_vars) \
                            for var_mean, var_vars  in zip(var_means, var_vars)]

                noise_pdf, noise_samples = self._get_samples_and_pdf(noise_dists,
                                                                     num_samples)
                samples = np.concatenate((samples, noise_samples), axis=1)
                param_names.append(f'std_dev{subset}')

            pdf *= noise_pdf

        if self._uniformly_weighted_proposal:
            pdf = np.ones_like(pdf)

        samples = dict(zip(param_names, samples.T))
        self._check_for_negative_added_noise(samples)

        return samples, pdf
    
    def randomize_subsets(self):
        self.subset_data.random_sample(self._full_src_num_pts, 
                                                self._src_num_pts)
    def _get_subset_data(self, subset):
        return self.subset_data.get_subset(subset)

    def estimate_covariance(self, individual, subset=False):
        if subset is False:
            x = self.training_data.x
            y = self.training_data.y
        else:
            x, y = self._get_subset_data(subset)

        self.do_local_opt(individual, subset)
        num_params = individual.get_number_local_optimization_params()

        f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)
        
        ssqe = np.sum((y - f) ** 2)
        var_ols = ssqe / (len(f) - num_params)
        cov = var_ols * np.linalg.inv(f_deriv.T.dot(f_deriv))
        mu4 = np.sum((y-f)**4) / x.shape[0]

        return individual.constants, cov, var_ols, ssqe, mu4

    def _get_dists(self, individual, num_multistarts, subset=False):

        param_dists = []
        cov_estimates = []
        for _ in range(8*num_multistarts):
            mean, cov, var_ols, ssqe, mu4 = self.estimate_covariance(individual,
                                                                        subset)
            try:
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                if min_eig < 0:
                    cov += 1e-12 * np.eye(*cov.shape)
                dists = mvn(mean, cov, allow_singular=True)
            except ValueError as e:
                continue
            cov_estimates.append((mean, cov, var_ols, ssqe, mu4))
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
            new_samples = np.random.choice(distributions) \
                .rvs(missed_samples).reshape((missed_samples, -1))
            K4-standardandom.choice(distributions) \
                .rvs(missed_samples).reshape((missed_samples, -1))
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
