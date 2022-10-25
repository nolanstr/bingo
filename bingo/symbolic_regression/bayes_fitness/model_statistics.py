import numpy as np

from scipy.stats import invgamma, \
                        truncnorm, \
                        multivariate_normal as mvn
class Statistics:

    def __init__(self):
        pass

    def generate_proposal_samples(self, individual, num_samples, param_names):

        samples = np.ones((num_samples, len(param_names)))
        n_params = individual.get_number_local_optimization_params()
        if n_params > 0:

            param_dists, cov_estimates = self._get_dists(individual)
            _, samples[:,:n_params] = self._get_samples_and_pdf(param_dists,
                                                                    num_samples)

        for subset, len_data in enumerate(self._multisource_num_pts):
            
            param_dist, cov_estimates = self._get_dists(individual, subset)
            noise_pdf, noise_samples = self._get_added_noise_samples(cov_estimates,
                                                                     len_data, 
                                                                     num_samples)
            samples[:, n_params+subset] = noise_samples.flatten()

        pdf = np.ones((num_samples, 1))
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
        ns = 0.01
        noise_dists = [invgamma((ns + len_data) / 2,
                        scale=(ns * var_ols + ssqe) / 2)
                       for _, _, var_ols, ssqe in cov_estimates]

        noise_pdf, noise_samples = self._get_samples_and_pdf(noise_dists,
                                                             num_samples)
        return noise_pdf, np.sqrt(noise_samples)

    def estimate_covariance(self, individual, subset=None):
        
        self.do_local_opt(individual, subset)

        x, y = self.get_dataset(subset=subset)

        num_params = individual.get_number_local_optimization_params()
        f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)
        
        ssqe = np.sum((f - y) ** 2)
        #var_ols = ssqe / max(1, len(f) - num_params)
        var_ols = ssqe / len(f)
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

