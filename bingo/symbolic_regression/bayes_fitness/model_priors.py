from smcpy import AdaptiveSampler,
                  ImproperUniform,
                  MultiSourceNormal

noise_priors = {'ImproperUniform':ImproperUniform(0,None),
                'InverseGamma':InvGamma()}

class ModelPriors:

    def __init__(self, noise_prior='ImproperUniform'):
        self._noise_prior = noise_prior

    def __call__(self, ind, noise_terms):

        n_params = ind.get_number_local_optimization_params()
        priors = [ImproperUniform() for _ in range(n_params)] + \
                [noise_priors[self._noise_prior] for _ in range(noise_terms)]
        
        return priors

    def upate_priors_for_inv_gamma(self, individual, subsets, priors):
        
        n_params = individual.get_number_local_optimization_params()
        self.do_local_opt(individual, None)

        for subset in subsets:

            x, y = self.subset_data.get_dataset(subset=subset)

            num_params = individual.get_number_local_optimization_params()
            f, f_deriv = individual.evaluate_equation_with_local_opt_gradient_at(x)

            diff = f - y
            mu = diff.mean()
            var = diff.var()
            alpha = np.square(mu) / var
            beta = mu / var
            priors[n_params+subset] = InvGamma(alpha=alpha, beta=beta)

        return priors

