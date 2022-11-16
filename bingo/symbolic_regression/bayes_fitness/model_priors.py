from smcpy import AdaptiveSampler, \
                  ImproperUniform, \
                  MultiSourceNormal, \
                  InvGamma

noise_priors = {'ImproperUniform': lambda particles: ImproperUniform(0,None),
                'InverseGamma':lambda particles: InvGamma(particles=particles)}

class Priors:

    def __init__(self, noise_prior):
        self._noise_prior = noise_prior

    def _create_priors(self, ind, noise_terms, particles):

        n_params = ind.get_number_local_optimization_params()
        param_names = self.get_parameter_names(ind) + \
                    [f'std_dev{i}' for i in range(len(self._multisource_num_pts))]
        priors = [ImproperUniform() for _ in range(n_params)] + \
                [noise_priors[self._noise_prior](particles) for _ in range(len(noise_terms))]
        
        return param_names, priors

    @staticmethod
    def get_parameter_names(individual):
        num_params = individual.get_number_local_optimization_params()
        return [f'p{i}' for i in range(num_params)]
