import os
import sys
import h5py
import scipy
import sympy as sy
from scipy.stats import uniform, norm, invgamma
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
from smcpy.log_likelihoods import BaseLogLike
from smcpy import AdaptiveSampler
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import ImproperUniform
from mpi4py import MPI


class ImplicitBayesFitnessFunction:
    
    def __init__(self, num_particles, mcmc_steps, ess_threshold, 
                                        training_data, clo, ensemble=8):
        self._h = 5e-3
        self._eval_count = 0
        self._num_particles = num_particles
        self._mcmc_steps = mcmc_steps
        self._ess_threshold = ess_threshold
        self._training_data = training_data
        self._cont_local_opt = clo

        n = training_data.x.shape[0]
        self._b = 1 / n
        self._ensemble = ensemble

    def __call__(self, individual, return_nmll_only=True):
        
        assert isinstance(individual, PytorchAGraph), \
                "PytorchAgraph must be used"

        if not return_nmll_only:
            self._ensemble = 1
        fits = np.empty(self._ensemble)

        for i in range(self._ensemble):
            try:
                ind = individual.copy()
                n = ind.get_number_local_optimization_params()
                priors = n * [ImproperUniform()] + [ImproperUniform(0, None)]
                param_names = [f"P{i}" for i in range(n)] + ["std_dev"]
                prop_dists = self._estimate_proposal(ind)
                sampled_params = [dist.rvs(self._num_particles) \
                                    for dist in prop_dists[:-1]] + \
                                [np.sqrt(prop_dists[-1].rvs(self._num_particles))]
                params_dict = dict(zip(param_names, sampled_params)) 

                proposal = [params_dict, np.ones(self._num_particles)/
                                                        self._num_particles]
                noise = None
                log_like_args = [(self._training_data.x.shape[0]), noise] 
                log_like_func = ImplicitLikelihood
                vector_mcmc = VectorMCMC(lambda inputs: self._eval_model(ind, 
                                         self._training_data.x, inputs),
                                         np.zeros(self._training_data.x.shape[0]),
                                         priors,
                                         log_like_args,
                                         log_like_func)

                mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
                smc = AdaptiveSampler(mcmc_kernel)
                step_list, marginal_log_likes = \
                    smc.sample(self._num_particles, self._mcmc_steps,
                               self._ess_threshold,
                               required_phi=self._b,
                               proposal=proposal)
                nmll = -1 * (marginal_log_likes[-1] - 
                             marginal_log_likes[smc.req_phi_index[0]])
                mean_params = np.average(step_list[-1].params, 
                                 weights=step_list[-1].weights.flatten(), axis=0) 
                individual.set_local_optimization_params(mean_params[:-1])
                
                fits[i] = nmll
                #fits[i] = -marginal_log_likes[-1]
                print('a')
                if not return_nmll_only:
                    return np.nanmedian(fits), marginal_log_likes, step_list 

            except:
                fits[i] = np.nan
            
        fits[np.isinf(fits)] = np.nan

        if np.isnan(fits).sum() > self._ensemble:
            return np.nan
        else:
            return np.nanmedian(fits)

    def _estimate_proposal(self, ind):
        
        self._cont_local_opt(ind)
        params = ind.get_local_optimization_params()
        ssqe = self._cont_local_opt._fitness_function.evaluate_fitness_vector(
                                                                            ind)
        #ns = 0.0001
        n = self._training_data.x.shape[0]
        var = ssqe / n 
        #prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
        #            [invgamma((ns + n)/2, scale=(ns*var + ssqe)/2)]
        #prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
        #            [uniform(loc=0, scale=2*var)]
        prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
                    [uniform(loc=0, scale=10)]
        #import pdb;pdb.set_trace() 
        return prop_dists

    def _eval_model(self, ind, X, params):
        vals = []
        variables = sy.symbols("".join([f" X_{i} " for i in \
                                    range(X.shape[1])])[1:-1])
        ind.set_local_optimization_params(params.T)
        ind._simplified_constants = np.array(params.T)
        f = ind.evaluate_equation_at(X)
        if f.shape[1] != params.shape[0]:
            f = np.repeat(f, params.shape[0], axis=1)

        partials = [ind.evaluate_equation_with_x_partial_at(X, [i]*2)[1] \
                            for i in range(X.shape[1])]
        df_dx = np.stack([partial[0] for partial in partials])
        df2_d2x = np.stack([partial[1] for partial in partials])
        return f, np.stack(df_dx), np.stack(df2_d2x)

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

class ImplicitLikelihood(BaseLogLike):

    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args

    def __call__(self, inputs):
        return self.estimate_likelihood(inputs)
    
    def estimate_likelihood(self, inputs):
        std_dev = inputs[:,-1]
        var = std_dev ** 2
        inputs = inputs[:,:-1]
        n, ssqe = self.estimate_ssqe(inputs)
        term1 = (-n/2) * np.log(2*np.pi*var)
        term2 = (-1 / (2*var)) * ssqe
        log_like = term1 + term2

        return log_like
    
    def estimate_ssqe(self, inputs):

        vals = self.model(inputs)
        n = vals[0][:,0].shape[0]
        f, df_dx, df2_d2x = vals
        v = df_dx / (1 + 2*df2_d2x)
        
        a = np.sum(df2_d2x*np.square(v), axis=0)
        b = -np.sum(df_dx*v, axis=0)
        c = f.astype(complex)

        l_pos = (-b + np.sqrt(np.square(b) - (4*a*c))) / (2*a)
        l_neg = (-b - np.sqrt(np.square(b) - (4*a*c))) / (2*a)
        
        x_pos = -l_pos*v
        x_neg = -l_neg*v
        ssqe_pos = np.square(np.linalg.norm(x_pos, axis=0)).sum(axis=0)
        ssqe_neg = np.square(np.linalg.norm(x_neg, axis=0)).sum(axis=0)
        ssqe_pos[np.isnan(ssqe_pos)] = np.inf
        ssqe_neg[np.isnan(ssqe_neg)] = np.inf
        adjustment = 2 * (np.minimum(ssqe_pos, ssqe_neg)/\
                            np.maximum(ssqe_pos, ssqe_neg))
        ssqe = np.minimum(ssqe_pos, ssqe_neg)

        return n, ssqe
