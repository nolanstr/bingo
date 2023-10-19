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
    def __init__(
        self,
        num_particles,
        mcmc_steps,
        ess_threshold,
        training_data,
        clo,
        ensemble=8,
        iters=10,
    ):
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
        self._iters = iters

    def __call__(self, individual, return_nmll_only=True):
        assert isinstance(individual, PytorchAGraph), "PytorchAgraph must be used"

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
                sampled_params = [
                    dist.rvs(self._num_particles) for dist in prop_dists[:-1]
                ] + [np.sqrt(prop_dists[-1].rvs(self._num_particles))]
                params_dict = dict(zip(param_names, sampled_params))

                proposal = [
                    params_dict,
                    np.ones(self._num_particles) / self._num_particles,
                ]
                noise = None
                log_like_args = [(self._training_data.x.shape[0]), noise, self._iters]
                log_like_func = ImplicitLikelihood
                vector_mcmc = VectorMCMC(
                    lambda info: self._eval_model(ind, info[0], info[1]),
                    self._training_data.x,
                    priors,
                    log_like_args,
                    log_like_func,
                )

                mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
                smc = AdaptiveSampler(mcmc_kernel)
                step_list, marginal_log_likes = smc.sample(
                    self._num_particles,
                    self._mcmc_steps,
                    self._ess_threshold,
                    required_phi=self._b,
                    proposal=proposal,
                )
                nmll = -1 * (
                    marginal_log_likes[-1] - marginal_log_likes[smc.req_phi_index[0]]
                )
                mean_params = np.average(
                    step_list[-1].params,
                    weights=step_list[-1].weights.flatten(),
                    axis=0,
                )
                individual.set_local_optimization_params(mean_params[:-1])
                shift_term = self.compute_ratio_term(ind, vector_mcmc)
                
                nmll += shift_term 
                
                fits[i] = nmll

                if not return_nmll_only:
                    return np.nanmedian(fits), marginal_log_likes, step_list

            except:
                fits[i] = np.nan

        fits[np.isinf(fits)] = np.nan

        if np.isnan(fits).sum() > self._ensemble:
            if not return_nmll_only:
                return np.nan, np.nan, np.nan
            else:
                return np.nan
        else:
            return np.nanmedian(fits)
    
    def compute_ratio_term(self, ind, vector_mcmc):
        n, ssqe, dx = vector_mcmc._log_like_func.estimate_ssqe(
                        ind.constants.T, return_ssqe_only=False)
        pos_prob = np.prod(np.sum(dx.squeeze()>0, axis=0)/dx.shape[0])
        neg_prob = np.prod(np.sum(dx.squeeze()<0, axis=0)/dx.shape[0])
        
        return np.log((pos_prob * neg_prob) / 0.25) 

    def _estimate_proposal(self, ind):
        self._cont_local_opt(ind)
        params = ind.get_local_optimization_params()
        ssqe = self._cont_local_opt._fitness_function.evaluate_fitness_vector(ind)
        n = self._training_data.x.shape[0]
        var = ssqe / n
        # ns = 0.0001
        # prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
        #            [invgamma((ns + n)/2, scale=(ns*var + ssqe)/2)]
        # prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
        #            [uniform(loc=0, scale=2*var)]
        prop_dists = [norm(loc=mu, scale=abs(0.1 * mu)) for mu in params] + [
            uniform(loc=0, scale=10)
        ]
        return prop_dists

    def _eval_model(self, ind, X, params):

        vals = []
        f = np.empty((X.shape[0], X.shape[2]))
        df_dx = np.zeros((X.shape[1], X.shape[0], X.shape[2]))
        df2_d2x = np.zeros_like(df_dx)

        for i in range(params.shape[0]):
            ind.set_local_optimization_params(np.expand_dims(params[i], axis=1))
            ind._simplified_constants = np.array(np.expand_dims(params[i], axis=1))
            _f = ind.evaluate_equation_at(X[:, :, i])
            partials = [
                ind.evaluate_equation_with_x_partial_at(X[:, :, i], [j] * 2)[1]
                for j in range(X.shape[1])
            ]
            _df_dx = np.stack([partial[0] for partial in partials])
            _df2_d2x = np.stack([partial[1] for partial in partials])
            f[:, i] = _f.squeeze()
            df_dx[:, :, i] = _df_dx.squeeze()
            df2_d2x[:, :, i] = _df2_d2x.squeeze()

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
        self.args = args[0:2]
        self._iters = args[2]

    def __call__(self, inputs):
        return self.estimate_likelihood(inputs)

    def estimate_likelihood(self, inputs):
        std_dev = inputs[:, -1]
        var = std_dev**2
        inputs = inputs[:, :-1]
        n, ssqe = self.estimate_ssqe(inputs)

        term1 = (-n / 2) * np.log(2 * np.pi * var)
        term2 = (-1 / (2 * var)) * ssqe
        log_like = term1 + term2

        return log_like

    def estimate_dx(self, data, inputs):
        vals = self.model([data, inputs])
        f, df_dx, df2_d2x = vals
        v = df_dx / (1 + 2 * df2_d2x)

        a = np.sum(df2_d2x * np.square(v), axis=0)
        b = -np.sum(df_dx * v, axis=0)
        c = f.astype(complex)

        l_pos = (-b + np.sqrt(np.square(b) - (4 * a * c))) / (2 * a)
        l_neg = (-b - np.sqrt(np.square(b) - (4 * a * c))) / (2 * a)

        x_pos = (-l_pos * v).real
        x_neg = (-l_neg * v).real
        # x_pos = x_pos.real + np.sqrt(np.square(x_pos.imag))
        # x_neg = x_neg.real + np.sqrt(np.square(x_neg.imag))

        # x_pos = np.sqrt(np.square(x_pos.real) + np.square(x_pos.imag))
        # x_neg = np.sqrt(np.square(x_neg.real) + np.square(x_neg.imag))
        return x_pos, x_neg

    def estimate_ssqe(self, inputs, return_ssqe_only=True, tol=1e-6):

        #shrinkage = 1 - 1/(self.data**2).sum(axis=0)
        #data = np.expand_dims(np.copy(shrinkage*self.data), axis=2)
        data = np.expand_dims(np.copy(self.data), axis=2)
        data = np.repeat(data, inputs.shape[0], axis=2)
        dx = np.zeros_like(data)

        for i in range(0, self._iters + 1):
            print(data.shape, inputs.shape)
            x_pos, x_neg = self.estimate_dx(data, inputs)
            ssqe_pos = np.square(np.linalg.norm(x_pos, axis=0)).sum(axis=0)
            ssqe_neg = np.square(np.linalg.norm(x_neg, axis=0)).sum(axis=0)
            ssqe_pos[np.isnan(ssqe_pos)] = np.inf
            ssqe_neg[np.isnan(ssqe_neg)] = np.inf

            x_pos = np.swapaxes(x_pos, 0, 1)
            x_neg = np.swapaxes(x_neg, 0, 1)

            _dx = np.where(x_pos, x_neg, x_pos <= x_neg)
            dx += _dx
            data += _dx
            if np.abs(_dx).min() < tol:
                break
            # ssqe = np.square(np.linalg.norm(dx, axis=0)).sum(axis=0)
        
        ssqe = np.square(np.linalg.norm(dx, axis=0)).sum(axis=0)
        if return_ssqe_only:
            return data.shape[0], ssqe

        return data.shape[0], ssqe, dx
