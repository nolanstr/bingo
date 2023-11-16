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


class MVNImplicitBayesFitnessFunction:
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
        self._b = 1 / np.sqrt(n)
        self._ensemble = ensemble
        self._iters = iters
        self._mvn_dims = training_data.x.shape[1]
        self._mvn_terms = int(self._mvn_dims*(self._mvn_dims+1)/2)

    def __call__(self, individual, return_nmll_only=True):
        assert isinstance(individual, PytorchAGraph), "PytorchAgraph must be used"

        if not return_nmll_only:
            self._ensemble = 1
        fits = np.empty(self._ensemble)

        for i in range(self._ensemble):
            try:
                ind = individual.copy()
                n = ind.get_number_local_optimization_params()
                priors = [ImproperUniform()] * n
                var_priors = [item for i in range(1, self._mvn_dims+1) \
                            for item in [ImproperUniform(0, None)] + \
                            [ImproperUniform()]*(self._mvn_dims-i)]
                priors += var_priors
                param_names = [f"P{i}" for i in range(n)] + \
                [f"var_{i}{j+i}" for i in range(self._mvn_dims) \
                            for j in range(self._mvn_dims-i)]
                prop_dists = self._estimate_proposal(ind)
                sampled_params = [
                    dist.rvs(self._num_particles) for dist in
                    prop_dists]
                params_dict = dict(zip(param_names, sampled_params))

                proposal = [
                    params_dict,
                    np.ones(self._num_particles) / self._num_particles,
                ]
                noise = [None] * self._mvn_terms
                log_like_args = [(self._training_data.x.shape[0]), noise, self._iters]
                log_like_func = MVNImplicitLikelihood
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
                import pdb;pdb.set_trace()
                nmll = -1 * (
                    marginal_log_likes[-1] - marginal_log_likes[smc.req_phi_index[0]]
                )
                mean_params = np.average(
                    step_list[-1].params,
                    weights=step_list[-1].weights.flatten(),
                    axis=0,
                )
                individual.set_local_optimization_params(mean_params[:-1])

                fits[i] = nmll

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
        ssqe = self._cont_local_opt._fitness_function.evaluate_fitness_vector(ind)
        # ns = 0.0001
        n = self._training_data.x.shape[0]
        var = ssqe / n
        # prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
        #            [invgamma((ns + n)/2, scale=(ns*var + ssqe)/2)]
        # prop_dists = [norm(loc=mu, scale=abs(0.1*mu)) for mu in params] + \
        #            [uniform(loc=0, scale=2*var)]
        prop_dists = [norm(loc=mu, scale=abs(0.1 * mu)) for mu in params]
        cov_dists = [item for i in range(1, self._mvn_dims+1) \
                        for item in [uniform(loc=0,scale=0.1)] + \
                        [uniform(loc=-0.05,scale=0.1)]*(self._mvn_dims-i)]
        return prop_dists + cov_dists

    def _eval_model(self, ind, X, params):
        vals = []

        f = np.empty((X.shape[0], X.shape[2]))
        df_dx = np.zeros((X.shape[1], X.shape[0], X.shape[2]))
        df2_d2x = np.zeros_like(df_dx)

        for i in range(params.shape[0]):
            ind.set_local_optimization_params(np.expand_dims(params[i], axis=1))
            ind._simplified_constants = np.array(np.expand_dims(params[i], axis=1))
            _f = ind.evaluate_equation_at(X[:, :, i])
            # if f.shape[1] != params.shape[0]:
            #    f = np.repeat(f, params.shape[0], axis=1)

            partials = [
                ind.evaluate_equation_with_x_partial_at(X[:, :, i], [i] * 2)[1]
                for i in range(X.shape[1])
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


class MVNImplicitLikelihood(BaseLogLike):
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args[0:2]
        self._iters = args[2]
        m = len(self.args[1])
        self._mvn_dims = int(np.sqrt(2*m + 0.25)-0.5)

    def __call__(self, inputs):
        return self.estimate_likelihood(inputs)
    
    def estimate_likelihood(self, inputs):
        cov_vals = inputs[:, -len(self.args[1]) :]
        cov = np.stack([np.lib.stride_tricks.sliding_window_view(row,
            (self._mvn_dims)) for row in cov_vals])
        inputs = inputs[:, : -len(self.args[1])]
        n, dx = self.estimate_ssqe(inputs)

        error_term = np.empty(inputs.shape[0])
        inv_cov = np.empty_like(cov)
        #cov[:,0,1] = 0
        #cov[:,1,0] = 0
        for i, (dx_i, cov_i) in enumerate(zip(dx.T, cov)):
            dx_i = np.expand_dims(dx_i.T, axis=2)
            inv_cov_i = np.linalg.inv(cov_i)
            inv_cov[i] = inv_cov_i
            errors = np.matmul(np.swapaxes(dx_i, 1, 2), np.matmul(inv_cov_i,
                                dx_i)).sum()
            if errors<0:
                pass
            error_term[i] = errors

        K = self._mvn_dims
        term1 = -n*K*np.log(2*np.pi)/2
        term2 = -n*np.log(np.linalg.det(cov))/2
        term3 = -0.5*error_term
        log_like = term1 + term2 + term3

        log_like[np.isnan(log_like)] = -np.inf

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

    def estimate_ssqe(self, inputs):
        data = np.expand_dims(np.copy(self.data), axis=2)
        data = np.repeat(data, inputs.shape[0], axis=2)
        dx = np.zeros_like(data)

        for i in range(0, self._iters + 1):
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
            # ssqe = np.square(np.linalg.norm(dx, axis=0)).sum(axis=0)

        return data.shape[0], dx*np.sqrt(self._mvn_dims)
