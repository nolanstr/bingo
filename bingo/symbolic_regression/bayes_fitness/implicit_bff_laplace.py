import os
import sys
import h5py
import scipy
import sympy as sy
from scipy.stats import uniform, norm, invgamma
import numpy as np
import torch
from sympy import *
import matplotlib.pyplot as plt

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
from bingo.symbolic_regression.agraph.pytorch_evaluation_backend\
                        import evaluation_backend 
from smcpy.log_likelihoods import BaseLogLike
from smcpy import AdaptiveSampler
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import ImproperUniform
from mpi4py import MPI


class ImplicitLaplaceBayesFitnessFunction:
    def __init__(
        self,
        num_particles,
        mcmc_steps,
        ess_threshold,
        training_data,
        clo,
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
        self._iters = iters

    def __call__(self, individual, return_nmll_only=True):

        assert isinstance(individual, PytorchAGraph), "PytorchAgraph must be used"

        try:
            p = individual.get_number_local_optimization_params()
            dx, theta_hat, cov = self._perform_MLE(individual)
            vals, vecs = np.linalg.eig(cov)
            cov = np.diag(vals)
            import pdb;pdb.set_trace()
            """
            Removing correlation terms due to apriori knowledge of noise. 
            """
            shift_term = self.compute_ratio_term(individual)
            if shift_term == 0:
                return np.nan
            else:
                K = cov.shape[0]
                n = dx.shape[0]
                cov_inv = np.linalg.inv(cov)
                term1 = (-n*K/2) * np.log(2*np.pi)
                term2 = (-n/2) * np.log(np.linalg.det(cov))
                term3 = -0.5 * np.sum([np.matmul(dx_i.reshape((1,-1)), 
                    np.matmul(cov_inv, dx_i.reshape((-1,1)))) for \
                            dx_i in dx])
                log_likelihood = term1 + term2 + term3 

                nmll = (1-self._b) * log_likelihood + \
                            (p/2)*np.log(self._b) + np.log(shift_term)
                if shift_term==0:
                    return np.nan
        except:
            nmll = np.nan

        return -nmll
    
    def compute_ratio_term(self, ind):
        dx = self._cont_local_opt._fitness_function.estimate_dx(ind,
                                                self.training_data.x)
        pos_prob = np.prod(np.sum(dx.squeeze()>=0, axis=0)/dx.shape[0])
        neg_prob = np.prod(np.sum(dx.squeeze()<=0, axis=0)/dx.shape[0])
        if (pos_prob==1) and (neg_prob==1):
            #If the model perfectly predicts all data.
            return 1
        return ((pos_prob * neg_prob) / pow(0.5, 2*dx.shape[1]))

    def _perform_MLE(self, ind):
        
        self._cont_local_opt(ind)
        theta_hat = ind.get_local_optimization_params()
        dx = self._cont_local_opt._fitness_function.estimate_dx(ind,
                                                self.training_data.x)
        dx *= (self.training_data.x.shape[1]**0.5)
        n = self._training_data.x.shape[0]
        ssqe = np.square(np.linalg.norm(dx, axis=1)).sum(axis=0)
        var_ols = ssqe/n
        f, df = ind.evaluate_equation_with_x_gradient_at(
                                                self.training_data.x)
        cov = np.cov(dx.T)

        return dx, theta_hat, cov

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
