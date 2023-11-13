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
            ind = individual.copy()
            p = ind.get_number_local_optimization_params()
            dx, theta_hat, cov = self._perform_MLE(ind)
            shift_term = self.compute_ratio_term(ind)
            if shift_term == 0:
                nmll = np.nan
            else:
                K = cov.shape[0]
                n = dx.shape[0]
                cov_inv = np.linalg.inv(cov)
                term1 = (-n*K/2) * np.log(2*np.pi)
                term2 = (-n/2) * np.log(np.linalg.det(cov))
                term3 = -0.5 * np.sum(np.matmul(dx, np.matmul(cov_inv, dx.T)))
                #import pdb;pdb.set_trace()
                #term3 = -0.5 * np.sum([np.matmul(dx_i, np.matmul(cov_inv, dx_i)) \
                #                           for dx_i in dx])
                log_likelihood = term1 + term2 + term3 
                #nmll = ((1-self._b) * log_likelihood / (p*K/2)) + \
                #            ((p/2) * np.log(self._b)) + np.log(shift_term)
                            #made changes to make consistent with iSMC
                nmll = (1-self._b) * log_likelihood / p + \
                                    np.log(shift_term)

        except:
            nmll = np.nan

        if not return_nmll_only:
            return -nmll, marginal_log_likes, step_list
        else:
            return -nmll
    
    def compute_ratio_term(self, ind):
        dx = self._cont_local_opt._fitness_function.estimate_dx(ind)
        pos_prob = np.prod(np.sum(dx.squeeze()>=0, axis=0)/dx.shape[0])
        neg_prob = np.prod(np.sum(dx.squeeze()<=0, axis=0)/dx.shape[0])
        if (pos_prob==1) and (neg_prob==1):
            #If the model perfectly predicts all data.
            return 1
        return ((pos_prob * neg_prob) / pow(0.5, 2*dx.shape[1]))

    def _perform_MLE(self, ind):
        
        self._cont_local_opt(ind)
        theta_hat = ind.get_local_optimization_params()
        dx = self._cont_local_opt._fitness_function.estimate_dx(ind)
        n = self._training_data.x.shape[0]
        cov = np.array([np.matmul(dx_i.reshape((-1,1)), dx_i.reshape((1,-1))) \
                            for dx_i in dx]).sum(axis=0) / n
        return dx, theta_hat, cov

    def _eval_model(self, ind, X, params):
        #n_data = X.shape[0]
        #n,m,d = X.shape
        #X_cust = X.transpose(2, 0, 1).reshape((-1, m))
        #X_cust = torch.from_numpy(X_cust).double()
        #cust_params = np.repeat(params, n_data, axis=0)
        #ind.set_local_optimization_params(cust_params.T)
        #ind._simplified_constants = np.array(cust_params.T)
        #_f = evaluation_backend.evaluate(ind._simplified_command_array,
        #                            X_cust, ind._simplified_constants,
        #                            final=False, return_pytorch_repr=True)
        #import pdb;pdb.set_trace()
        #partials = [
        #    ind.evaluate_equation_with_x_partial_at(X, [j] * 2)[1]
        #    for j in range(X.shape[1])
        #]
        import pdb;pdb.set_trace()
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
        n, ssqe, dx = self.estimate_ssqe(inputs, return_ssqe_only=False)

        term1 = (-n / 2) * np.log(2 * np.pi * var)
        term2 = (-1 / (2 * var)) * ssqe
        log_like = term1 + term2
        pos_prob = np.prod(np.sum(dx>0, axis=0)/dx.shape[0], axis=0)
        neg_prob = np.prod(np.sum(dx<0, axis=0)/dx.shape[0], axis=0)
        log_like += np.log(
                (pos_prob*neg_prob)/pow(0.5, 2*dx.shape[1]))

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
            if np.abs(_dx).max() < tol:
                break
            # ssqe = np.square(np.linalg.norm(dx, axis=0)).sum(axis=0)
        
        ssqe = np.square(np.linalg.norm(dx, axis=0)).sum(axis=0)
        if return_ssqe_only:
            return data.shape[0], ssqe

        return data.shape[0], ssqe, dx
