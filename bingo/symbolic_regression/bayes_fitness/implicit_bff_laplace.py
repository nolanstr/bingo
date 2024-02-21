import numpy as np
import scipy
import torch

from bingo.symbolic_regression.agraph.pytorch_agraph import PytorchAGraph
from bingo.symbolic_regression.agraph.pytorch_evaluation_backend\
                        import evaluation_backend 

class ImplicitLaplaceBayesFitnessFunction:
    def __init__(
        self,
        training_data,
        clo,
        iters=10,
    ):
        self._eval_count = 0
        self._training_data = training_data
        self._cont_local_opt = clo

        n = training_data.x.shape[0]
        self._b = 1 / np.sqrt(n)
        self._iters = iters

    def __call__(self, individual, return_nmll_only=True):

        assert isinstance(individual, PytorchAGraph), "PytorchAgraph must be used"

        try:
            individual._needs_opt = True
            p = individual.get_number_local_optimization_params()
            #import pdb;pdb.set_trace()
            dx, theta_hat, cov = self._perform_MLE(individual)
            if np.all(np.isnan(dx)):
                return np.nan

            vals, vecs = np.linalg.eig(cov)
            cov = np.diag(vals)
            """
            Removing correlation terms due to apriori knowledge of noise. 
            """
            shift_term = self.compute_ratio_term(dx)
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
                            (p/2)*np.log(self._b)

                shift_term = 1
                if shift_term==0:
                    return np.nan

            return -nmll + np.log(shift_term)

        except:

            return np.nan
    
    def compute_ratio_term(self, dx):

        if np.all(dx==0):
            #If the model perfectly predicts all data.
            return 1

        pos_prob = np.prod(np.sum(dx.squeeze()>=0, axis=0)/dx.shape[0])
        neg_prob = np.prod(np.sum(dx.squeeze()<=0, axis=0)/dx.shape[0])

        return ((pos_prob * neg_prob) / pow(0.5, 2*dx.shape[1]))

    def _perform_MLE(self, ind):
        
        fit = self._cont_local_opt(ind)
        if np.isnan(fit):
            return np.array([np.nan]), None, None
        dx, error, ssqe = \
        self._cont_local_opt._fitness_function.evaluate_fitness_vector(ind,
                return_all_fitness_metrics=True)
        theta_hat = ind.get_local_optimization_params()
        dx *= (self.training_data.x.shape[1]**0.5) #Heuristic
        n = self._training_data.x.shape[0]
        ssqe = np.square(np.linalg.norm(dx, axis=1)).sum(axis=0)
        var_ols = ssqe/n
        f, df = ind.evaluate_equation_with_x_gradient_at(
                                                self.training_data.x)
        cov = np.cov(dx.T)

        return dx, theta_hat, cov

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
