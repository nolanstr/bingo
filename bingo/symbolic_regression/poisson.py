import numpy as np
import problems.data_generation_helpers as util
import torch


def analytic_solution(X, k):
    return np.sin(k * X[:, 0]) * np.sin(k * X[:, 1])

def get_pdefn(k):

    def pdefn(X, U):

        k = np.pi

        pen_mult = np.inf

        if U.grad_fn is not None:

            u_x = torch.autograd.grad(
                U.sum(), X[0], create_graph=True, allow_unused=True)[0]
            u_y = torch.autograd.grad(
                U.sum(), X[1], create_graph=True, allow_unused=True)[0]

            if u_x is not None or u_y is not None:
                pen_mult = np.inf

            if u_x is not None and u_y is not None:
                pen_mult = np.inf

                if u_x.grad_fn is not None:
                    u_xx = torch.autograd.grad(
                        u_x.sum(), X[0], create_graph=True, allow_unused=True)[0]
                else:
                    u_xx = torch.zeros_like(U)

                if u_y.grad_fn is not None:
                    u_yy = torch.autograd.grad(
                        u_y.sum(), X[1], create_graph=True, allow_unused=True)[0]
                else:
                    u_yy = torch.zeros_like(U)

                if u_xx is not None and u_yy is not None:

                    return [u_xx + u_yy + 2. * (k**2.) * torch.sin(k*X[0]) * torch.sin(k*X[1])]

        return [torch.ones_like(U) * pen_mult]

    return pdefn

def gen_training_data(k, low=[0, 0], high=[1, 1], n_b=16, n_df=50):

    X_boundary = util.boundary_X_2d(low, high, n_b)
    U_boundary = analytic_solution(X_boundary, k)[:, None]

    X_df = np.random.uniform(low=low, high=high, size=(n_df, 2))

    return X_boundary, U_boundary, X_df


def gen_testing_data(k, low=[0, 0], high=[1, 1], n=256):
    X = util.grid_X_2d(low, high, n)
    U = analytic_solution(X, k)[:, None]

    return X, U
