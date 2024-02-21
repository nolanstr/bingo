"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""
import numpy as np

import torch
from torch.autograd import grad
from  torch.distributions import multivariate_normal
from .operator_eval import forward_eval_function
from .evaluation_backend import _get_torch_const, evaluate

ENGINE = "Python"

def first_order_optimization(pytorch_repr, x, constants, 
                                        iters=100, tol=np.inf):
    """Evaluate equation and take derivative wrt x, c, and xc

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.
    x : MxD array of numeric.
        Values at which to evaluate the equations. D is the number of
        dimensions in x and M is the number of data points in x.
    constants : list-like of numeric.
        numeric constants that are used in the equation

    Returns
    -------
    MxD array of numeric or MxL array of numeric:
        Derivatives of all dimensions of x/constants at location x.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    constants = _get_torch_const(constants, x.size(1))
    eval, J, dx, df_dx, dJ_dc = execute_first_order_optimization(
                                        pytorch_repr, x, constants,
                                        iters=iters, tol=tol)
    return eval, J, dx, df_dx, dJ_dc 

def execute_first_order_optimization(pytorch_repr, x, constants, 
                                                iters=100, tol=np.inf):

    inputs1 = x
    inputs2 = constants
    inputs1.requires_grad = True
    inputs2.requires_grad = True
    x_dim, n_x = x.shape
    c_dim, n_c = constants.shape[:-1]
    dx = torch.zeros_like(x)
    dx.requires_grad = True
    data = x.clone()

    for _ in range(0, iters+1):
        eval = evaluate(pytorch_repr, inputs1, constants, final=False)
        df_dx = grad(outputs=eval.sum(), 
                        inputs=inputs1, 
                        create_graph=True, retain_graph=True, 
                        materialize_grads=True)[0].squeeze().T
        _dx = -(eval * df_dx) / \
            torch.pow(torch.norm(df_dx, p=2, dim=1), 2).reshape((-1,1))
        dx = torch.add(dx, _dx.T)
        inputs1 = torch.add(inputs1, _dx.T)
        if torch.abs(_dx).max() < 1e-6:
            break
    
    eval = evaluate(pytorch_repr, inputs1, constants, final=False)
    #eval[torch.isnan(eval)] = torch.inf
    df_dx = grad(outputs=eval.sum(), 
                    inputs=inputs1, 
                    create_graph=True, retain_graph=True, 
                    materialize_grads=True)[0].squeeze().T
    J = torch.sum(torch.pow(torch.norm(dx, p=2, dim=1), 2)) + \
                    torch.abs(constants[:,0,0]).sum()
    dJ_dc = grad(outputs=J, 
                    inputs=inputs2, 
                    create_graph=True, retain_graph=True, 
                    materialize_grads=True)[0].squeeze()
    #print(constants[:,0,0])
    if torch.std(eval) > tol:
        dx *= torch.nan
        J *= torch.nan
        dJ_dc *= torch.nan
        
    #print(constants[:,0,0].detach().numpy(),
    #        torch.abs(_dx).max().reshape(1).detach().numpy()[0])
    #print(dJ_dc.sum(axis=1).detach().numpy())
    #print()
    return eval.detach().numpy().reshape((n_x, -1)),\
            J.detach().numpy().reshape(1)[0],\
            dx.T.detach().numpy(), df_dx.detach().numpy(),\
            dJ_dc.T.detach().numpy()

