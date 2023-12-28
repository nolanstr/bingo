"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""
import numpy as np

import torch
from torch.autograd import grad
from .operator_eval import forward_eval_function

ENGINE = "Python"


def _get_torch_const(constants, data_len):
    with torch.no_grad():
        # assumes simplified constants are up to date
        if isinstance(constants, tuple) and len(constants) > 0 and isinstance(constants[0], np.ndarray):
            constants = np.array(constants)

        try:
            if constants.ndim == 2:
                constants = torch.from_numpy(constants[:, None, :]).double()
                constants = constants.expand(-1, data_len, -1)

        # TODO second case is a special version of the first case, can just
        #   convert and then use first case (e.g., [10, 20] -> [[10], [20]])
        except AttributeError:
            constants = torch.tensor(constants).double()
            constants = constants.view(-1, 1, 1).expand(-1, data_len, -1)
        return constants


def get_pytorch_repr(command_array):
    # TODO see if we can do this more efficiently
    # TODO this reruns on every eval, how to return just expression?

    def get_expr(X, constants):  # assumes X is column-order
        expr = []

        for (node, param1, param2) in command_array:
            expr.append(forward_eval_function(node, param1, param2, X, constants,
                                              expr))

        return expr[-1]

    return get_expr


def evaluate(pytorch_repr, x, constants, final=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    if final:
        constants = _get_torch_const(constants, x.size(1))
    return_eval = get_pytorch_repr(pytorch_repr)(x, constants)
    if final:
        return _reshape_output(return_eval.detach().numpy(), constants, x)
    return return_eval


def _reshape_output(output, constants, x):
    x_dim = x.size(1)
    c_dim = 1
    if len(constants) > 0:
        if isinstance(constants[0], np.ndarray):
            c_dim = len(constants[0])
    if isinstance(output, np.ndarray):
        if output.shape == (x_dim, c_dim):
            return output
        elif output.shape == (x_dim,):
            return output.reshape((x_dim, 1))
    return np.ones((x_dim, c_dim)) * output


def evaluate_with_derivative(pytorch_repr, x, constants, wrt_param_x_or_c):
    """Evaluate equation and take derivative

    Evaluate the derivatives of the equation associated with an Agraph, at the
    values x.

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
    wrt_param_x_or_c : boolean
        Take derivative with respect to x or constants. True signifies
        derivatives are wrt x. False signifies derivatives are wrt constants.

    Returns
    -------
    MxD array of numeric or MxL array of numeric:
        Derivatives of all dimensions of x/constants at location x.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    constants = _get_torch_const(constants, x.size(1))
    eval, deriv = _evaluate_with_derivative(pytorch_repr, x, constants, wrt_param_x_or_c)
    return eval, deriv


def _evaluate_with_derivative(pytorch_repr, x, constants, wrt_param_x_or_c):
    inputs = x
    if not wrt_param_x_or_c:  # c
        inputs = constants
    inputs.requires_grad = True

    eval = evaluate(pytorch_repr, x, constants, final=False)

    if eval.requires_grad:
        derivative = grad(outputs=eval.sum(), inputs=inputs, create_graph=True, retain_graph=True, allow_unused=False)[0]
    else:
        derivative = None
    if derivative is None:
        derivative = torch.zeros((inputs.shape[0], eval.shape[0]))
    elif not wrt_param_x_or_c:
        derivative = derivative[:, :, 0]
    return _reshape_output(eval.detach().numpy(), constants, x), derivative.T.detach().numpy()

def evaluate_with_both_derivatives(pytorch_repr, x, constants):
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
    eval, deriv_1, deriv_2, deriv_3 = _evaluate_with_both_derivatives(
                    pytorch_repr, x, constants)
    return eval, deriv_1, deriv_2, deriv_3

def _evaluate_with_both_derivatives(pytorch_repr, x, constants):
    """
    deriv_1 = (x_term, n) --> deriv_1(2,:) --> df_dx1
    deriv_2 = (c_term, n) --> deriv_2(1,:) --> df_dc1
    deriv_3 = (x_term, c_term, n) --> deriv_3(1,2,:) --> d^2f/dx1dc2
    """
    inputs1 = x
    inputs2 = constants
    inputs1.requires_grad = True
    inputs2.requires_grad = True
    x_dim, n_x = x.shape
    c_dim, n_c = constants.shape[:-1]

    eval = evaluate(pytorch_repr, x, constants, final=False)

    if eval.requires_grad:
        deriv_1 = grad(outputs=eval.sum(), 
                        inputs=inputs1, 
                        create_graph=True, retain_graph=True, 
                        allow_unused=False)[0]
        deriv_2 = grad(outputs=eval.sum(), 
                        inputs=inputs2, 
                        create_graph=True, retain_graph=True, 
                        allow_unused=False)[0].squeeze()
        deriv_3 = torch.zeros((x_dim, c_dim, n_x))
        for i in range(x_dim):
            deriv_3[i,:,:] = grad(outputs=deriv_1[i,:].sum(), 
                                  inputs=inputs2, 
                                  create_graph=True, 
                                  retain_graph=True, 
                                  allow_unused=False)[0].squeeze()
    return eval.detach().numpy().reshape((n_x, -1)),\
            deriv_1.T.detach().numpy(), deriv_2.T.detach().numpy(),\
            deriv_3.detach().numpy()

def evaluate_iSMC_dJ_dc(pytorch_repr, x, constants):
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
    eval, J, dJ_dc = _evaluate_iSMC_dJ_dc(
                    pytorch_repr, x, constants)
    return eval, J, dJ_dc 

def _evaluate_iSMC_dJ_dc(pytorch_repr, x, constants):
    inputs1 = x
    inputs2 = constants
    inputs1.requires_grad = True
    inputs2.requires_grad = True
    x_dim, n_x = x.shape
    c_dim, n_c = constants.shape[:-1]

    eval = evaluate(pytorch_repr, x, constants, final=False)

    if eval.requires_grad:
        df_dx = grad(outputs=eval.sum(), 
                        inputs=inputs1, 
                        create_graph=True, retain_graph=True, 
                        allow_unused=False)[0]
        J = torch.pow(eval, 2) / \
            torch.pow(torch.norm(df_dx, dim=0).reshape((-1,1)), 2)
        dJ_dc = grad(outputs=J.sum(), 
                        inputs=inputs2, 
                        create_graph=True, retain_graph=True, 
                        allow_unused=False)[0].squeeze()
    return eval.detach().numpy().reshape((n_x, -1)),\
            J.detach().numpy().reshape((n_x, -1)),\
            dJ_dc.T.detach().numpy()

def evaluate_with_partials(pytorch_repr, x, constants, partial_order):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    if not isinstance(constants, np.ndarray):
        constants = np.array(constants).T

    final_eval = evaluate(pytorch_repr, x, constants, final=True)

    x = x[:, :, None].expand(-1, -1, constants.shape[1])
    x.requires_grad = True
    constants = _get_torch_const(constants, x.size(1))
    eval = evaluate(pytorch_repr, x, constants, final=False)

    partial = eval
    partials = []
    for variable in partial_order:
        try:
            partial = grad(outputs=partial.sum(), inputs=x,
                         allow_unused=True,
                         create_graph=True)[0][variable]
            if partial is None:
                partial = torch.zeros_like(x[0])
        except (IndexError, RuntimeError):
            partial = torch.zeros_like(x[0])
        partials.append(partial.detach().numpy())

    return _reshape_output(final_eval, constants, x), partials
