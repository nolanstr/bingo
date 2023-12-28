# TODO doc

import numpy as np
import logging
import torch

from .agraph import AGraph
from .pytorch_evaluation_backend import evaluation_backend
from .pytorch_evaluation_backend import implicit_backend

LOGGER = logging.getLogger(__name__)

class PytorchAGraph(AGraph):
    def evaluate_equation_at(self, x):
        if self._modified:
            self._update()
        try:
            return evaluation_backend.evaluate(
                self._simplified_command_array, x,
                self._simplified_constants)
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation", err)
            return torch.full((len(x[0]), 1), float("nan"))

    def evaluate_equation_with_x_gradient_at(self, x):
        """Evaluate `AGraph` and get its derivatives.

        Evaluate the `AGraph` equation at x and the gradient of x.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxD array of numeric)
            :math:`f(x)` and :math:`df(x)/dx_i`
        """
        if self._modified:
            self._update()
        try:
            f_of_x, df_dx = evaluation_backend.evaluate_with_derivative(
                self._simplified_command_array, x,
                self._simplified_constants, True)
            return f_of_x, df_dx
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/deriv", err)
            nan_f_of_x = torch.full((len(x[0]), 1), float("nan"))
            nan_df_dx = torch.full((len(x[0]), len(x)), float("nan"))
            return nan_f_of_x.detach().numpy(), nan_df_dx.detach().numpy()

    def evaluate_equation_with_local_opt_gradient_at(self, x):
        """Evaluate `AGraph` and get its derivatives.

        Evaluate the `AGraph` equation at x and get the gradient of constants.
        Constants are of length L.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxL array of numeric)
            :math:`f(x)` and :math:`df(x)/dc_i`
        """
        if self._modified:
            self._update()
        try:
            f_of_x, df_dc = evaluation_backend.evaluate_with_derivative(
                self._simplified_command_array, x,
                self._simplified_constants,
                False)
            return f_of_x, df_dc
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_f_of_x = torch.full((len(x[0]), 1), float("nan"))
            nan_df_dc = torch.full(
                (len(x[0]), self._simplified_constants.shape[0]), float("nan"))
            return nan_f_of_x.detach().numpy(), nan_df_dc.detach().numpy()

    def evaluate_equation_with_both_gradients_at(self, x):
        """Evaluate `AGraph` and get its derivatives.

        Evaluate the `AGraph` equation at x and get the gradient of x,
        constants, and constants and data (df_dx, df_dc, d^2f/dxdc).
        Constants are of length L.

        Parameters
        ----------
        x : MxD array of numeric.
            Values at which to evaluate the equations. D is the number of
            dimensions in x and M is the number of data points in x.

        Returns
        -------
        tuple(Mx1 array of numeric, MxL array of numeric)
            :math:`f(x)` and :math:`df(x)/dc_i`
        """
        if self._modified:
            self._update()
        try:
            f_of_x, df_dx, df_dc, d2f_dxdc = \
                evaluation_backend.evaluate_with_both_derivatives(
                self._simplified_command_array, x,
                self._simplified_constants)
            return f_of_x, df_dx, df_dc, d2f_dxdc
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_f_of_x = torch.full((len(x[0]), 1), float("nan"))
            nan_df_dc = torch.full(
                (len(x[0]), self._simplified_constants.shape[0]), float("nan"))
            return nan_f_of_x.detach().numpy(), nan_df_dc.detach().numpy()

    def compute_iSMC_dJ_dc(self, x):
        if self._modified:
            self._update()
        try:
            f, J, dJ_dc = evaluation_backend.evaluate_iSMC_dJ_dc(
                self._simplified_command_array, x,
                self._simplified_constants)
            return f, J, dJ_dc 
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_f_of_x = torch.full((len(x[0]), 1), float("nan"))
            nan_df_dc = torch.full(
                (len(x[0]), self._simplified_constants.shape[0]), float("nan"))
            return nan_f_of_x.detach().numpy(), nan_df_dc.detach().numpy()
    
    def iSMC_pytorch(self, x, iters=100, tol=np.inf):
        if self._modified:
            self._update()
        try:
            f, J, dx, df_dx, dJ_dc = \
                implicit_backend.first_order_optimization(
                self._simplified_command_array, x,
                self._simplified_constants, 
                iters, tol)
            return f, J, dx, df_dx, dJ_dc 
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/const-deriv", err)
            nan_f_of_x = torch.full((len(x[0]), 1), float("nan"))
            nan_df_dc = torch.full(
                (len(x[0]), self._simplified_constants.shape[0]), float("nan"))
            return nan_f_of_x.detach().numpy(), nan_df_dc.detach().numpy()

    def evaluate_equation_with_x_partial_at(self, x, partial_order):
        if self._modified:
            self._update()
        try:
            f_of_x, df_dx = evaluation_backend.evaluate_with_partials(
                self._simplified_command_array, x,
                self._simplified_constants,
                partial_order)
            return f_of_x, df_dx
        except (ArithmeticError, OverflowError, ValueError,
                FloatingPointError) as err:
            LOGGER.warning("%s in stack evaluation/partial-deriv", err)
            nan_f_of_x = torch.full((len(x[0]), 1), float("nan"))
            nan_df_dx = torch.full((len(x[0]), len(x)), float("nan"))
            return nan_f_of_x.detach().numpy(), nan_df_dx.detach().numpy()
