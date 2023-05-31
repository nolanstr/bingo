"""Component Generator for Agraph equations.

This module covers the random generation of components of an Acyclic graph
command stack. It can generate full commands or sub-components such as
operators, terminals, and their associated parameters.
"""
import logging
import numpy as np

from copy import deepcopy
from sympy.core import Expr

from .operator_definitions import OPERATOR_NAMES
from .string_parsing import eq_string_to_command_array_and_constants
from ...util.probability_mass_function import ProbabilityMassFunction
from ...util.argument_validation import argument_validation

LOGGER = logging.getLogger(__name__)


class ComponentGenerator:
    """Generates commands or components of a command for an agraph stack

    Parameters
    ----------
    input_x_dimension : int
        number of independent variables
    num_initial_load_statements : int
        number of commands at the beginning of stack which are required to be
        terminals. Default 1
    terminal_probability : float [0.0-1.0]
        probability that a new node will be a terminal. Default 0.1
    constant_probability : float [0.0-1.0] (optional)
        probability that a new terminal will be a constant

    Attributes
    ----------
    input_x_dimension : int
        number of independent variables
    """
    @argument_validation(input_x_dimension={">=": 0},
                         num_initial_load_statements={">=": 1},
                         terminal_probability={">=": 0.0, "<=": 1.0},
                         operator_probability={">=": 0.0, "<=": 1.0},
                         equation_probability={">=": 0.0, "<=": 1.0},
                         constant_probability={">=": 0.0, "<=": 1.0})
    def __init__(self, input_x_dimension, num_initial_load_statements=1,
                 terminal_probability=0.1,
                 operator_probability=0.9,
                 equation_probability=0.0,
                 constant_probability=None):

        self.input_x_dimension = input_x_dimension
        self._num_initial_load_statements = num_initial_load_statements

        self._terminal_pmf = self._make_terminal_pdf(constant_probability)
        self._operator_pmf = ProbabilityMassFunction()
        self._equation_pmf = ProbabilityMassFunction()
        self._full_random_command_function_pmf = \
            self._make_random_command_pmf(terminal_probability,
                                          operator_probability,
                                          equation_probability)
        self._random_command_function_pmf = \
            self._make_random_command_pmf(terminal_probability,
                                          operator_probability,
                                          0)

    def _make_terminal_pdf(self, constant_probability):
        if constant_probability is None:
            terminal_weight = [1, self.input_x_dimension]
        else:
            terminal_weight = [constant_probability,
                               1.0 - constant_probability]
        return ProbabilityMassFunction(items=[1, 0], weights=terminal_weight)

    def _make_random_command_pmf(self, terminal_probability,
                                 operator_probability, equation_probability):
        command_weights = [terminal_probability,
                           operator_probability,
                           equation_probability]
        return ProbabilityMassFunction(items=[self.random_terminal_command,
                                              self.random_operator_command,
                                              self.random_equation],
                                       weights=command_weights)

    def add_operator(self, operator_to_add, operator_weight=None):
        """Add an operator number to the set of possible operators

        Parameters
        ----------
        operator_to_add : int, str
            operator integer code (e.g. 2, 3) defined in Agraph operator maps
            or an operator string description (e.g. "+", "addition")
        operator_weight : number
                          relative weight of operator probability
        """
        if isinstance(operator_to_add, str):
            operator_number = self._get_operator_number_from_string(
                operator_to_add)
        else:
            operator_number = operator_to_add

        self._operator_pmf.add_item(operator_number, operator_weight)

    def add_equation(self, equation_to_add, equation_weight=None):
        """Add an equation to the set of possible equations

        Parameters
        ----------
        equation_to_add : str, sympy expr, command_arr
            either a command array, sympy str, or sympy expression of
            the equation to add
        equation_weight : number
            relative weight of equation probability
        """

        # string or sympy expression, need to turn into a command array
        if isinstance(equation_to_add, str) \
                or isinstance(equation_to_add, Expr):
            equation_to_add, _ = \
                eq_string_to_command_array_and_constants(str(equation_to_add))

        self._equation_pmf.add_item(equation_to_add, equation_weight)

    @staticmethod
    def _get_operator_number_from_string(operator_string):
        for operator_number, operator_names in OPERATOR_NAMES.items():
            if operator_string in operator_names:
                return operator_number
        raise ValueError(f"Could not find operator {operator_string}. ")

    def random_command(self, stack_location):
        """Get a random command

        Parameters
        ----------
        stack_location : int
            location in the stack for the command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2]

        """
        if stack_location < self._num_initial_load_statements:
            return self.random_terminal_command(stack_location)
        return self._random_command_function_pmf.draw_sample()(stack_location)

    def random_command_w_eq(self, stack_location):
        """Get a random command with equations

        Parameters
        ----------
        stack_location : int
            location in the stack for the command

        Returns
        -------
        array of array of int
            a random command in the form [[node, parameter 1, parameter 2]]
            or
            a random equation in the form
                [[node, parameter 1, parameter 2],
                [node, parameter 1, parameter 2],
                ...]

        """
        if stack_location < self._num_initial_load_statements:
            return self.random_terminal_command(stack_location)
        command = self._full_random_command_function_pmf.draw_sample()(stack_location)
        if len(command.shape) == 1:
            command = np.array([command])
        return command


    def random_operator_command(self, stack_location):
        """Get a random operator (non-terminal) command

        Parameters
        ----------
        stack_location : int
            location in the stack for the command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2]

        """
        return np.array([self.random_operator(),
                         self.random_operator_parameter(stack_location),
                         self.random_operator_parameter(stack_location)],
                        dtype=int)

    def random_operator(self):
        """Get a random operator

         Get a random operator from the list of possible operators.

        Returns
        -------
        int
            an operator number
        """
        return self._operator_pmf.draw_sample()

    @staticmethod
    def random_operator_parameter(stack_location):
        """Get random operator parameter

        Parameters
        ----------
        stack_location : int
            location of command in stack

        Returns
        -------
        int
            parameter to be used in an operator command


        Notes
        -----
        The returned random operator parameter is guaranteed to be less than
        stack_location.
        """
        return np.random.randint(stack_location)

    def random_terminal_command(self, _=None):
        """Get a random terminal (non-operator) command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2]

        """
        terminal = self.random_terminal()
        param = self.random_terminal_parameter(terminal)
        return np.array([terminal, param, param], dtype=int)

    def random_terminal(self):
        """Get a random terminal

         Get a random VARIABLE or CONSTANT terminal.

        Returns
        -------
        int
            terminal number (VARIABLE or CONSTANT)
        """
        return self._terminal_pmf.draw_sample()

    def random_terminal_parameter(self, terminal_number):
        """Get random terminal parameter

        Parameters
        ----------
        terminal_number : int
            terminal number for which random parameter should be generated

        Returns
        -------
        int
            parameter to be used in a terminal command
        """
        if terminal_number == 0:
            param = np.random.randint(self.input_x_dimension)
        else:
            param = -1
        return param

    @staticmethod
    def _shift_command_array(cmd_arr, stack_location):
        """Shift indices of command array to start at
        stack_location instead of 0"""
        new_cmd_arr = deepcopy(cmd_arr)
        operator_idx = np.logical_and.reduce((new_cmd_arr[:, 0] != -1,
                                              new_cmd_arr[:, 0] != 0,
                                              new_cmd_arr[:, 0] != 1))
    
        # add stack_location to the params of operator commands
        new_cmd_arr[operator_idx] = new_cmd_arr[operator_idx] + \
                                np.array([0, stack_location, stack_location])
        return new_cmd_arr

    def random_equation(self, stack_location):
        command_array = self._equation_pmf.draw_sample()
        return self._shift_command_array(command_array, stack_location)

    def get_number_of_terminals(self):
        """Gets number of possible terminals

        Returns
        -------
        int :
            number of terminals
        """
        return len(self._terminal_pmf.items)

    def get_number_of_operators(self):
        """Gets number of possible operators

        Returns
        -------
        int :
            number of operators
        """
        return len(self._operator_pmf.items)
