"""Generator of acyclic graph individuals.

This module contains the implementation of the generation of random acyclic
graph individuals.
"""
import numpy as np

try:
    from bingocpp import AGraph

    BINGOCPP = True
except (ImportError, KeyError, ModuleNotFoundError) as e:
    from .agraph import AGraph

    BINGOCPP = False
from .agraph import AGraph as pyAGraph
from .agraph import force_use_of_python_simplification
from .pytorch_agraph import PytorchAGraph as torchAGraph
from ...chromosomes.generator import Generator
from ...util.argument_validation import argument_validation


class AGraphGenerator(Generator):
    """Generates acyclic graph individuals

    Parameters
    ----------
    agraph_size : int
                  command array size of the generated acyclic graphs
    component_generator : agraph.ComponentGenerator
                          Generator of stack components of agraphs
    """

    @argument_validation(agraph_size={">=": 1})
    def __init__(
        self,
        agraph_size,
        component_generator,
        use_python=False,
        use_simplification=False,
        use_pytorch=False,
    ):
        self.agraph_size = agraph_size
        self.component_generator = component_generator
        self._use_simplification = use_simplification
        if use_python:
            self._backend_generator_function = self._python_generator_function
        elif use_pytorch:
            self._backend_generator_function = self._pytorch_generator_function
        else:
            self._backend_generator_function = self._generator_function

        if use_simplification:
            force_use_of_python_simplification()

    def __call__(self):
        """Generates random agraph individual.

        Fills stack based on random commands from the component generator.

        Returns
        -------
        Agraph
            new random acyclic graph individual
        """
        individual = self._backend_generator_function()
        individual._valid_parameters, individual.command_array = self._create_command_array()
        return individual

    def _python_generator_function(self):
        return pyAGraph(use_simplification=self._use_simplification)

    def _pytorch_generator_function(self):
        return torchAGraph(use_simplification=self._use_simplification)

    def _generator_function(self):
        return AGraph(use_simplification=self._use_simplification)

    def _create_command_array(self):
        command_array = np.empty((self.agraph_size, 3), dtype=int)
        commands_to_generate = self.agraph_size
        i = 0

        valid_parameters = np.array([0])

        while commands_to_generate > 0:

            equ_command, new_command = self.component_generator.random_command_w_eq(
                                                                    i, valid_parameters)
            attempts = 0
            while commands_to_generate < new_command.shape[0]:
                if attempts > 99:
                    raise RuntimeError("Couldn't generate small enough agraph command")
                attempts += 1
                equ_command, new_command = self.component_generator.random_command_w_eq(
                                                                        i, valid_parameters)
            if new_command.ndim == 1:
                new_command = new_command.reshape((1,-1))
            command_array[i : i + new_command.shape[0]] = new_command
            commands_to_generate -= new_command.shape[0]
            i += new_command.shape[0]
            if i > 1:
                valid_parameters = np.append(valid_parameters, i-1)
        return valid_parameters, command_array
