import numpy as np
import pandas as pd
import re
import sympy

from itertools import count

from research.RandomEqGeneration import get_utilized_idx
from bingo.symbolic_regression.agraph.agraph import AGraph


def replace_float_with_constant_str(eq_str):
    counter = count(0)
    new_eq_str = re.sub(r"1.0", lambda x: f"C_{next(counter)}", eq_str)
    return new_eq_str


def get_agraph_from_cmd_arr(cmd_arr):
    agraph = AGraph()
    agraph.command_array = cmd_arr
    return agraph


class SubgraphSeedGenerator:
    @staticmethod
    def get_seed_eqs(original_cmd_arr):
        sub_tree_idx = get_utilized_idx(original_cmd_arr) - \
                       {len(original_cmd_arr) - 1}
        seeds = []
        for index in sub_tree_idx:
            new_agraph_cmd_arr = np.copy(original_cmd_arr)[:index + 1]
            seeds.append(get_agraph_from_cmd_arr(new_agraph_cmd_arr))
        return seeds

    @staticmethod
    def get_seed_strs(original_cmd_arr):
        seed_eqs = SubgraphSeedGenerator.get_seed_eqs(original_cmd_arr)
        eq_strs = []
        for seed_eq in seed_eqs:
            eq_str = seed_eq.get_formatted_string("sympy")
            eq_strs.append(replace_float_with_constant_str(eq_str))
        return set(eq_strs)


if __name__ == '__main__':
    eq = AGraph(equation="(X_0 - C_0)^2 + (X_1 - C_1)^2 - 1")
    X_0, X_1, X_2 = sympy.symbols("X_0 X_1 X_2")
    string = eq.get_formatted_string(format_="sympy")
    exp = sympy.simplify(string)
    expanded_exp = sympy.expand(exp)
    import pdb;pdb.set_trace()
    str(eq)
    print(eq)
    print(eq.command_array)
    print(SubgraphSeedGenerator.get_seed_strs(eq.command_array))
