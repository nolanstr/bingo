import dill
import os
import re
import sys

import pandas as pd

from bingo.evolutionary_optimizers.parallel_archipelago import load_parallel_archipelago_from_file


def get_ea_diag_from_pickle_path(pickle_path):
    with open(pickle_path, "rb") as f:
        archs = dill.load(f)

    ea_dias = [arch.island.get_ea_diagnostic_info() for arch in archs]
    for ea_dia in ea_dias:
        ea_dia._crossover_types = []
        ea_dia._mutation_types = []
        ea_dia._crossover_type_stats = {}
        ea_dia._mutation_type_stats = {}
        ea_dia._crossover_mutation_type_stats = {}
    return sum(ea_dias)


if __name__ == "__main__":
    checkpoint_pattern = re.compile(r"checkpoint_(\d+)")
    sample_path = sys.argv[1]

    rows = []

    for checkpoint_file in os.listdir(sample_path):
        match = checkpoint_pattern.match(checkpoint_file)
        if match:
            checkpoint_gen = match.group(1)
            checkpoint_path = os.path.join(sample_path, checkpoint_file)
            ea_diag = get_ea_diag_from_pickle_path(checkpoint_path)
            row = [checkpoint_gen] + list(ea_diag.summary)
            rows.append(row)

    rows = sorted(rows, key=lambda row: row[0])
    df = pd.DataFrame(rows, columns=["pickle_gen", "bene_cross", "detr_cross", "bene_mut", "detr_mut", "bene_cm", "detr_cm"])

    df_filename = "ea_stats.csv"
    df.to_csv(os.path.join(sample_path, df_filename))

