import json
import os
import pandas as pd
import re

from sklearn.model_selection import train_test_split

from bingo.evolutionary_optimizers.parallel_archipelago import \
    load_parallel_archipelago_from_file as load_archipelago
from bingo.symbolic_regression import ExplicitTrainingData, ExplicitRegression

checkpt_pattern = re.compile(r"checkpoint_(\d+)")

# TODO CLO on test and train fit


def get_most_recent_checkpoint(sample_dir):
    largest_checkpoint_n = -1
    largest_checkpoint_name = None

    for file_name in os.listdir(sample_dir):
        match = checkpt_pattern.match(file_name)
        if match:
            checkpoint_n = int(match.group(1))
            if checkpoint_n > largest_checkpoint_n:
                largest_checkpoint_n = checkpoint_n
                largest_checkpoint_name = file_name

    return largest_checkpoint_name


def get_sample_results_dict(sample_dir, sample_i, dataset_i, dataset_row):
    results = dict()
    results["dataset_i"] = dataset_i
    results["sample_i"] = sample_i

    checkpoint_name = get_most_recent_checkpoint(sample_dir)
    archipelago = load_archipelago(sample_dir + "/" + checkpoint_name)

    results["generations"] = archipelago.generational_age
    results["true_eq_complexity"] = dataset_row["true_eq"].get_complexity()

    with open(sample_dir + "/hyperparams.json", "r") as f:
        hyperparams = json.load(f)

    train_X, test_X, train_y, test_y = \
        train_test_split(dataset_row["true_X"], dataset_row["true_y"],
                         train_size=hyperparams["TRAIN_PERCENT"],
                         random_state=hyperparams["train_test_split_seed"])

    train_data = ExplicitTrainingData(train_X, train_y)
    train_fitness = ExplicitRegression(train_data)

    test_data = ExplicitTrainingData(test_X, test_y)
    test_fitness = ExplicitRegression(test_data)

    try:
        best_ind = archipelago.hall_of_fame[0]
    except IndexError:
        best_ind = archipelago.get_best_individual()

    end_train_fitness = train_fitness(best_ind)
    converged = end_train_fitness <= hyperparams["FITNESS_THRESHOLD"]

    results["converged"] = converged
    results["end_train_fitness"] = end_train_fitness
    results["end_test_fitness"] = test_fitness(best_ind)

    approx_eq = dataset_row["approx_eq"]
    results["approx_eq_train_err"] = train_fitness(approx_eq)
    results["approx_eq_test_err"] = test_fitness(approx_eq)
    return results


def get_dataset_info(dataset_dir, dataset_i, dataset_row):
    rows = []

    sample_dir_i = 0
    for sample_dir in os.listdir(dataset_dir):
        sample_dir = dataset_dir + "/" + sample_dir
        if os.path.isdir(sample_dir):
            rows.append(get_sample_results_dict(sample_dir,
                                                sample_dir_i,
                                                dataset_i,
                                                dataset_row))
            sample_dir_i += 1

    return rows


if __name__ == '__main__':
    output_dir = r"../seeding_output/no_seeding_1"
    data_path = r"../seeding_output/data/1000_points_100_eq_16_stack.csv"

    if not os.path.exists(output_dir) or not os.path.exists(data_path):
        raise RuntimeError("output or data dir doesn't exist")

    data_df = pd.read_csv(data_path)

    run_info = []

    dataset_i = 0
    for dataset_dir in os.listdir(output_dir):
        dataset_dir = output_dir + "/" + dataset_dir
        if os.path.isdir(dataset_dir):
            dataset_row = data_df.iloc[dataset_i]
            run_info.extend(get_dataset_info(dataset_dir,
                                             dataset_i,
                                             dataset_row))
            dataset_i += 1

    results_df = pd.DataFrame(run_info)
    print(results_df)
    results_df.to_csv(output_dir + "/results.csv")
