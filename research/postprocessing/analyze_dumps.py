import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def compare_distributions(first, second, stat, alpha=0.05):
    first_better = 0
    second_better = 0
    total = 0
    for dataset_i in range(100):
        first_stat = first[(first["dataset_i"] == dataset_i)][stat].values
        second_stat = second[(second["dataset_i"] == dataset_i)][stat].values

        if not np.mean(first_stat) == 0 or not np.mean(second_stat) == 0:
            _, first_greater = ttest_ind(first_stat, second_stat, alternative="greater")
            _, first_less = ttest_ind(first_stat, second_stat, alternative="less")

            if first_greater < alpha:
                first_better += 1
            if first_less < alpha:
                second_better += 1
            total += 1

    print("first better (significant) %:", float(first_better) / float(total))
    print("second better (significant) %:", float(second_better) / float(total))

    _, overall_sign = ttest_ind(first[stat], second[stat])
    _, first_greater = ttest_ind(first[stat], second[stat], alternative="greater")
    _, first_less = ttest_ind(first[stat], second[stat], alternative="less")
    print("overall different?", overall_sign < alpha)
    print("overall first > second?", first_greater < alpha)
    print("overall first < second?", first_less < alpha)


if __name__ == '__main__':
    output_dir = r"C:/Users/David/Desktop/GPSR Research/bingoNASAFork/research/seeding_output/"
    no_seed = pd.read_csv(output_dir + "no_seeding_results.csv")
    pop_seed = pd.read_csv(output_dir + "pop_seeding_results.csv")
    mut_seed = pd.read_csv(output_dir + "mut_seeding_results.csv")

    print(mut_seed.columns)
    # 'dataset_i', 'sample_i', 'generations', 'true_eq_complexity',
    # 'converged', 'end_train_fitness', 'end_test_fitness',
    # 'approx_eq_train_err', 'approx_eq_test_err'
    
    compare_distributions(no_seed, mut_seed, "end_test_fitness")
