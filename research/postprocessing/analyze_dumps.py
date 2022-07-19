import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

if __name__ == '__main__':
    output_dir = r"C:/Users/dlranda2/Desktop/GPSR/bingo/research/seeding_output/"
    no_seed = pd.read_csv(output_dir + "no_seeding_1/results.csv")
    pop_seed = pd.read_csv(output_dir + "subgraph_seeding_1/results.csv")
    mut_seed = pd.read_csv(output_dir + "subgraph_seeding_mutation_1/results.csv")

    good = 0
    bad = 0
    total = 0
    for dataset_i in range(100):
        no_seed_gens = no_seed[(no_seed["dataset_i"] == dataset_i)]["generations"].values
        mut_seed_gens = mut_seed[(mut_seed["dataset_i"] == dataset_i)]["generations"].values

        if not np.mean(no_seed_gens) == 0 or not np.mean(mut_seed_gens) == 0:
            _, no_greater = ttest_ind(no_seed_gens, mut_seed_gens,
                                          alternative="greater")
            _, no_less = ttest_ind(no_seed_gens, mut_seed_gens,
                                       alternative="less")

            if no_greater < 0.05:
                good += 1
            if no_less < 0.05:
                bad += 1
            total += 1
            # print(f"dataset_i: {dataset_i}, no seeding > pop seeding: {no_greater_pop < 0.05}")
            # print(f"               no seeding < pop seeding: {no_less_pop < 0.05}")

    print("good %:", float(good) / float(total))
    print("bad %:", float(bad) / float(total))

    _, overall_sign = ttest_ind(no_seed["generations"], mut_seed["generations"])
    _, no_greater = ttest_ind(no_seed["generations"], mut_seed["generations"], alternative="greater")
    _, no_less = ttest_ind(no_seed["generations"], mut_seed["generations"], alternative="less")
    print("overall different?", overall_sign < 0.05)
    print("overall no > mut?", no_greater < 0.05)
    print("overall no < mut?", no_less < 0.05)
    # print(f"no seeding > mut seeding: {no_greater_mut < 0.05}")
    # print(f"pop seeding != mut seeding: {pop_diff_mut < 0.05}")
