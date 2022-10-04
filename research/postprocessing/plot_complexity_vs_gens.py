import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


def get_dist_stats(dist):
    mean = np.mean(dist)
    std_err = st.sem(dist)
    interval = st.t.interval(alpha=0.95, df=len(dist) - 1, loc=mean,
                             scale=std_err)
    if np.array_equal(interval, (np.nan, np.nan)):
        half_interval = 0
    else:
        half_interval = interval[1] - mean

    return mean, half_interval


def get_complexity_stats(dataframe, complexities, stat):
    means = []
    errs = []
    for complexity in complexities:
        dist = dataframe[dataframe["true_eq_complexity"] == complexity][stat]
        mean, err = get_dist_stats(dist)
        means.append(mean)
        errs.append(err)
    return means, errs


if __name__ == '__main__':
    output_dir = r"C:/Users/dlranda2/Desktop/GPSR/bingo/research/seeding_output_rerun/"
    plt.rcParams["font.family"] = "calibri"
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.figsize"] = (10, 5)

    width = 0.2
    n_datasets = 4

    normal = pd.read_csv(output_dir + "no_seeding.csv")
    mutation = pd.read_csv(output_dir + "mutation.csv")
    feature = pd.read_csv(output_dir + "features.csv")
    population = pd.read_csv(output_dir + "population.csv")

    complexities = np.array(sorted(normal["true_eq_complexity"].unique()))

    normal_means, normal_errs = get_complexity_stats(normal, complexities, "generations")
    mutation_means, mutation_errs = get_complexity_stats(mutation, complexities, "generations")
    feature_means, feature_errs = get_complexity_stats(feature, complexities, "generations")
    population_means, population_errs = get_complexity_stats(population, complexities, "generations")

    plt.bar(complexities, normal_means, width, label="no_seeding", yerr=normal_errs)
    plt.bar(complexities + 1 * width, population_means, width, label="population", yerr=population_errs)
    plt.bar(complexities + 2 * width, feature_means, width, label="feature", yerr=feature_errs)
    plt.bar(complexities + 3 * width, mutation_means, width, label="mutation", yerr=mutation_errs)

    plt.xticks(complexities + 1.5 * width, complexities)
    plt.xlabel("complexity (higher = more difficult)")
    plt.ylabel("generations (lower = better)")
    plt.title("Comparison of Seeding Methods")
    plt.legend(loc="upper left")
    plt.savefig("../figures/seeding_comparison.svg")
    plt.savefig("../figures/seeding_comparison.png")
    plt.show()
