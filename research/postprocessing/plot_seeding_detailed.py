import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.stats as st

base_dir = r"/research/output/seeding_detailed_results_hof"


def get_cleaned_df(df_name, method_name):
    df = pd.read_csv(os.path.join(base_dir, df_name))
    df["method"] = method_name
    df = df.sort_values(by=["sample_i", "gens"])
    return df


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


# scatter plot is probably better for this task
def plot_gens(df, method):
    gens = np.array(sorted(df["gens"].unique()))
    means = []
    errs = []

    for gen in gens:
        fitness_values = df[df["gens"] == gen]["fitness"].values
        if len(fitness_values) > 1:
            mean, err = get_dist_stats(fitness_values)

            if np.isnan(err):
                err = 0.000001
        else:
            mean = 0
            err = 0

        means.append(mean)
        errs.append(err)

    means = np.array(means)
    errs = np.array(errs)

    plt.plot(gens, means, label=method.replace("_", " "))
    plt.fill_between(gens, means - errs, means + errs, alpha=0.3)

plt.rcParams["font.family"] = "calibri"
plt.rcParams["font.size"] = 12

plot_gens(get_cleaned_df("no_seeding.csv", "no_seeding"), "no_seeding")
plot_gens(get_cleaned_df("population.csv", "population"), "population")
plot_gens(get_cleaned_df("features.csv", "feature"), "feature")
plot_gens(get_cleaned_df("mutation.csv", "mutation"), "mutation")

plt.xlabel("generations (lower is better)")
plt.ylabel("fitness (lower is better)")

ax = plt.gca()
ax.set_xlim([0, 3000])
ax.set_ylim([1e-5, 1e3])
plt.yscale("log")
plt.legend()
plt.title("Comparison of Seeding Methods")
plt.savefig(os.path.join("..", "figures", "detailed_seeding_results.svg"))
plt.savefig(os.path.join("..", "figures", "detailed_seeding_results.png"))
plt.show()
