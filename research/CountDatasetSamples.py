import os
import re
import pandas as pd

if __name__ == "__main__":
    #output_dir = r"/u/dlranda2/gpsr/output/subgraph_seeding_1"
    output_dir = r"/u/dlranda2/gpsr/output/subgraph_seeding_mutation"

    data = r"/u/dlranda2/gpsr/data/1000_points_100_eq_16_stack.pkl"
    df = pd.read_pickle(data)

    checkpoint_pattern = r"checkpoint_(\d+)\.pkl"
    datadir_pattern = "dataset_(\d+)"

    for subdir in os.listdir(output_dir):
        if not os.path.isfile(subdir):
            for sample_dir in os.listdir(os.path.join(output_dir, subdir)):
                sample_checkpoints = []
                for checkpoint in os.listdir(os.path.join(output_dir, subdir, sample_dir)):
                    match = re.fullmatch(checkpoint_pattern, checkpoint)
                    if match:
                        sample_checkpoints.append(int(match.group(1)))

                if max(sample_checkpoints) > 0:
                    subdir_i = int(re.fullmatch(datadir_pattern, subdir).group(1))
                    print(subdir, df.iloc[subdir_i]["true_eq"])
                    break
