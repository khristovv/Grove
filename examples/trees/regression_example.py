# training data is used from https://www.kaggle.com/competitions/playground-series-s3e16/data?select=test.csv
import pandas as pd

import os
import sys


# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", "..")
sys.path.append(grove_dir)

from grove.trees import RegressionTree  # noqa
from grove.utils.sampling import Sampler  # noqa

DATA_PATH = "./data/Regression/data.csv"
CONFIG_PATH = "./data/Regression/config.csv"

if __name__ == "__main__":
    x = pd.read_csv(DATA_PATH, index_col="id")
    config = pd.read_csv(CONFIG_PATH)

    # get a subset of the data for faster development
    x = x.sample(n=20_000, replace=False, random_state=1)

    y = x["Age"]
    x = x.drop(columns=["Age"])

    x_train, y_train, x_test, y_test = Sampler().get_train_test_split(x=x, y=y, random_state=1)

    tree_model = RegressionTree(
        encoding_config=config,
        max_children=5,
        min_samples_per_node=500,
        max_depth=6,
        # criterion_threshold=10.0,
        logging_enabled=True,
        statistics_enabled=True,
        consecutive_splits_on_same_feature_enabled=False,
    )

    tree_model.train(x=x_train, y=y_train)
    test_results = tree_model.test(
        x=x_test,
        y=y_test,
        save_results=True,
        output_dir="test_results_DT_regression",
    )
