# all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex
import pandas as pd

import os
import sys


# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", "..")
sys.path.append(grove_dir)

from grove.forests import RandomForestRegressor  # noqa
from grove.utils.sampling import Sampler  # noqa

DATA_PATH = "./data/Regression/data.csv"
CONFIG_PATH = "./data/Regression/config.csv"

if __name__ == "__main__":
    x = pd.read_csv(DATA_PATH, index_col="id")
    y = x["Age"]
    x.drop("Age", axis=1, inplace=True)
    encoding_config = pd.read_csv(CONFIG_PATH)

    x_train, y_train, x_test, y_test = Sampler().get_y_proportional_train_test_split(x=x, y=y, seed=1)

    random_forest_model = RandomForestRegressor(
        n_trees=10,
        encoding_config=encoding_config,
        # train_in_parallel=False,
        tree_args={
            "max_children": 5,
            "min_samples_per_node": 100,
            "max_depth": 4,
            "consecutive_splits_on_same_feature_enabled": False,
        },
        m_split=4,
        n_bag=5_000,
        seed=1,
    )

    random_forest_model.train(x=x_train, y=y_train)
    random_forest_model.test(
        x_test=x_test,
        y_test=y_test,
        save_results=True,
        output_dir="test_results_RF_regression",
    )
