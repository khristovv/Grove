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

    random_forest_model = RandomForestRegressor(
        n_trees=10,
        encoding_config=encoding_config,
        allowed_diff=3.0,  # 3 months allowed difference
        # train_in_parallel=False,
        tree_args={
            "max_children": 4,
            "min_samples_per_node": 100,
            "max_depth": 5,
            # "consecutive_splits_on_same_feature_enabled": False,
        },
        auto_split=True,
        m_split=4,
        n_bag=5_000,
        seed=1,
    )

    random_forest_model.train(x=x, y=y)
    random_forest_model.test(
        save_results=True,
        output_dir="test_results_RF_regression",
    )
