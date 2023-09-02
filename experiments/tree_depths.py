# training data is used from https://www.kaggle.com/competitions/playground-series-s3e16/data?select=test.csv
import pandas as pd

import os
import sys


# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", ".")
sys.path.append(grove_dir)

from grove.constants import Metrics  # noqa
from grove.forests import RandomForestRegressor  # noqa
from grove.utils.sampling import Sampler  # noqa
from grove.utils.plotting import Plotter  # noqa

DATA_PATH = "./data/Classification/Intermediate/data.csv"
CONFIG_PATH = "./data/Classification/Intermediate/config.csv"

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH, index_col="UDI")
    y = data["Machine failure"]
    x = data.drop("Machine failure", axis=1)
    encoding_config = pd.read_csv(CONFIG_PATH)

    seed = 1
    x_train, y_train, x_test, y_test = Sampler().get_y_proportional_train_test_split(x=x, y=y, seed=seed)

    actual_column = f"ACTUAL_{y.name}"
    predicted_column = f"PREDICTED_{y.name}"

    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    train_metrics_df = pd.DataFrame(index=max_depths, columns=Metrics.CLASSIFICATION)
    test_metrics_df = pd.DataFrame(index=max_depths, columns=Metrics.CLASSIFICATION)
    oob_metrics_df = pd.DataFrame(index=max_depths, columns=Metrics.CLASSIFICATION)
    in_bag_metrics_df = pd.DataFrame(index=max_depths, columns=Metrics.CLASSIFICATION)

    for max_depth in max_depths:
        cut_off = 0.25

        random_forest_regressor = RandomForestRegressor(
            n_trees=10,
            encoding_config=encoding_config,
            # train_in_parallel=False,
            tree_args={
                "max_children": 4,
                "min_samples_per_node": 20,
                "max_depth": max_depth,
                "logging_enabled": False,
                "statistics_enabled": False,
                "consecutive_splits_on_same_feature_enabled": False,
            },
            cut_off=cut_off,
            m_split=4,
            n_bag=8000,
            seed=seed,
            logging_enabled=True,
            oob_score_enabled=True,
            test_on_in_bag_samples_enabled=True,
            min_number_of_classes=200,
        )
        random_forest_regressor.train(x=x_train, y=y_train)

        test_results_on_train_split = random_forest_regressor.test(x_test=x_train, y_test=y_train)

        train_metrics_df.loc[max_depth] = {
            label: value
            for label, value in test_results_on_train_split.metrics.items()
            if label in Metrics.CLASSIFICATION
        }

        test_results_on_test_split = random_forest_regressor.test(x_test=x_test, y_test=y_test)

        test_metrics_df.loc[max_depth] = {
            label: value
            for label, value in test_results_on_test_split.metrics.items()
            if label in Metrics.CLASSIFICATION
        }

        test_results_on_oob, test_results_on_in_bag = random_forest_regressor.oob_test(original_y=y)

        oob_metrics_df.loc[max_depth] = {
            label: value for label, value in test_results_on_oob.metrics.items() if label in Metrics.CLASSIFICATION
        }
        in_bag_metrics_df.loc[max_depth] = {
            label: value for label, value in test_results_on_in_bag.metrics.items() if label in Metrics.CLASSIFICATION
        }

    with Plotter() as plotter:
        plotter.plot_metric_grid(
            metrics_df=train_metrics_df,
            title="RF Regressor Metrics on Train Dataset",
            x_label="Maximum depth of each tree",
            y_label="Metric",
        )
        plotter.plot_metric_grid(
            metrics_df=test_metrics_df,
            title="RF Regressor Metrics on Test Dataset",
            x_label="Maximum depth of each tree",
            y_label="Metric",
        )
        plotter.plot_metric_grid(
            metrics_df=oob_metrics_df,
            title="RF Regressor Metrics on OOB Dataset",
            x_label="Maximum depth of each tree",
            y_label="Metric",
        )
        plotter.plot_metric_grid(
            metrics_df=in_bag_metrics_df,
            title="RF Regressor Metrics on In-Bag Dataset",
            x_label="Maximum depth of each tree",
            y_label="Metric",
        )
