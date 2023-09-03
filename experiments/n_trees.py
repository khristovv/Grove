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

    # number_of_trees = [10, 20, 50, 100, 200]  # would it work with 500 ?
    # number_of_trees = [10, 20, 40, 80, 160, 320]
    number_of_trees = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    columns = ["Train", "Test", "In-Bag", "OOB"]

    accuracy_change_df = pd.DataFrame(index=number_of_trees, columns=columns)
    precision_change_df = pd.DataFrame(index=number_of_trees, columns=columns)
    recall_change_df = pd.DataFrame(index=number_of_trees, columns=columns)
    f1_score_change_df = pd.DataFrame(index=number_of_trees, columns=columns)

    for n_trees in number_of_trees:
        cut_off = 0.25

        random_forest_regressor = RandomForestRegressor(
            n_trees=n_trees,
            encoding_config=encoding_config,
            # train_in_parallel=False,
            tree_args={
                "max_children": 4,
                "min_samples_per_node": 20,
                "max_depth": 4,
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

        accuracy_change_df.loc[n_trees, "Train"] = test_results_on_train_split.metrics[Metrics.ACCURACY]
        precision_change_df.loc[n_trees, "Train"] = test_results_on_train_split.metrics[Metrics.PRECISION]
        recall_change_df.loc[n_trees, "Train"] = test_results_on_train_split.metrics[Metrics.RECALL]
        f1_score_change_df.loc[n_trees, "Train"] = test_results_on_train_split.metrics[Metrics.F1_SCORE]

        test_results_on_test_split = random_forest_regressor.test(x_test=x_test, y_test=y_test)

        accuracy_change_df.loc[n_trees, "Test"] = test_results_on_test_split.metrics[Metrics.ACCURACY]
        precision_change_df.loc[n_trees, "Test"] = test_results_on_test_split.metrics[Metrics.PRECISION]
        recall_change_df.loc[n_trees, "Test"] = test_results_on_test_split.metrics[Metrics.RECALL]
        f1_score_change_df.loc[n_trees, "Test"] = test_results_on_test_split.metrics[Metrics.F1_SCORE]

        test_results_on_oob, test_results_on_in_bag = random_forest_regressor.oob_test(original_y=y)

        accuracy_change_df.loc[n_trees, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.ACCURACY]
        precision_change_df.loc[n_trees, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.PRECISION]
        recall_change_df.loc[n_trees, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.RECALL]
        f1_score_change_df.loc[n_trees, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.F1_SCORE]

        accuracy_change_df.loc[n_trees, "OOB"] = test_results_on_oob.metrics[Metrics.ACCURACY]
        precision_change_df.loc[n_trees, "OOB"] = test_results_on_oob.metrics[Metrics.PRECISION]
        recall_change_df.loc[n_trees, "OOB"] = test_results_on_oob.metrics[Metrics.RECALL]
        f1_score_change_df.loc[n_trees, "OOB"] = test_results_on_oob.metrics[Metrics.F1_SCORE]

    with Plotter() as plotter:
        plotter.plot_metric_grid(
            metrics_df=accuracy_change_df[["Train", "Test"]],
            title="RF Regressor Accuracy on Train & Test datasets",
            x_label="Number of Trees",
            y_label="Accuracy",
        )
        plotter.plot_metric_grid(
            metrics_df=accuracy_change_df[["In-Bag", "OOB"]],
            title="RF Regressor Accuracy on In-Bag & OOB datasets",
            x_label="Number of Trees",
            y_label="Accuracy",
        )

        plotter.plot_metric_grid(
            metrics_df=precision_change_df[["Train", "Test"]],
            title="RF Regressor Precision on Train & Test datasets",
            x_label="Number of Trees",
            y_label="Precision",
        )
        plotter.plot_metric_grid(
            metrics_df=precision_change_df[["In-Bag", "OOB"]],
            title="RF Regressor Precision on In-Bag & OOB datasets",
            x_label="Number of Trees",
            y_label="Precision",
        )

        plotter.plot_metric_grid(
            metrics_df=recall_change_df[["Train", "Test"]],
            title="RF Regressor Recall on Train & Test datasets",
            x_label="Number of Trees",
            y_label="Recall",
        )
        plotter.plot_metric_grid(
            metrics_df=recall_change_df[["In-Bag", "OOB"]],
            title="RF Regressor Recall on In-Bag & OOB datasets",
            x_label="Number of Trees",
            y_label="Recall",
        )

        plotter.plot_metric_grid(
            metrics_df=f1_score_change_df[["Train", "Test"]],
            title="RF Regressor F1 Score on Train & Test datasets",
            x_label="Number of Trees",
            y_label="F1 Score",
        )
        plotter.plot_metric_grid(
            metrics_df=f1_score_change_df[["In-Bag", "OOB"]],
            title="RF Regressor F1 Score on In-Bag & OOB datasets",
            x_label="Number of Trees",
            y_label="F1 Score",
        )
