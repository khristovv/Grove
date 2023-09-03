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
    columns = ["Train", "Test", "In-Bag", "OOB"]

    accuracy_change_df = pd.DataFrame(index=max_depths, columns=columns)
    precision_change_df = pd.DataFrame(index=max_depths, columns=columns)
    recall_change_df = pd.DataFrame(index=max_depths, columns=columns)
    f1_score_change_df = pd.DataFrame(index=max_depths, columns=columns)

    for max_depth in max_depths:
        cut_off = 0.25

        random_forest_regressor = RandomForestRegressor(
            n_trees=20,
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

        accuracy_change_df.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.ACCURACY]
        precision_change_df.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.PRECISION]
        recall_change_df.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.RECALL]
        f1_score_change_df.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.F1_SCORE]

        test_results_on_test_split = random_forest_regressor.test(x_test=x_test, y_test=y_test)

        accuracy_change_df.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.ACCURACY]
        precision_change_df.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.PRECISION]
        recall_change_df.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.RECALL]
        f1_score_change_df.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.F1_SCORE]

        test_results_on_oob, test_results_on_in_bag = random_forest_regressor.oob_test(original_y=y)

        accuracy_change_df.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.ACCURACY]
        precision_change_df.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.PRECISION]
        recall_change_df.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.RECALL]
        f1_score_change_df.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.F1_SCORE]

        accuracy_change_df.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.ACCURACY]
        precision_change_df.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.PRECISION]
        recall_change_df.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.RECALL]
        f1_score_change_df.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.F1_SCORE]

    with Plotter() as plotter:
        plotter.plot_metric(
            title="RF Regressor Accuracy on Train & Test datasets",
            x_label="Tree Depth",
            y_label="Accuracy",
            metrics=[accuracy_change_df["Train"], accuracy_change_df["Test"]],
        )
        plotter.plot_metric(
            title="RF Regressor Accuracy on In-Bag & OOB datasets",
            x_label="Tree Depth",
            y_label="Accuracy",
            metrics=[accuracy_change_df["In-Bag"], accuracy_change_df["OOB"]],
        )

        plotter.plot_metric(
            title="RF Regressor Precision on Train & Test datasets",
            x_label="Tree Depth",
            y_label="Precision",
            metrics=[precision_change_df["Train"], precision_change_df["Test"]],
        )
        plotter.plot_metric(
            title="RF Regressor Precision on In-Bag & OOB datasets",
            x_label="Tree Depth",
            y_label="Precision",
            metrics=[precision_change_df["In-Bag"], precision_change_df["OOB"]],
        )

        plotter.plot_metric(
            title="RF Regressor Recall on Train & Test datasets",
            x_label="Tree Depth",
            y_label="Recall",
            metrics=[recall_change_df["Train"], recall_change_df["Test"]],
        )
        plotter.plot_metric(
            title="RF Regressor Recall on In-Bag & OOB datasets",
            x_label="Tree Depth",
            y_label="Recall",
            metrics=[recall_change_df["In-Bag"], recall_change_df["OOB"]],
        )

        plotter.plot_metric(
            title="RF Regressor F1 Score on Train & Test datasets",
            x_label="Tree Depth",
            y_label="F1 Score",
            metrics=[f1_score_change_df["Train"], f1_score_change_df["Test"]],
        )
        plotter.plot_metric(
            title="RF Regressor F1 Score on In-Bag & OOB datasets",
            x_label="Tree Depth",
            y_label="F1 Score",
            metrics=[f1_score_change_df["In-Bag"], f1_score_change_df["OOB"]],
        )
