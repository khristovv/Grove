# all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex
import pandas as pd


import os
import sys


# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", "..")
sys.path.append(grove_dir)

from grove.utils.plotting import Plotter  # noqa
from grove.utils.sampling import Sampler  # noqa
from grove.forests import RandomForestClassifer  # noqa

DATA_PATH = "./data/Classification/Intermediate/data.csv"
CONFIG_PATH = "./data/Classification/Intermediate/config.csv"

if __name__ == "__main__":
    x = pd.read_csv(DATA_PATH, index_col="UDI")
    y = x["Machine failure"]
    x.drop("Machine failure", axis=1, inplace=True)
    encoding_config = pd.read_csv(CONFIG_PATH)

    x_train, y_train, x_test, y_test = Sampler().get_y_proportional_train_test_split(x=x, y=y, seed=1)

    random_forest_model = RandomForestClassifer(
        n_trees=10,
        encoding_config=encoding_config,
        # train_in_parallel=False,
        tree_args={
            "y_dtype": "bin",
            "max_children": 4,
            "min_samples_per_node": 10,
            "max_depth": 6,
            "statistics_enabled": True,
            "consecutive_splits_on_same_feature_enabled": False,
        },
        m_split=3,
        n_bag=1_500,
        seed=1,
        # auto_split=True,
        oob_score_enabled=True,
        min_number_of_classes=200,
    )

    random_forest_model.train(x=x_train, y=y_train)

    actual_column = f"ACTUAL_{y.name}"
    predicted_column = f"PREDICTED_{y.name}"

    with Plotter() as plotter:
        test_results_on_train_split = random_forest_model.test(
            x_test=x_train,
            y_test=y_train,
            save_results=True,
            output_dir="test_results_RF_classification",
            labeled_data_filename="labeled_data(train).csv",
            score_filename="score(train).csv",
        )
        labeled_data = test_results_on_train_split.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Confusion Matrix on Train Split",
        )

        test_results_on_test_split = random_forest_model.test(
            x_test=x_test,
            y_test=y_test,
            save_results=True,
            output_dir="test_results_RF_classification",
            labeled_data_filename="labeled_data(test).csv",
            score_filename="score(test).csv",
        )
        labeled_data = test_results_on_test_split.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Confusion Matrix on Test Split",
        )

        test_results_on_oob = random_forest_model.oob_test(
            original_y=y,
            save_results=True,
            output_dir="test_results_RF_classification",
            labeled_data_filename="labeled_data(OOB).csv",
            score_filename="score(OOB).csv",
        )
        labeled_data = test_results_on_oob.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Confusion Matrix Out-of-Bag matrix",
        )
