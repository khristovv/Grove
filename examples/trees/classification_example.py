import argparse

import pandas as pd

import os
import sys


# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", "..")
sys.path.append(grove_dir)


from grove.trees import ClassificationTree  # noqa
from grove.utils.sampling import Sampler  # noqa


def load_simple_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    # all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex  # noqa

    X_PATH = "./data/Classification/Simple/data.csv"
    CONFIG_PATH = "./data/Classification/Simple/config.csv"

    data = pd.read_csv(X_PATH)
    y = pd.Series(name="Overweight", data=(data["Index"] >= 3).replace({True: 1, False: 0}))
    x = data.drop("Index", axis=1)
    config = pd.read_csv(CONFIG_PATH)

    y_dtype = "bin"

    tree_model = ClassificationTree(
        encoding_config=config,
        y_dtype=y_dtype,
        max_children=3,
        min_samples_per_node=20,
        max_depth=7,
        # criterion_threshold=10.0,
        logging_enabled=True,
        statistics_enabled=True,
        # consecutive_splits_on_same_feature_enabled=False,
    )

    x_train, y_train, x_test, y_test = Sampler().get_y_proportional_train_test_split(x=x, y=y, random_state=1)

    tree_model.train(x=x_train, y=y_train)
    tree_model.test(
        x=x_test,
        y=y_test,
        save_results=True,
        output_dir="test_results_DT_classification",
    )


def load_intermediate_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    # all training data is used from https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020  # noqa
    DATA_PATH = "./data/Classification/Intermediate/data.csv"
    CONFIG_PATH = "./data/Classification/Intermediate/config.csv"

    data = pd.read_csv(DATA_PATH, index_col="UDI")
    y = data["Machine failure"]
    x = data.drop("Machine failure", axis=1)
    config = pd.read_csv(CONFIG_PATH)

    y_dtype = "bin"

    tree_model = ClassificationTree(
        encoding_config=config,
        y_dtype=y_dtype,
        max_children=4,
        min_samples_per_node=10,
        max_depth=10,
        # criterion_threshold=10.0,
        logging_enabled=True,
        statistics_enabled=True,
        # consecutive_splits_on_same_feature_enabled=False,
    )

    x_train, y_train, x_test, y_test = Sampler().get_y_proportional_train_test_split(x=x, y=y, random_state=1)

    tree_model.train(x=x_train, y=y_train)
    tree_model.test(x=x_test, y=y_test, save_results=True, output_dir="test_results_DT_classification")


SIMPLE = "s"
INTERMEDIATE = "i"

OPTIONS = [SIMPLE, INTERMEDIATE]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classification examples")
    parser.add_argument(
        "--dataset",
        "-d",
        dest="dataset",
        default=SIMPLE,
        help="The dataset to use for the example. Options are: " + ", ".join(OPTIONS),
    )

    args = parser.parse_args()

    if args.dataset == SIMPLE:
        load_simple_dataset()
    elif args.dataset == INTERMEDIATE:
        load_intermediate_dataset()
    else:
        load_simple_dataset()
