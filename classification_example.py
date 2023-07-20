# all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from grove.trees import ClassificationTree


def load_simple_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    # DATA_PATH = "./data/Classification/Simple/data.csv"
    X_PATH = "./data/Classification/Simple/data_x.csv"
    Y_PATH = "./data/Classification/Simple/data_y.csv"
    CONFIG_PATH = "./data/Classification/Simple/config.csv"

    x = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)
    config = pd.read_csv(CONFIG_PATH)

    # y = x["Index"]
    # x.drop("Index", axis=1, inplace=True)

    y_dtype = "bin"

    return x, y, config, y_dtype


def load_intermediate_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    DATA_PATH = "./data/Classification/Intermediate/data.csv"
    CONFIG_PATH = "./data/Classification/Intermediate/config.csv"

    x = pd.read_csv(DATA_PATH, index_col="UDI")
    y = x["Machine failure"]
    x.drop("Machine failure", axis=1, inplace=True)
    config = pd.read_csv(CONFIG_PATH)

    y_dtype = "bin"

    return x, y, config, y_dtype


def main(x: pd.DataFrame, y: pd.Series, config: pd.DataFrame, y_dtype: str) -> None:
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    tree_model = ClassificationTree(
        encoding_config=config,
        y_dtype=y_dtype,
        max_children=3,
        min_samples_per_node=50,
        max_depth=5,
        # criterion_threshold=10.0,
        logging_enabled=True,
        statistics_enabled=True,
        # consecutive_splits_on_same_feature_enabled=False
    )

    print("============================== Start ==============================")
    print("============================== Training Model ==============================\n")

    tree_model.train(
        x=x_train,
        y=pd.DataFrame(y_train),
    )
    print(tree_model)

    print("============================== Training Complete ==============================\n")
    print("============================== Statistics ==============================\n")

    print(tree_model.get_statistics())

    print("\n============================== Test Results ==============================\n")

    test_results = tree_model.test(
        x=x_test,
        y=pd.DataFrame(y_test),
        save_results=True,
        output_dir="test_results_classification",
    )
    print(test_results)
    print("============================== Done ==============================\n")


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
        x, y, config, y_dtype = load_simple_dataset()
    elif args.dataset == INTERMEDIATE:
        x, y, config, y_dtype = load_intermediate_dataset()
    else:
        x, y, config, y_dtype = load_simple_dataset()

    main(x=x, y=y, config=config, y_dtype=y_dtype)
