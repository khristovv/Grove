# all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex
import pandas as pd


import os
import sys


# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", "..")
sys.path.append(grove_dir)

from grove.forests import RandomForestClassifer  # noqa

DATA_PATH = "./data/Classification/Intermediate/data.csv"
CONFIG_PATH = "./data/Classification/Intermediate/config.csv"

if __name__ == "__main__":
    x = pd.read_csv(DATA_PATH, index_col="UDI")
    y = x["Machine failure"]
    x.drop("Machine failure", axis=1, inplace=True)
    encoding_config = pd.read_csv(CONFIG_PATH)

    x_train, y_train = x, y

    y_dtype = "bin"

    random_forest_model = RandomForestClassifer(
        n_trees=10,
        encoding_config=encoding_config,
        # train_in_parallel=False,
        tree_args={
            "y_dtype": y_dtype,
            "max_children": 4,
            "min_samples_per_node": 10,
            "max_depth": 6,
            "statistics_enabled": True,
            "consecutive_splits_on_same_feature_enabled": False,
        },
        m_split=3,
        n_bag=1_500,
        seed=1,
        auto_split=True,
        oob_score_enabled=True,
        min_number_of_classes=200,
    )

    random_forest_model.train(x=x_train, y=y_train)
    random_forest_model.test(
        save_results=True,
        output_dir="test_results_RF_classification",
    )
    confusion_matrix = random_forest_model.build_confusion_matrix()
    random_forest_model.plot_confusion_matrix(confusion_matrix=confusion_matrix)
