# training data is used from https://www.kaggle.com/competitions/playground-series-s3e16/data?select=test.csv
import pandas as pd
from sklearn.model_selection import train_test_split

from grove.trees import RegressionTree

DATA_PATH = "./data/Regression/data.csv"
CONFIG_PATH = "./data/Regression/config.csv"

if __name__ == "__main__":
    x = pd.read_csv(DATA_PATH, index_col="id")
    config = pd.read_csv(CONFIG_PATH)

    # get a subset of the data for faster development
    x = x.sample(n=5000, replace=False, random_state=1)

    y = x["Age"]
    x = x.drop(columns=["Age"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    tree_model = RegressionTree(
        encoding_config=config,
        max_children=3,
        min_samples_per_node=100,
        allowed_diff=3.0,  # 3 months allowed difference
        max_depth=5,
        # criterion_threshold=10.0,
        logging_enabled=True,
        statistics_enabled=True,
        # consecutive_splits_on_same_feature_enabled=False,
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
        output_dir="test_results_regression",
    )
    print(test_results)

    print("============================== Done ==============================\n")
