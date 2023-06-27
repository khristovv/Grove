# all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex
import pandas as pd
from sklearn.model_selection import train_test_split

from grove.trees import NTree

X_PATH = "./data/500_Person_Gender_Height_Weight_Index/data_x.csv"
Y_PATH = "./data/500_Person_Gender_Height_Weight_Index/data_y.csv"
CONFIG_PATH = "./data/500_Person_Gender_Height_Weight_Index/config.csv"

if __name__ == "__main__":
    x = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)
    config = pd.read_csv(CONFIG_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        x[["Gender", "Height", "Weight"]],
        y,
    )

    tree_model = NTree(
        encoding_config=config,
        max_children=3,
        min_samples_per_node=50,
        max_depth=5,
        # criterion_threshold=10.0,
        logging_enabled=True,
        statistics_enabled=True,
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
    test_results = tree_model.test(x=x_test, y=pd.DataFrame(y_test), save_results=True)
    print(test_results)
    print("============================== Done ==============================\n")
