# all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex
import pandas as pd
from sklearn.model_selection import train_test_split

from grove.trees.n_tree import NTree

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
        x=x_train,
        y=pd.DataFrame(y_train),
        config=config,
        max_children=3,
        min_samples_per_node=50,
        # criterion_threshold=10.0,
        max_depth=5,
        logging_enabled=True,
        statistics_enabled=True,
    )

    print("============================== Bulding Tree ==============================")
    tree_model.build()
    print(tree_model)
    print("============================== Statistics ==============================")
    print(tree_model.get_statistics())
    print("============================== Classifying ==============================")
    labeled_data = tree_model.classify(x_test)
    print(labeled_data)
