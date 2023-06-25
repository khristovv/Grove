# all training data is used from https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex
import pandas as pd
from sklearn.model_selection import train_test_split

from grove.trees.n_tree import NTree

DATA_SAMPLES_PATH = "./data/500_Person_Gender_Height_Weight_Index/data_x.csv"
TARGET_VARIABLE_PATH = "./data/500_Person_Gender_Height_Weight_Index/data_y.csv"
CONFIG_PATH = "./data/500_Person_Gender_Height_Weight_Index/config.csv"

if __name__ == "__main__":
    dataset = pd.read_csv(DATA_SAMPLES_PATH)
    class_labels = pd.read_csv(TARGET_VARIABLE_PATH)
    config = pd.read_csv(CONFIG_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        dataset[["Gender", "Height", "Weight"]],
        class_labels,
    )

    tree_model = NTree(
        dataset=x_train,
        target=pd.DataFrame(y_train),
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
