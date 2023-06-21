# all training data is used from https://www.kaggle.com/datasets/uciml/iris
import pandas as pd
from sklearn.model_selection import train_test_split

from grove.trees.n_tree import NTree

DATA_PATH = "./data/500_Person_Gender_Height_Weight_Index/data.csv"
CONFIG_PATH = "./data/500_Person_Gender_Height_Weight_Index/config.csv"

if __name__ == "__main__":
    dataset = pd.read_csv(DATA_PATH)
    config = pd.read_csv(CONFIG_PATH)

    dataset["Overweight"] = dataset["Index"] > 2
    dataset.drop(columns=["Index"], inplace=True)
    dataset["Overweight"].replace({True: 1, False: 0}, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(
        dataset[["Gender", "Height", "Weight", "Overweight"]],
        dataset["Overweight"],
    )

    model = NTree(
        dataset=x_train,
        target=pd.DataFrame(y_train),
        features=["Gender", "Height", "Weight", "Overweight"],
        config=config,
        max_children=3,
        min_samples_per_node=50,
        logging_enabled=True,
    )

    model.build()
    model.print()
