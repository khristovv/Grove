# all training data is used from https://www.kaggle.com/datasets/uciml/iris
import pandas as pd
from sklearn.model_selection import train_test_split

from grove import BinaryTree

IRIS_DATA_PATH = "./data/Iris.csv"

if __name__ == "__main__":
    dataset = pd.read_csv(IRIS_DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        dataset[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]],
        dataset["Species"],
    )

    model = BinaryTree(
        dataset=dataset,
        features=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        target="Species",
    )

    model.build()

    model.print()
