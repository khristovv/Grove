# all training data is used from https://www.kaggle.com/c/titanic/data
import pandas as pd

from grove import DecisionTree

IRIS_DATA_PATH = "./data/Iris.csv"

if __name__ == "__main__":
    dataset = pd.read_csv(IRIS_DATA_PATH)

    model = DecisionTree(
        dataset=dataset,
        features=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        target="Species",
    )

    model.train()
    # model.build()

    model.view()
