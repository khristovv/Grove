import pandas as pd


class RandomForestInterface:
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that trains the random forest model from the training set (x, y)."""
        raise NotImplementedError

    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame, *args, **kwrags):
        """A method that tests the random forest on the test set (x_test, y_test)."""
        raise NotImplementedError

    def predict(self, x: pd.DataFrame, y_label: str):
        """A method that classifies a new data points."""
        raise NotImplementedError
