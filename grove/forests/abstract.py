import pandas as pd


class AbstractForest:
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that trains the random forest model from the training set (x, y)."""
        raise NotImplementedError

    def test(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that tests the random forest on the test set (x, y)."""
        raise NotImplementedError

    def predict(self):
        """A method that classifies a new data points."""
        raise NotImplementedError
