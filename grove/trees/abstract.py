import pandas as pd


class AbstractTree:
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that trains a decision tree model from the training set (x, y)."""
        raise NotImplementedError

    def test(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that tests the decision tree model on the test set (x, y)."""
        raise NotImplementedError

    def classify(self, data: pd.DataFrame, y_label: str):
        """A method that classifies new data using the trained decision tree model."""
        raise NotImplementedError
