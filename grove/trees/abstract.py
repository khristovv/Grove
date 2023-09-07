import pandas as pd


class TreeInterface:
    def train(self, x: pd.DataFrame, y: pd.Series, *args, **kwargs):
        """A method that trains a decision tree model from the training set (x, y)."""
        raise NotImplementedError

    def predict(self, x: pd.DataFrame, y_label: str, *args, **kwargs):
        """A method that classifies new data using the trained decision tree model."""
        raise NotImplementedError

    def test(self, x_test: pd.DataFrame, y_test: pd.Series, *args, **kwargs):
        """A method that tests the decision tree model on the test set (x_test, y_test)."""
        raise NotImplementedError
