import numpy as np
import pandas as pd


def r2_score(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate the R2 score."""
    residuals = actual - predicted

    ss_res = (residuals**2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()

    return 1 - (ss_res / ss_tot)


def mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate the mean absolute error."""
    return (actual - predicted).abs().mean()


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate the root mean squared error."""
    return np.sqrt(((actual - predicted) ** 2).mean())


def confusion_matrix(
    actual: pd.Series,
    predicted: pd.Series,
    as_tuple=False,
) -> pd.DataFrame | tuple[float, float, float, float]:
    """Calculate the confusion matrix."""

    # the positive sample was correctly identified by the classifier
    subset = predicted[actual == 1]
    tp = len(subset[subset == 1])
    # the positive sample is incorrectly identified by the classifier as being negative
    fn = len(subset[subset == 0])

    # negative sample is incorrectly identified by the classifier as being positive
    subset = predicted[actual == 0]
    fp = len(subset[subset == 1])
    # the negative sample gets correctly identified by the classifier
    tn = len(subset[subset == 0])

    if as_tuple:
        return tp, fn, fp, tn

    return pd.DataFrame([{"PP": tp, "PN": fn}, {"PP": fp, "PN": tn}], index=["P", "N"])


def accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate how many observations, both positive and negative, were correctly classified."""
    tp, fn, fp, tn = confusion_matrix(actual=actual, predicted=predicted, as_tuple=True)

    try:
        return (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        return 0.0


def precision(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate how many of the positive samples were correctly classified."""
    tp, _, fp, _ = confusion_matrix(actual=actual, predicted=predicted, as_tuple=True)

    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0


def recall(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate how many of the positive samples were correctly classified."""
    tp, fn, _, _ = confusion_matrix(actual=actual, predicted=predicted, as_tuple=True)

    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0


def f1_score(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate the F1 score."""
    p = precision(actual, predicted)
    r = recall(actual, predicted)

    try:
        return 2 * ((p * r) / (p + r))
    except ZeroDivisionError:
        return 0.0
