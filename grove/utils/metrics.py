import pandas as pd


def r2_score(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate the R2 score."""
    residuals = actual - predicted

    ss_res = (residuals**2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()

    return 1 - (ss_res / ss_tot)


def confusion_matrix_values(actual: pd.Series, predicted: pd.Series) -> [float, float, float, float]:
    """Calculate the confusion matrix."""

    cross_tab = pd.crosstab(actual, predicted)

    tp, fn, fp, tn = 0, 0, 0, 0

    # ['ACTUAL']['POSITIVE']

    # the positive sample was correctly identified by the classifier
    try:
        tp = cross_tab.loc[1][1]
    except KeyError:
        pass

    # the positive sample is incorrectly identified by the classifier as being negative
    try:
        fn = cross_tab.loc[0][1]
    except KeyError:
        pass

    # negative sample is incorrectly identified by the classifier as being positive
    try:
        fp = cross_tab.loc[1][0]
    except KeyError:
        pass

    # the negative sample gets correctly identified by the classifier
    try:
        tn = cross_tab.loc[0][0]
    except KeyError:
        pass

    return (tp, fn, fp, tn)


def accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate how many observations, both positive and negative, were correctly classified."""
    tp, fn, fp, tn = confusion_matrix_values(actual, predicted)

    return (tp + tn) / (tp + tn + fp + fn)


def precision(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate how many of the positive samples were correctly classified."""
    tp, _, fp, _ = confusion_matrix_values(actual, predicted)

    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0


def recall(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate how many of the positive samples were correctly classified."""
    tp, fn, _, _ = confusion_matrix_values(actual, predicted)

    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0
