import matplotlib.pyplot as plt

import pandas as pd
import seaborn

from grove.utils.metrics import confusion_matrix


class Plotter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        plt.show()

    def plot_confusion_matrix(
        self, actual_column: pd.Series, predicted_column: pd.Series, title: str = "Confusion Matrix"
    ):
        cm = confusion_matrix(actual=actual_column, predicted=predicted_column)

        _, ax = plt.subplots()
        seaborn.heatmap(cm, annot=True, cmap="crest", linewidth=0.5, fmt="d", ax=ax).set(
            title=title,
            xlabel="Predicted",
            ylabel="Actual",
        )
