import matplotlib.pyplot as plt

import pandas as pd
import seaborn


class Plotter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix: pd.DataFrame, title: str = "Confusion Matrix"):
        _, ax = plt.subplots()
        seaborn.heatmap(confusion_matrix, annot=True, cmap="crest", linewidth=0.5, fmt="d", ax=ax).set(
            title=title,
            xlabel="Predicted",
            ylabel="Actual",
        )
