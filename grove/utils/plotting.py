import matplotlib.pyplot as plt

import pandas as pd
import seaborn


class PlottingMixin:
    def plot_confusion_matrix(self, confusion_matrix: pd.DataFrame, title: str = "Confusion Matrix"):
        seaborn.heatmap(confusion_matrix, annot=True, cmap="crest", linewidth=0.5, fmt="d").set(
            title=title,
            xlabel="Predicted",
            ylabel="Actual",
        )
        plt.show()
