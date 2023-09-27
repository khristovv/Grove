import matplotlib.pyplot as plt

import pandas as pd
import seaborn

from grove.metrics import confusion_matrix


class Plotter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        plt.show()

    def plot_confusion_matrix(
        self,
        actual_column: pd.Series,
        predicted_column: pd.Series,
        title: str = "Confusion Matrix",
        xlabel: str = "Predicted",
        ylabel: str = "Actual",
        font_size: int | float = 20,
    ):
        cm = confusion_matrix(actual=actual_column, predicted=predicted_column)

        seaborn.set(font_scale=font_size / 10)

        _, ax = plt.subplots()
        seaborn.heatmap(
            cm,
            annot=True,
            cmap="crest",
            linewidth=0.5,
            fmt="d",
            ax=ax,
        ).set(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    def plot_metric(
        self,
        title: str,
        x_label: str,
        y_label: str,
        metrics: list[pd.Series],
        fontsize: int | float = 25,
    ):
        _, ax = plt.subplots()

        for metric in metrics:
            x = metric.index
            y = metric
            ax.plot(x, y, label=metric.name, marker=".", markersize=15, linewidth=3)
            ax.set_title(title, fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)
            ax.set_xlabel(x_label, fontsize=fontsize)
            ax.set_ylabel(y_label, fontsize=fontsize)

        ax.legend(fontsize=fontsize)
        ax.grid()

    def plot_metric_grid(
        self,
        metrics_df: pd.DataFrame,
        title: str,
        x_label: str,
        y_label: str,
        font_size: int | float = 25,
    ):
        fig, axs = plt.subplots(2, 1)
        fig.suptitle(title)

        x = metrics_df.index

        for ax, (label, y) in zip(axs.flat, metrics_df.items()):
            ax.plot(x, y, marker=".", markersize=15, linewidth=3)
            ax.set_title(label, font_size=font_size)
            ax.set(xlabel=x_label, ylabel=y_label, font_size=font_size)
            ax.grid()

            # # display value over each marker
            # for i in x:
            # # Hide x labels and tick labels for top plots and y ticks for right plots.
            #     ax.annotate(text=f"{y.loc[i]:.3f}", xy=(i, y.loc[i]), textcoords="offset points", xytext=(0, 15))

            # ax.label_outer()
