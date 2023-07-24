from pathlib import Path

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class EncodedData:
    """Placeholder for the encoded dataset"""

    x: pd.DataFrame
    xtp: pd.Series
    y: pd.DataFrame
    ytp: str
    features: Iterable[str]
    vtp: pd.Series  # ?


@dataclass
class SplitResult:
    """Placeholder for the results values of a split"""

    gain: float
    value: float
    feature: str


@dataclass
class TestResults:
    DEFAULT_OUTPUT_DIR = "test_results"
    DEFAULT_LABELED_DATA_FILENAME = "labeled_data.csv"
    DEFAULT_SCORE_FILENAME = "score.csv"
    DEFAULT_TREE_STATISTICS_FILENAME = "tree_statistics.csv"

    labeled_data: pd.DataFrame
    tree_statistics: pd.DataFrame
    misclassification_error: float
    misclassified_indexes: pd.Series

    def __str__(self) -> str:
        return (
            f"Test Sample Size {len(self.labeled_data)}\n"
            f"Missclassified Records Count {len(self.misclassified_indexes)}\n"
            f"Misclassification rate: {self.misclassification_error_perc:.2f}%\n"
            f"Accuracy: {self.accuracy:.2f}%\n"
            f"Misclassified Records indexes: {', '.join(str(v) for v in self.misclassified_indexes.values)}\n"
        )

    @property
    def misclassification_error_perc(self):
        return self.misclassification_error * 100

    @property
    def accuracy(self):
        return 100 - self.misclassification_error_perc

    def save(
        self,
        output_dir: str = None,
        labeled_data_filename: str = None,
        score_filename: str = None,
        tree_statistics_filename: str = None,
    ):
        """Save the test results to a file."""
        output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        labeled_data_filename = labeled_data_filename or self.DEFAULT_LABELED_DATA_FILENAME
        score_filename = score_filename or self.DEFAULT_SCORE_FILENAME
        tree_statistics_filename = tree_statistics_filename or self.DEFAULT_TREE_STATISTICS_FILENAME

        # save labeled data to file
        labeled_data = self.labeled_data

        labeled_data_filepath = Path(f"{output_dir}/{labeled_data_filename}")
        labeled_data_filepath.parent.mkdir(parents=True, exist_ok=True)
        labeled_data.to_csv(labeled_data_filepath)

        # save score data to file
        score_df = pd.DataFrame(
            {
                "Test Sample Size": [len(self.labeled_data)],
                "Missclassified Records Count": [len(self.misclassified_indexes)],
                "Misclassification Rate": [self.misclassification_error_perc],
                "Accuracy": [self.accuracy],
                "Misclassified indexes": [", ".join(str(v) for v in self.misclassified_indexes.values)],
            }
        )

        score_filepath = Path(f"{output_dir}/{score_filename}")
        score_filepath.parent.mkdir(parents=True, exist_ok=True)
        score_df.to_csv(score_filepath, index=False)

        # save tree statistics to file
        tree_statistics_filepath = Path(f"{output_dir}/{tree_statistics_filename}")
        tree_statistics_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.tree_statistics.to_csv(tree_statistics_filepath)


@dataclass
class Coordinates:
    """Placeholder for the coordinates of a node"""

    depth: int
    index: int

    def __str__(self):
        return f"[{self.depth},{self.index}]"

    @property
    def as_tuple(self):
        return self.depth, self.index