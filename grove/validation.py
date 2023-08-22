from pathlib import Path

import pandas as pd


class TestResults:
    DEFAULT_OUTPUT_DIR = "test_results"
    DEFAULT_LABELED_DATA_FILENAME = "labeled_data.csv"
    DEFAULT_SCORE_FILENAME = "score.csv"

    def __init__(
        self,
        labeled_data: pd.DataFrame,
        # misclassified_indexes: pd.Series,
    ):
        self.labeled_data = labeled_data
        # self.misclassified_indexes = misclassified_indexes

        self.metrics = {}
        self._set_default_metrics()

    def __str__(self) -> str:
        return "\n".join(f"{label} {value}" for label, value in self.metrics.items())

    def add_metric(self, label: str, value: float | int | str):
        self.metrics[label] = value

    def _set_default_metrics(self) -> dict[str, float | int]:
        self.metrics = {
            "Test Sample Size": len(self.labeled_data),
        }

    def save(
        self,
        output_dir: str = None,
        labeled_data_filename: str = None,
        score_filename: str = None,
    ):
        """Save the test results to a file."""
        output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        labeled_data_filename = labeled_data_filename or self.DEFAULT_LABELED_DATA_FILENAME
        score_filename = score_filename or self.DEFAULT_SCORE_FILENAME

        # save labeled data to file
        labeled_data = self.labeled_data

        labeled_data_filepath = Path(f"{output_dir}/{labeled_data_filename}")
        labeled_data_filepath.parent.mkdir(parents=True, exist_ok=True)
        labeled_data.to_csv(labeled_data_filepath)

        # save score data to file
        score_df = pd.Series(self.metrics, name="Value")

        score_filepath = Path(f"{output_dir}/{score_filename}")
        score_filepath.parent.mkdir(parents=True, exist_ok=True)
        score_df.to_csv(score_filepath, index_label="Metric")
