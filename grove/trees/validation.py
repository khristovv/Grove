from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from grove.validation import TestResults


@dataclass
class TreeTestResults(TestResults):
    DEFAULT_TREE_STATISTICS_FILENAME = "tree_statistics.csv"

    tree_statistics: pd.DataFrame

    def save(
        self,
        output_dir: str = None,
        labeled_data_filename: str = None,
        score_filename: str = None,
        tree_statistics_filename: str = None,
    ):
        super().save(output_dir, labeled_data_filename, score_filename)

        tree_statistics_filename = tree_statistics_filename or self.DEFAULT_TREE_STATISTICS_FILENAME

        # save tree statistics to file
        tree_statistics_filepath = Path(f"{output_dir}/{tree_statistics_filename}")
        tree_statistics_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.tree_statistics.to_csv(tree_statistics_filepath)
