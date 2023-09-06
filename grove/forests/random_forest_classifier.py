from functools import partial
from typing import Callable

import pandas as pd

from grove.trees import ClassificationTree

from grove.constants import Metrics
from grove.forests.base_random_forest import BaseRandomForest
from grove.metrics import accuracy, f1_score, precision, recall
from grove.utils.sampling import Sampler
from grove.validation import TestResults


class RandomForestClassifer(BaseRandomForest):
    def __init__(
        self,
        n_trees: int,
        encoding_config: pd.DataFrame,
        train_in_parallel: bool = True,
        tree_args: dict = None,
        m_split: int = None,
        n_bag: int = None,
        seed: int | str = None,
        logging_enabled: bool = True,
        oob_score_enabled: bool = False,
        test_on_in_bag_samples_enabled: bool = False,
        auto_split: bool = False,
        min_number_of_classes: int | None = None,
    ):
        super().__init__(
            n_trees=n_trees,
            encoding_config=encoding_config,
            tree_model=ClassificationTree,
            train_in_parallel=train_in_parallel,
            tree_args=tree_args,
            m_split=m_split,
            n_bag=n_bag,
            seed=seed,
            logging_enabled=logging_enabled,
            oob_score_enabled=oob_score_enabled,
            test_on_in_bag_samples_enabled=test_on_in_bag_samples_enabled,
            auto_split=auto_split,
        )
        self.min_number_of_classes = min_number_of_classes

    def _get_sampling_method(self) -> Callable:
        return partial(self.bootstrap_balanced, min_number_of_classes=self.min_number_of_classes)

    def _get_test_train_split(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        return Sampler().get_y_proportional_train_test_split(x=x, y=y, seed=self.seed)

    def _vote(self, predictions_df: pd.DataFrame):
        return predictions_df.apply(lambda row: row.mode()[0], axis=1)

    def _build_test_results(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> TestResults:
        """Build the test results."""
        test_results = TestResults(labeled_data=labeled_data)

        misclassified_records = labeled_data[labeled_data[actual_column] != labeled_data[predicted_column]]

        test_results.add_metric(
            label="Misclassified Records Count",
            value=len(misclassified_records),
        )
        test_results.add_metric(
            label=Metrics.ACCURACY,
            value=accuracy(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]),
        )
        test_results.add_metric(
            label=Metrics.PRECISION,
            value=precision(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]),
        )
        test_results.add_metric(
            label=Metrics.RECALL,
            value=recall(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]),
        )
        test_results.add_metric(
            label=Metrics.F1_SCORE,
            value=f1_score(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]),
        )

        return test_results
