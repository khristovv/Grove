import pandas as pd

from grove.constants import Criteria, Metrics
from grove.nodes import Node
from grove.trees.base_tree import BaseTree
from grove.trees.validation import TreeTestResults
from grove.metrics import mean_absolute_error, root_mean_squared_error, r2_score


class RegressionTree(BaseTree):
    def __init__(
        self,
        encoding_config: pd.DataFrame,
        max_children: int,
        min_samples_per_node: int,
        criterion_threshold: float | None = None,
        max_depth: int | None = None,
        logging_enabled: bool = False,
        statistics_enabled: bool = False,
        consecutive_splits_on_same_feature_enabled: bool = True,
        config_values_delimiter: str = "|",
        identifier: str = "",
    ):
        super().__init__(
            encoding_config=encoding_config,
            y_dtype="num",
            max_children=max_children,
            min_samples_per_node=min_samples_per_node,
            criterion=Criteria.F,
            criterion_threshold=criterion_threshold,
            max_depth=max_depth,
            logging_enabled=logging_enabled,
            statistics_enabled=statistics_enabled,
            consecutive_splits_on_same_feature_enabled=consecutive_splits_on_same_feature_enabled,
            config_values_delimiter=config_values_delimiter,
            identifier=identifier,
        )

    @property
    def allowed_criteria(self) -> list[Criteria]:
        return [Criteria.F]

    def _leafify_node(self, node: Node, y: pd.Series):
        """Leafify node by calculating the mean of the target variable"""
        predicted_value = y.iloc[node.indexes].mean()

        node.children = []
        node.predicted_value = predicted_value

    def _build_test_results(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> TreeTestResults:
        """Build the test results."""
        test_results = TreeTestResults(labeled_data=labeled_data)

        test_results.add_metric(
            label=Metrics.R2_SCORE,
            value=r2_score(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]),
        )
        test_results.add_metric(
            label=Metrics.MAE,
            value=mean_absolute_error(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]),
        )
        test_results.add_metric(
            label=Metrics.RMSE,
            value=root_mean_squared_error(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]),
        )

        return test_results
