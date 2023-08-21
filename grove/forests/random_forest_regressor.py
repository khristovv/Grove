import pandas as pd

from grove.trees import RegressionTree

from grove.forests.base_random_forest import BaseRandomForest
from grove.utils.metrics import mean_absolute_error, mean_squared_error, r2_score
from grove.utils.sampling import Sampler
from grove.validation import TestResults


class RandomForestRegressor(BaseRandomForest):
    def __init__(
        self,
        n_trees: int,
        encoding_config: pd.DataFrame,
        train_in_parallel: bool = True,
        tree_args: dict = None,
        m_split: int = None,
        n_bag: int = None,
        seed: int | str = None,
        oob_score_enabled: bool = False,
        auto_split: bool = False,
    ):
        super().__init__(
            n_trees=n_trees,
            encoding_config=encoding_config,
            tree_model=RegressionTree,
            train_in_parallel=train_in_parallel,
            tree_args=tree_args,
            m_split=m_split,
            n_bag=n_bag,
            seed=seed,
            oob_score_enabled=oob_score_enabled,
            auto_split=auto_split,
        )

    def _get_test_train_split(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        return Sampler().get_train_test_split(x, y)

    def _vote(self, predictions_df: pd.DataFrame):
        return predictions_df.apply(lambda row: row.mean(), axis=1)

    def _build_test_results(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> TestResults:
        """Build the test results."""
        test_results = TestResults(labeled_data=labeled_data)

        test_results.add_metric(
            label="R2 Score",
            value=f"{r2_score(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]):.2}",
        )
        mean_absolute_error_value = mean_absolute_error(
            actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]
        )
        test_results.add_metric(
            label="Mean Absolute Error",
            value=f"{mean_absolute_error_value:.2}",
        )
        mean_squared_error_value = mean_squared_error(
            actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]
        )
        test_results.add_metric(
            label="Mean Squared Error",
            value=f"{mean_squared_error_value:.2}",
        )

        return test_results
