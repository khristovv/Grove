import pandas as pd

from grove.trees import RegressionTree

from grove.forests.base_random_forest import BaseRandomForest


class RandomForestRegressor(BaseRandomForest):
    def __init__(
        self,
        n_trees: int,
        encoding_config: pd.DataFrame,
        allowed_diff: float = None,
        train_in_parallel: bool = True,
        tree_args: dict = None,
        m_split: int = None,
        n_bag: int = None,
        seed: int | str = None,
    ):
        self.allowed_diff = allowed_diff
        super().__init__(
            n_trees=n_trees,
            encoding_config=encoding_config,
            tree_model=RegressionTree,
            train_in_parallel=train_in_parallel,
            tree_args=tree_args,
            m_split=m_split,
            n_bag=n_bag,
            seed=seed,
        )

    def _vote(self, predictions_df: pd.DataFrame):
        return predictions_df.apply(lambda row: row.mean(), axis=1)

    def _get_misclassified_values(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> pd.Series:
        """Get the misclassified values."""
        diff = labeled_data[actual_column] - labeled_data[predicted_column]
        abs_diff = diff.abs()

        return abs_diff > self.allowed_diff

    def test(self, x: pd.DataFrame, y: pd.DataFrame, save_results: bool = False, output_dir: str = None):
        if self.allowed_diff is None:
            raise ValueError("The 'allowed_diff' parameter must be set to use the RandomForestRegressor.test method.")

        return super().test(x, y, save_results, output_dir)
