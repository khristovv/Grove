import pandas as pd

from grove.trees import ClassificationTree

from grove.forests.base_random_forest import BaseRandomForest


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
        )

    def _vote(self, predictions_df: pd.DataFrame):
        return predictions_df.apply(lambda row: row.mode()[0], axis=1)

    def _get_misclassified_values(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> pd.Series:
        return labeled_data[actual_column] != labeled_data[predicted_column]
