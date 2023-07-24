import pandas as pd

from grove.trees import RegressionTree

from grove.forests.base_random_forest import BaseRandomForest


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
        )

    def _vote(self, predictions_df: pd.DataFrame):
        return predictions_df.apply(lambda row: row.mean()[0], axis=1)
