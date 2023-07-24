import multiprocessing as mp

import pandas as pd

from grove.types import Trees

from grove.bagging import BaggingMixin
from grove.forests.abstract import AbstractForest

from grove.forests.constants import DEFAULT_TREE_ARGS


class BaseRandomForest(AbstractForest, BaggingMixin):
    def __init__(
        self,
        n_trees: int,
        encoding_config: pd.DataFrame,
        tree_model: Trees,
        train_in_parallel: bool = True,
        tree_args: dict = None,
        # колко броя от свойствата ще участват в обучението на всяко дърво
        # ако не е зададено, ще се ползват всички свойства
        m_split: int = None,
        # колко ще е голяма извадката, с която ще обучаваме дървото
        n_bag: int = None,
        # когато ще избираме следващите N случайни велични от X
        # важното е seed-a да ни е същия
        seed: int | str = None,
    ):
        """A random forest model that uses decision trees as base learners."""
        self.n_trees = n_trees
        self.encoding_config = encoding_config
        self.tree_model = tree_model
        self.train_in_parallel = train_in_parallel
        self.tree_args = tree_args or DEFAULT_TREE_ARGS
        self.m_split = m_split
        self.n_bag = n_bag
        self.seed = seed

        self.trees = []

    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that trains the random forest model from the training set (x, y)."""
        # y_column_name = y.columns[0]

        def _get_training_data():
            """
            We use a generator so we don't store all the data used to train each individual tree in memory.
            """
            for i in range(self.n_trees):
                x_subset = x[self.encoding_config["cname"]]

                if self.m_split:
                    # get `m_split` random columns from `x`
                    x_subset = x_subset.sample(n=self.m_split, axis="columns", random_state=self.seed + i)

                bootstrap_dataset, out_of_bag_dataset = self.bootstrap(
                    dataset=x_subset, n=self.n_bag, seed=self.seed + i
                )
                y_subset = y.loc[bootstrap_dataset.index]

                encoding_config_subset = self.encoding_config.loc[self.encoding_config["cname"].isin(x_subset.columns)]
                encoding_config_subset.reset_index(drop=True, inplace=True)

                yield (encoding_config_subset, bootstrap_dataset, y_subset)

        if self.train_in_parallel:
            with mp.Pool() as process_pool:
                trained_trees = process_pool.starmap(self._train_tree, _get_training_data())
        else:
            trained_trees = [self._train_tree(*args) for args in _get_training_data()]

        self.trees.extend(trained_trees)

    def _train_tree(
        self,
        encoding_config_subset: pd.DataFrame,
        bootstrap_dataset: pd.DataFrame,
        y_subset: pd.DataFrame,
    ):
        tree = self.tree_model(encoding_config=encoding_config_subset, **self.tree_args)

        tree.train(
            x=bootstrap_dataset,
            y=y_subset,
        )

        return tree

    def _vote(self, predictions_df: pd.DataFrame) -> pd.Series:
        """A method that returns the most common prediction from the predictions_df."""
        raise NotImplementedError

    def predict(self, x: pd.DataFrame, y_label: str) -> pd.DataFrame:
        predictions_df = pd.DataFrame()
        labeled_data = x.copy()

        for index, tree in enumerate(self.trees):
            predictions = tree.predict(x=x, y_label=y_label, return_y_only=True)
            predictions_df[f"{y_label}_tree_{index}"] = predictions

        labeled_data[y_label] = self._vote(predictions_df=predictions_df)

        return labeled_data
