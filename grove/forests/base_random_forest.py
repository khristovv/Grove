import multiprocessing as mp

import pandas as pd

from grove.types import Trees

from grove.bagging import BaggingMixin
from grove.forests.abstract import AbstractForest

from grove.forests.constants import DEFAULT_TREE_ARGS
from grove.utils.logging import Logger
from grove.validation import TestResults


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
        logging_enabled: bool = True,
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

        self.logging_enabled = logging_enabled
        self.logger = Logger(name=self.__class__.__name__, logging_enabled=self.logging_enabled)

    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that trains the random forest model from the training set (x, y)."""
        # y_column_name = y.columns[0]

        def _get_training_data():
            """
            We use a generator so we don't store all the data used to train each individual tree in memory.
            """
            for i in range(self.n_trees):
                identifier = f"Tree_{i}"
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

                yield (identifier, encoding_config_subset, bootstrap_dataset, y_subset)

        self.logger.log_section("Training - Start", add_newline=False)

        if self.train_in_parallel:
            with mp.Pool() as process_pool:
                trained_trees = process_pool.starmap(self._train_tree, _get_training_data())
        else:
            trained_trees = [self._train_tree(*args) for args in _get_training_data()]

        self.trees.extend(trained_trees)

        self.logger.log_section("Training - Complete")

    def _train_tree(
        self,
        identifier,
        encoding_config_subset: pd.DataFrame,
        bootstrap_dataset: pd.DataFrame,
        y_subset: pd.DataFrame,
    ):
        self.logger.log(f"Training '{identifier}'")
        tree = self.tree_model(identifier=identifier, encoding_config=encoding_config_subset, **self.tree_args)

        tree.train(
            x=bootstrap_dataset,
            y=y_subset,
        )
        self.logger.log(f"Training '{identifier}' - Complete - Tree:")
        self.logger.log(tree)

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

    def _get_misclassified_values(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> pd.Series:
        """Get the misclassified values."""
        raise NotImplementedError

    def test(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        save_results: bool = False,
        output_dir: str = None,
    ):
        """Test the model on a test dataset."""
        self.logger.log_section("Testing", add_newline=False)

        y_label = y.columns[0]
        predicted_column = f"PREDICTED_{y_label}"
        actual_column = f"ACTUAL_{y_label}"

        labeled_data = self.predict(x=x, y_label=predicted_column)
        labeled_data[actual_column] = y

        misclassifed_values = self._get_misclassified_values(
            labeled_data=labeled_data,
            actual_column=actual_column,
            predicted_column=predicted_column,
        )
        misclassifed_values_count = misclassifed_values.value_counts()[True]
        misclassification_error = misclassifed_values_count / len(labeled_data)

        test_results = TestResults(
            labeled_data=labeled_data,
            misclassification_error=misclassification_error,
            misclassified_indexes=labeled_data[misclassifed_values].index,
        )

        if save_results:
            test_results.save(output_dir=output_dir)

        self.logger.log_section("Test Results:")
        self.logger.log(test_results)

        return test_results
