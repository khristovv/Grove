import multiprocessing as mp
from typing import Callable

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
        oob_score_enabled: bool = False,
        auto_split: bool = False,
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

        self.oob_score_enabled = oob_score_enabled
        self.oob_dataset = pd.DataFrame()

        self.auto_split = auto_split
        self.test_set: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def _get_sampling_method(self) -> Callable:
        return self.bootstrap

    def _get_test_train_split(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """A method that splits the dataset into a training and test set."""
        raise NotImplementedError

    def train(self, x: pd.DataFrame, y: pd.Series):
        """A method that trains the random forest model from the training set (x, y)."""

        x_train, y_train = x, y

        if self.auto_split:
            x_train, y_train, x_test, y_test = self._get_test_train_split(x=x, y=y)
            self.test_set = x_test, y_test

        def _get_training_data():
            """
            We use a generator so we don't store all the data used to train each individual tree in memory.
            """
            for i in range(self.n_trees):
                identifier = f"Tree_{i}"
                x_subset = x_train[self.encoding_config["cname"]]

                if self.m_split:
                    # get `m_split` random columns from `x`
                    x_subset = x_subset.sample(n=self.m_split, axis="columns", random_state=self.seed + i)

                sampling_method = self._get_sampling_method()

                x_bootstrap, y_bootstrap, x_out_of_bag, y_out_of_bag = sampling_method(
                    x=x_subset, y=y_train, bootstrap_size=self.n_bag, seed=self.seed + i
                )

                encoding_config_subset = self.encoding_config.loc[self.encoding_config["cname"].isin(x_subset.columns)]
                encoding_config_subset.reset_index(drop=True, inplace=True)

                yield (identifier, encoding_config_subset, x_bootstrap, y_bootstrap, x_out_of_bag, y_out_of_bag)

        self.logger.log_section("Training - Start", add_newline=False)

        if self.oob_score_enabled:
            self.initial_y = y_train.copy()

        if self.train_in_parallel:
            with mp.Pool() as process_pool:
                results = process_pool.starmap(self._train_tree, _get_training_data())
        else:
            results = [self._train_tree(*args) for args in _get_training_data()]

        for tree, oob_predictions in results:
            if oob_predictions is not None:
                self.oob_dataset = pd.merge(
                    left=self.oob_dataset,
                    right=oob_predictions,
                    left_index=True,
                    right_index=True,
                    how="outer",
                )

            self.trees.append(tree)

        self.logger.log_section("Training - Complete")

    def _train_tree(
        self,
        identifier,
        encoding_config_subset: pd.DataFrame,
        x_bootstrap: pd.DataFrame,
        y_bootstrap: pd.Series,
        x_out_of_bag: pd.DataFrame,
        y_out_of_bag: pd.Series,
    ) -> tuple[Trees, pd.DataFrame] | tuple[Trees, None]:
        self.logger.log(f"Training '{identifier}'")
        tree = self.tree_model(identifier=identifier, encoding_config=encoding_config_subset, **self.tree_args)

        tree.train(x=x_bootstrap, y=y_bootstrap.to_frame())
        self.logger.log(f"Training '{identifier}' - Complete - Tree:")
        self.logger.log(tree)

        oob_predictions = None
        if self.oob_score_enabled:
            self.logger.log(f"Testing '{identifier}' on its out-of-bag dataset")

            test_results = tree.test(x=x_out_of_bag, y=y_out_of_bag.to_frame())

            y_label = y_out_of_bag.name
            oob_predictions = pd.DataFrame({identifier: test_results.labeled_data[f"PREDICTED_{y_label}"]})

            self.logger.log(f"Testing '{identifier}' on its out-of-bag dataset - Complete - Results:")
            self.logger.log(test_results)

        return tree, oob_predictions

    def _vote(self, predictions_df: pd.DataFrame) -> pd.Series:
        """A method that returns the most common prediction from the predictions_df."""
        raise NotImplementedError

    def predict(self, x: pd.DataFrame, y_label: str, trees: list[Trees] = []) -> pd.DataFrame:
        if not trees:
            trees = self.trees

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
        x_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        save_results: bool = False,
        output_dir: str | None = None,
    ):
        """Test the model on a test dataset."""
        self.logger.log_section("Testing", add_newline=False)

        if self.auto_split and (x_test is not None or y_test is not None):
            raise ValueError(
                "You cannot provide a test set when the model is configured to automatically split the dataset."
            )

        if not self.auto_split and (x_test is None or y_test is None):
            raise ValueError(
                "You must provide a test set when the model is not configured to automatically split the dataset."
            )

        if self.auto_split:
            x_test, y_test = self.test_set

        y_label = y_test.name
        predicted_column = f"PREDICTED_{y_label}"
        actual_column = f"ACTUAL_{y_label}"

        labeled_data = self.predict(x=x_test, y_label=predicted_column)
        labeled_data[actual_column] = y_test

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

    def get_oob_score(self) -> pd.DataFrame:
        if not self.oob_score_enabled:
            raise ValueError(
                "OOB datasets were not saved during training. "
                "Set `oob_score_enabled` to `True` and re-train the model to access this metric."
            )

        oob_score_df = self.oob_dataset.copy()
        oob_score_df["PREDICTED"] = self._vote(predictions_df=oob_score_df)
        oob_score_df["ACTUAL"] = self.initial_y

        return oob_score_df
