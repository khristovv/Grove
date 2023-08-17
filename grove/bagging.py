import pandas as pd


class BaggingMixin:
    def bootstrap(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        bootstrap_size: int = None,
        seed: int = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """A method that performs bootstrapping on the training set."""
        bootstrap_size = bootstrap_size or len(x)

        x_bootstrap = x.sample(n=bootstrap_size, replace=True, random_state=seed)
        y_bootstrap = y.loc[x_bootstrap.index]
        x_out_of_bag = x.drop(x_bootstrap.index)
        y_out_of_bag = y.loc[x_out_of_bag.index]

        return x_bootstrap, y_bootstrap, x_out_of_bag, y_out_of_bag

    def bootstrap_balanced(
        self,
        min_number_of_classes: int,
        x: pd.DataFrame,
        y: pd.Series,
        bootstrap_size: int | None = None,
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Can be used with inbalanced datasets to get a more balanced traning set.
        """
        bootstrap_size = bootstrap_size or len(x)

        # Get the class with the least number of samples
        class_with_least_samples = y.value_counts().idxmin()

        # Get a subset `y` that belong to the class with the least number of samples
        y_least_samples_subset = y[y == class_with_least_samples]
        y_least_samples_subset = y_least_samples_subset.sample(
            n=min_number_of_classes, replace=True, random_state=seed
        )

        x_subset = x.loc[y_least_samples_subset.index]
        x_remaining = x.drop(index=x_subset.index.unique())

        n = bootstrap_size - len(x_subset)

        x_bootstrap = pd.concat([x_subset, x_remaining.sample(n=n, replace=True, random_state=seed)])
        y_bootstrap = y.loc[x_bootstrap.index]
        x_out_of_bag = x.drop(x_bootstrap.index.unique())
        y_out_of_bag = y.loc[x_out_of_bag.index]

        return x_bootstrap, y_bootstrap, x_out_of_bag, y_out_of_bag
