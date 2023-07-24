import pandas as pd


class BaggingMixin:
    def bootstrap(self, dataset: pd.DataFrame, n: int = None, seed: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """A method that performs bootstrapping on the training set."""
        n = n or len(dataset)

        boostrap_dataset = dataset.sample(n=n, replace=True, random_state=seed)
        out_of_bag_dataset = dataset.drop(boostrap_dataset.index)

        return boostrap_dataset, out_of_bag_dataset
