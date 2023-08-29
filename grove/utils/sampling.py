import pandas as pd


class Sampler:
    def get_train_test_split(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        training_portion: float = 0.7,  # 70 % of the data is used for training
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if not (0 < training_portion < 1):
            raise ValueError(f"'training_portion' must be between 0 and 1, inclusive. Got {training_portion}")

        n = round(len(x) * training_portion)

        x_train = x.sample(n=n, random_state=seed)
        y_train = y.loc[x_train.index]

        x_test = x.drop(index=x_train.index)
        y_test = y.loc[x_test.index]

        return x_train, y_train, x_test, y_test

    def get_y_proportional_train_test_split(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        training_portion: float = 0.8,  # 80 % of the data is used for training
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if not (0 < training_portion < 1):
            raise ValueError(f"'training_portion' must be between 0 and 1, inclusive. Got {training_portion}")

        y_labels = set(y.values)
        y_train = pd.Series(name=y.name)

        for label in y_labels:
            label_subset = y[y == label]

            label_proportion = round(len(label_subset) * training_portion)

            proportional_label_subset = label_subset.sample(n=label_proportion, random_state=seed)

            y_train = pd.concat([y_train, proportional_label_subset])

        x_train = x.loc[y_train.index]

        x_test = x.drop(index=x_train.index)
        y_test = y.loc[x_test.index]

        return x_train, y_train, x_test, y_test
