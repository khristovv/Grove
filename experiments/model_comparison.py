# training data is used from https://www.kaggle.com/competitions/playground-series-s3e16/data?select=test.csv
import pandas as pd

import os
import sys


# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", ".")
sys.path.append(grove_dir)

from grove.trees import ClassificationTree  # noqa
from grove.forests import RandomForestClassifer, RandomForestRegressor  # noqa
from grove.utils.sampling import Sampler  # noqa
from grove.utils.plotting import Plotter  # noqa

DATA_PATH = "./data/Classification/Intermediate/data.csv"
CONFIG_PATH = "./data/Classification/Intermediate/config.csv"

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH, index_col="UDI")
    y = data["Machine failure"]
    x = data.drop("Machine failure", axis=1)
    encoding_config = pd.read_csv(CONFIG_PATH)

    seed = 1
    x_train, y_train, x_test, y_test = Sampler().get_y_proportional_train_test_split(x=x, y=y, seed=seed)

    actual_column = f"ACTUAL_{y.name}"
    predicted_column = f"PREDICTED_{y.name}"

    classification_tree = ClassificationTree(
        y_dtype="bin",
        encoding_config=encoding_config,
        max_children=4,
        max_depth=4,
        min_samples_per_node=20,
        statistics_enabled=True,
        consecutive_splits_on_same_feature_enabled=False,
    )
    classification_tree.train(x=x_train, y=y_train)

    random_forest_classifier = RandomForestClassifer(
        n_trees=20,
        encoding_config=encoding_config,
        # train_in_parallel=False,
        tree_args={
            "y_dtype": "bin",
            "max_children": 4,
            "min_samples_per_node": 20,
            "max_depth": 4,
            "statistics_enabled": True,
            "consecutive_splits_on_same_feature_enabled": False,
        },
        m_split=4,
        n_bag=1500,
        seed=seed,
        oob_score_enabled=True,
        test_on_in_bag_samples_enabled=True,
        min_number_of_classes=200,
    )
    random_forest_classifier.train(x=x_train, y=y_train)

    cut_off = 0.25
    random_forest_regressor = RandomForestRegressor(
        n_trees=20,
        encoding_config=encoding_config,
        # train_in_parallel=False,
        tree_args={
            "max_children": 4,
            "min_samples_per_node": 20,
            "max_depth": 4,
            "logging_enabled": False,
            "statistics_enabled": False,
            "consecutive_splits_on_same_feature_enabled": False,
        },
        cut_off=cut_off,
        m_split=4,
        n_bag=8000,
        seed=seed,
        logging_enabled=True,
        oob_score_enabled=True,
        test_on_in_bag_samples_enabled=True,
        min_number_of_classes=200,
    )
    random_forest_regressor.train(x=x_train, y=y_train)

    with Plotter() as plotter:
        # Classification Tree model
        tree_test_results = classification_tree.test(x_test=x_train, y_test=y_train)
        labeled_data = tree_test_results.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Матрица на неточностите на Класификационно дърво върху Обучителната извадка",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        tree_test_results = classification_tree.test(x_test=x_test, y_test=y_test)
        labeled_data = tree_test_results.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Матрица на неточностите на Класификационно дърво върху Тестовата извадка",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        # Random Forest Classifier model
        test_results_on_train_split = random_forest_classifier.test(x_test=x_train, y_test=y_train)
        labeled_data = test_results_on_train_split.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Матрица на неточностите на Класификационна гора върху Обучителната извадка",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        test_results_on_test_split = random_forest_classifier.test(x_test=x_test, y_test=y_test)
        labeled_data = test_results_on_test_split.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Матрица на неточностите на Класификационна гора върху Тестовата извадка",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        test_results_on_oob, test_results_on_in_bag = random_forest_classifier.oob_test(original_y=y)
        labeled_data = test_results_on_oob.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Матрица на неточностите на Класификационна гора върху Багинг извадката",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        labeled_data = test_results_on_in_bag.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title="Матрица на неточностите на Класификационна гора върху Извън Багинг извадката",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        # Random Forest Regressor model with cut off
        test_results = random_forest_regressor.test(x_test=x_train, y_test=y_train)
        labeled_data = test_results.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title=f"Матрица на неточностите на Регресионноа гора с cut_off '{cut_off}' върху Обучителната извадка",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        test_results = random_forest_regressor.test(x_test=x_test, y_test=y_test)
        labeled_data = test_results.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title=f"Матрица на неточностите на Регресионноа гора с cut_off '{cut_off}' върху Тестовата извадка",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        test_results_on_oob, test_results_on_in_bag = random_forest_regressor.oob_test(original_y=y)
        labeled_data = test_results_on_oob.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title=f"Матрица на неточностите на Регресионноа гора с cut_off '{cut_off}' върху Багинг извадката",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )

        labeled_data = test_results_on_in_bag.labeled_data
        plotter.plot_confusion_matrix(
            actual_column=labeled_data[actual_column],
            predicted_column=labeled_data[predicted_column],
            title=f"Матрица на неточностите на Регресионноа гора с cut_off '{cut_off}' върху Извън Багинг извадката",
            ylabel="Действителни",
            xlabel="Прогнозни",
        )
