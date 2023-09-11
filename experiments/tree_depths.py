# training data is used from https://www.kaggle.com/competitions/playground-series-s3e16/data?select=test.csv
import pandas as pd

import os
import sys

# Add the parent directory (Grove) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))

grove_dir = os.path.join(current_dir, "..", ".")
sys.path.append(grove_dir)

from grove.trees import ClassificationTree  # noqa
from grove.constants import Metrics  # noqa
from grove.forests import RandomForestRegressor  # noqa
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

    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    columns = ["Train", "Test", "In-Bag", "OOB"]

    accuracy_change_rf = pd.DataFrame(index=max_depths, columns=columns)
    precision_change_rf = pd.DataFrame(index=max_depths, columns=columns)
    recall_change_rf = pd.DataFrame(index=max_depths, columns=columns)
    f1_score_change_rf = pd.DataFrame(index=max_depths, columns=columns)

    accuracy_change_dt = pd.DataFrame(index=max_depths, columns=columns[:2])
    precision_change_dt = pd.DataFrame(index=max_depths, columns=columns[:2])
    recall_change_dt = pd.DataFrame(index=max_depths, columns=columns[:2])
    f1_score_change_dt = pd.DataFrame(index=max_depths, columns=columns[:2])

    for max_depth in max_depths:
        # Classification tree
        classification_tree = ClassificationTree(
            y_dtype="bin",
            encoding_config=encoding_config,
            max_children=4,
            min_samples_per_node=20,
            max_depth=max_depth,
            logging_enabled=False,
            statistics_enabled=False,
            consecutive_splits_on_same_feature_enabled=False,
        )
        classification_tree.train(x=x_train, y=y_train)

        test_results_on_train_split = classification_tree.test(
            x_test=x_train,
            y_test=y_train,
            save_results=True,
            output_dir=f"test_results_max_depth_{max_depth}",
            labeled_data_filename="dt_labeled_data_train_split.csv",
            score_filename="dt_train_split_score_results.csv",
        )

        accuracy_change_dt.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.ACCURACY]
        precision_change_dt.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.PRECISION]
        recall_change_dt.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.RECALL]
        f1_score_change_dt.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.F1_SCORE]

        test_results_on_test_split = classification_tree.test(
            x_test=x_test,
            y_test=y_test,
            save_results=True,
            output_dir=f"test_results_max_depth_{max_depth}",
            labeled_data_filename="dt_labeled_data_test_split.csv",
            score_filename="dt_test_split_score_results.csv",
        )

        accuracy_change_dt.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.ACCURACY]
        precision_change_dt.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.PRECISION]
        recall_change_dt.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.RECALL]
        f1_score_change_dt.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.F1_SCORE]

        cut_off = 0.41

        # Regression Forest
        random_forest_regressor = RandomForestRegressor(
            n_trees=64,
            encoding_config=encoding_config,
            tree_args={
                "max_children": 4,
                "min_samples_per_node": 20,
                "max_depth": max_depth,
                "logging_enabled": False,
                "statistics_enabled": False,
                "consecutive_splits_on_same_feature_enabled": False,
            },
            cut_off=cut_off,
            m_split=4,
            n_bag=2000,
            seed=seed,
            logging_enabled=True,
            oob_score_enabled=True,
            test_on_in_bag_samples_enabled=True,
            min_number_of_classes=200,
        )
        random_forest_regressor.train(x=x_train, y=y_train)

        test_results_on_train_split = random_forest_regressor.test(
            x_test=x_train,
            y_test=y_train,
            save_results=True,
            output_dir=f"test_results_max_depth_{max_depth}",
            labeled_data_filename="rf_labeled_data_train_split.csv",
            score_filename="rf_train_split_score_results.csv",
        )

        accuracy_change_rf.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.ACCURACY]
        precision_change_rf.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.PRECISION]
        recall_change_rf.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.RECALL]
        f1_score_change_rf.loc[max_depth, "Train"] = test_results_on_train_split.metrics[Metrics.F1_SCORE]

        test_results_on_test_split = random_forest_regressor.test(
            x_test=x_test,
            y_test=y_test,
            save_results=True,
            output_dir=f"test_results_max_depth_{max_depth}",
            labeled_data_filename="rf_labeled_data_test_split.csv",
            score_filename="rf_test_split_score_results.csv",
        )

        accuracy_change_rf.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.ACCURACY]
        precision_change_rf.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.PRECISION]
        recall_change_rf.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.RECALL]
        f1_score_change_rf.loc[max_depth, "Test"] = test_results_on_test_split.metrics[Metrics.F1_SCORE]

        test_results_on_oob, test_results_on_in_bag = random_forest_regressor.oob_test(
            original_y=y,
            output_dir=f"test_results_max_depth_{max_depth}",
            oob_labeled_data_filename="rf_oob_labeled_data.csv",
            oob_score_filename="rf_oob_score_results.csv",
            in_bag_labeled_data_filename="rf_in_bag_labeled_data.csv",
            in_bag_score_filename="rf_in_bag_score_results.csv",
        )

        accuracy_change_rf.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.ACCURACY]
        precision_change_rf.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.PRECISION]
        recall_change_rf.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.RECALL]
        f1_score_change_rf.loc[max_depth, "In-Bag"] = test_results_on_in_bag.metrics[Metrics.F1_SCORE]

        accuracy_change_rf.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.ACCURACY]
        precision_change_rf.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.PRECISION]
        recall_change_rf.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.RECALL]
        f1_score_change_rf.loc[max_depth, "OOB"] = test_results_on_oob.metrics[Metrics.F1_SCORE]

    with Plotter() as plotter:
        # tree
        plotter.plot_metric(
            title="'Точност' на Класификационно дърво върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="Точност",
            metrics=[accuracy_change_dt["Train"], accuracy_change_dt["Test"]],
        )
        plotter.plot_metric(
            title="'Прецизност' на Класификационно дърво върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="Прецизност",
            metrics=[precision_change_dt["Train"], precision_change_dt["Test"]],
        )
        plotter.plot_metric(
            title="'Пълнота' на Класификационно дърво върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="Пълнота",
            metrics=[recall_change_dt["Train"], recall_change_dt["Test"]],
        )
        plotter.plot_metric(
            title="'F1-оценка' на Класификационно дърво върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="F1-оценка",
            metrics=[f1_score_change_dt["Train"], f1_score_change_dt["Test"]],
        )

        plotter.plot_metric(
            title="'Точност' на Регресионна Гора върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="Точност",
            metrics=[accuracy_change_rf["Train"], accuracy_change_rf["Test"]],
        )
        plotter.plot_metric(
            title="'Точност' на Регресионна Гора върху Багинг и Извън Багинг извадките",
            x_label="Дълбочина",
            y_label="Точност",
            metrics=[accuracy_change_rf["In-Bag"], accuracy_change_rf["OOB"]],
        )
        plotter.plot_metric(
            title="'Прецизност' на Регресионна Гора върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="Прецизност",
            metrics=[precision_change_rf["Train"], precision_change_rf["Test"]],
        )
        plotter.plot_metric(
            title="'Прецизност'на Регресионна Гора върху Багинг и Извън Багинг извадките",
            x_label="Дълбочина",
            y_label="Прецизност",
            metrics=[precision_change_rf["In-Bag"], precision_change_rf["OOB"]],
        )
        plotter.plot_metric(
            title="'Пълнота' на Регресионна Гора върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="Пълнота",
            metrics=[recall_change_rf["Train"], recall_change_rf["Test"]],
        )
        plotter.plot_metric(
            title="'Пълнота'на Регресионна Гора върху Багинг и Извън Багинг извадките",
            x_label="Дълбочина",
            y_label="Пълнота",
            metrics=[recall_change_rf["In-Bag"], recall_change_rf["OOB"]],
        )
        plotter.plot_metric(
            title="'F1-оценка' на Регресионна Гора върху Обучителната и Тестовата извадка",
            x_label="Дълбочина",
            y_label="F1-оценка",
            metrics=[f1_score_change_rf["Train"], f1_score_change_rf["Test"]],
        )
        plotter.plot_metric(
            title="'F1-оценка' на Регресионна Гора върху Багинг и Извън Багинг извадките",
            x_label="Дълбочина",
            y_label="F1-оценка",
            metrics=[f1_score_change_rf["In-Bag"], f1_score_change_rf["OOB"]],
        )
