class SpecialChars:
    TREE_BRANCH = "├──"
    TREE_LAST_BRANCH = "└──"
    TREE_PATH = "│"

    ELEMENT_OF = "∈"


class Criteria:
    GINI = "Gini"
    CHI2 = "Chi2"
    F = "Frat"

    ALL = [GINI, CHI2, F]


class TreeStatistics:
    LABEL = "Label"
    SPLIT_FEATURE = "Split_Variable"
    DEPTH = "Depth"
    CHILDREN = "Children"
    SIZE = "Size"
    MY0 = "my0"
    MY1 = "my1"

    ALL = [LABEL, SPLIT_FEATURE, DEPTH, CHILDREN, SIZE]


class Metrics:
    # classification metrics
    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    F1_SCORE = "F1 Score"

    CLASSIFICATION = [ACCURACY, PRECISION, RECALL, F1_SCORE]

    # regression metrics
    R2_SCORE = "R2 Score"
    MAE = "Mean Absolute Error"
    RMSE = "Root Mean Squared Error"

    REGRESSION = [R2_SCORE, MAE, RMSE]

    ALL = CLASSIFICATION + REGRESSION
