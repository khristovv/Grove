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
    MEAN_SQUARED_ERROR = "Mean Squared Error"
    MEAN_ABSOLUTE_ERROR = "Mean Absolute Error"

    REGRESSION = [R2_SCORE, MEAN_SQUARED_ERROR, MEAN_ABSOLUTE_ERROR]

    ALL = CLASSIFICATION + REGRESSION
