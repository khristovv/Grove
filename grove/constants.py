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
