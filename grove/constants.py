class SpecialChars:
    TREE_BRANCH = "├──"
    TREE_LAST_BRANCH = "└──"
    TREE_PATH = "│"

    ELEMENT_OF = "∈"


class Criteria:
    GINI = "Gini"
    CHI2 = "Chi2"

    ALL = [GINI, CHI2]


class TreeStatistics:
    LABEL = "Label"
    SPLIT_FEATURE = "Split Variable"
    DEPTH = "Depth"
    CHILDREN = "Children"
    SIZE = "Size"

    ALL = [LABEL, SPLIT_FEATURE, DEPTH, CHILDREN, SIZE]
