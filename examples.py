# all training data is used from https://www.kaggle.com/c/titanic/data
import pandas as pd


from grove import DecisionTree

TEST_DATA_PATH = './data/test.csv'
TRAINING_DATA_PATH = './data/train.csv'
IRIS_DATA_PATH = './data/iris.csv'

if __name__ == "__main__":
    # survival 	Survival 	                    0 = No, 1 = Yes
    # pclass 	Ticket class 	                1 = 1st, 2 = 2nd, 3 = 3rd
    # sex 	    Sex
    # Age 	    Age in years
    # sibsp 	Number of siblings/
    #           spouses aboard the Titanic
    # parch 	Number of parents /
    #           children aboard the Titanic
    # ticket 	Ticket number
    # fare 	    Passenger fare
    # cabin 	Cabin number
    # embarked 	Port of Embarkation 	        C = Cherbourg, Q = Queenstown, S = Southampton

    dataset = pd.read_csv(IRIS_DATA_PATH)

    model = DecisionTree(
        dataset=dataset,
        features=[
            'sepal_length',
            'sepal_width',
            'petal_length',
            'petal_width'
        ],
        target='species',
    )

    model.build()

    print(model.root)

    dataset = [[2.771244718,1.784783929,0],
                [1.728571309,1.169761413,0],
                [3.678319846,2.81281357,0],
                [3.961043357,2.61995032,0],
                [2.999208922,2.209014212,0],
                [7.497545867,3.162953546,1],
                [9.00220326,3.339047188,1],
                [7.444542326,0.476683375,1],
                [10.12493903,3.234550982,1],
                [6.642287351,3.319983761,1]]
