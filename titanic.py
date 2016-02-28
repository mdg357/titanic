#!/c/Users/User/Anaconda2/python

""" Titanic: Machine Learning from Disaster
https://www.kaggle.com/c/titanic/details
"""

import csv as csv
import math
import sys
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action="ignore", category=FutureWarning)

RUN_STATISTICS = []
MIN_WEIGHT = 1
MAX_WEIGHT = 1
MAX_INT = sys.maxsize
MIN_INT = -MAX_INT - 1

def read_data_from_file(file_name):
    """ Read the raw data in from the .csv file
    """
    # For .read_csv, always use header=0 when you know row 0 is the header row
    return pd.read_csv(file_name, header=0)


def drop_unused_columns(d_frame, drop_fields):
    """ Drop columns from the data frame that are not numeric
    """
    return d_frame.drop(drop_fields, axis=1)


def transform_data(d_frame):
    """ Perform some basic transformation and cleansing of the data
    """

    drop_fields = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', \
        'Age', 'PassengerId', 'AgeIsNull']
    # Determine all of the values of Embarked series
    ports = list(enumerate(np.unique(d_frame['Embarked'])))
    # Setup a dictionary of the form Ports : Index
    ports_dict = {name : i for i, name in ports}

    # Map sex to binary gender column
    d_frame['Gender'] = d_frame['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Compute the median age by class and gender
    median_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i, j] = d_frame[(d_frame['Gender'] == i) & \
                (d_frame['Pclass'] == j + 1)]['Age'].dropna().median()

    # Create a copy of the Age column
    d_frame['AgeFill'] = d_frame['Age']

    # Note which ages were originally null
    d_frame['AgeIsNull'] = pd.isnull(d_frame.Age).astype(int)

    # Populate the AgeFill column with the median values
    for i in range(0, 2):
        for j in range(0, 3):
            d_frame.loc[(d_frame.Age.isnull()) & (d_frame.Gender == i) \
                & (d_frame.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]

    # Again convert all Embarked strings to int
    d_frame.Embarked = d_frame.Embarked.map(lambda x: ports_dict[x]).astype(int)

    # All the missing Fares -> assume median of their respective class
    if len(d_frame.Fare[d_frame.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        for fare in range(0, 3):
            median_fare[fare] = d_frame[d_frame.Pclass == fare + 1]['Fare'].dropna().median()
        for fare in range(0, 3):
            d_frame.loc[(d_frame.Fare.isnull()) & \
                (d_frame.Pclass == fare + 1), 'Fare'] = median_fare[fare]

    # Drop the unnecessary columns and return the data frame
    return drop_unused_columns(d_frame, drop_fields)


def create_forest(train_data, test_data, survivors, weight_dict, row_to_remove):
    """ Create and train a random forest based on the given parameters
    """

    # Remove the indicated row from the numpy array
    train_data = np.delete(train_data, row_to_remove, 0)
    test_data = np.delete(test_data, row_to_remove, 0)
    survivors = np.delete(survivors, row_to_remove, 0)

    forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, \
        class_weight=weight_dict, max_features=None)
    forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
    forest.predict(test_data).astype(int) #OUTPUT = FOREST.predict(test_data).astype(int)
    score = forest.score(test_data, survivors)

    #PREDICTIONS_FILE = open("myfirstforest.csv", "wb")
    #OPEN_FILE_OBJ = csv.writer(PREDICTIONS_FILE)
    #OPEN_FILE_OBJ.writerow(["PassengerId", "Survived"])
    #OPEN_FILE_OBJ.writerows(zip(PASSENGER_IDS, OUTPUT))
    #PREDICTIONS_FILE.close()

    RUN_STATISTICS.append(score)
    
    print 'Removed row {0:000}, Score: {1}'.format(row_to_remove, score)
    sys.stdout.flush()
    return


def generate_weight_dictionaries(min_weight, max_weight):
    """ Create a list of weight dictionaries based on the provided min/max
    """
    weight_dicts = []

    for wgt_one in range(min_weight, max_weight + 1):
        for wgt_two in range(min_weight, max_weight + 1):
            weight_dicts.append({0: wgt_one, 1: wgt_two})

    return weight_dicts


if __name__ == "__main__":
    TEST_DATA = read_data_from_file('./data/test_2.csv')
    TRAIN_DATA = read_data_from_file('./data/train.csv')
    RECORDS_TO_IGNORE = math.ceil(len(TRAIN_DATA) * 0.05)
    ROW_COUNT = len(TRAIN_DATA)

    # Collect the test data's PassengerIds before dropping the series
    PASSENGER_IDS = TEST_DATA['PassengerId'].values
    SURVIVORS = TRAIN_DATA[['Survived']].values

    # Massage the data
    TEST_DATA = transform_data(TEST_DATA)
    TRAIN_DATA = transform_data(TRAIN_DATA)

    # Convert back to a numpy array
    TRAIN_DATA = TRAIN_DATA.values
    TEST_DATA = TEST_DATA.values

    WEIGHT_DICTS = generate_weight_dictionaries(MIN_WEIGHT, MAX_WEIGHT)

    for row_num in range(0, ROW_COUNT):
        create_forest(TRAIN_DATA, TEST_DATA, SURVIVORS, None, row_num)
        
    OUTLIER_FILE = open("outliers.csv", "wb")
    OPEN_FILE_OBJ = csv.writer(OUTLIER_FILE)
    OPEN_FILE_OBJ.writerow(["RowRemoved", "Score"])
    OPEN_FILE_OBJ.writerows(zip(PASSENGER_IDS, RUN_STATISTICS))
    OUTLIER_FILE.close()
