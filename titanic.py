#!/c/Users/User/Anaconda2/python

""" Titanic: Machine Learning from Disaster
https://www.kaggle.com/c/titanic/details
"""

import csv as csv
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action="ignore", category=FutureWarning)

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


if __name__ == "__main__":
    TEST_DATA = read_data_from_file('./data/test_2.csv')
    TRAIN_DATA = read_data_from_file('./data/train.csv')

    # Collect the test data's PassengerIds before dropping the series
    PASSENGER_IDS = TEST_DATA['PassengerId'].values
    SURVIVORS = TRAIN_DATA[['Survived']].values

    # Massage the data
    TEST_DATA = transform_data(TEST_DATA)
    TRAIN_DATA = transform_data(TRAIN_DATA)

    # The data is now ready to go. So lets fit to the train, then predict to the test!
    # Convert back to a numpy array
    TRAIN_DATA = TRAIN_DATA.values
    TEST_DATA = TEST_DATA.values

    print 'Training...'
    FOREST = RandomForestClassifier(n_estimators=1000)
    FOREST = FOREST.fit(TRAIN_DATA[0::, 1::], TRAIN_DATA[0::, 0])

    print 'Predicting...'
    OUTPUT = FOREST.predict(TEST_DATA).astype(int)
    SCORE = FOREST.score(TEST_DATA, SURVIVORS)

    print 'Writing output file...'
    PREDICTIONS_FILE = open("myfirstforest.csv", "wb")
    OPEN_FILE_OBJ = csv.writer(PREDICTIONS_FILE)
    OPEN_FILE_OBJ.writerow(["PassengerId", "Survived"])
    OPEN_FILE_OBJ.writerows(zip(PASSENGER_IDS, OUTPUT))
    PREDICTIONS_FILE.close()

    print 'Score: {0}'.format(SCORE)
    print 'Done.'
