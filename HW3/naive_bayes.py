# -------------------------------------------------------------------------
# AUTHOR: Nythi (Ned) Udomkesmalee
# FILENAME: naive_bayes.py
# SPECIFICATION: Create Naive Bayes classifier for predicting tempurature
# FOR: CS 5990 - Assignment #3
# TIME SPENT: 2 Hours
# -------------------------------------------------------------------------

# importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

DEBUG = False


def main(training_file, test_file):
    # 11 classes after discretization
    classes = np.array([i for i in range(-22, 40, 6)])
    centers = (classes[1:] + classes[:-1])/2  # need list of midpoints to round temp to nearest bin

    # reading the training data
    # --> add your Python code here
    df_train = pd.read_csv(training_file, sep=',', header=0)  # reading a dataset eliminating the header
    X_train = np.array(df_train.values)[:, 1:-1].astype('f')  # creating a training matrix without the id or Temp
    y_train = np.array(df_train.values)[:, -1].astype('f')  # creating a training label array

    # update the training class values according to the discretization (11 values only)
    # --> add your Python code here
    y_train_classes = classes[np.digitize(y_train, centers)]  # moves the temp value to the nearest bin
    if DEBUG:
        print('CLASSES:', classes)
        print('TRAINING TEMP:', y_train[:5])
        print('TRAINING TEMP BINNED:', y_train_classes[:5])
        print('')

    # reading the test data
    # --> add your Python code here
    df_test = pd.read_csv(test_file, sep=',', header=0)  # reading a dataset eliminating the header
    X_test = np.array(df_test.values)[:, 1:-1].astype('f')  # creating a test matrix without the id or Temp
    y_test = np.array(df_test.values)[:, -1].astype('f')  # creating a test label array
    num_test_samples = len(y_test)

    # update the test class values according to the discretization (11 values only)
    # --> add your Python code here
    y_test_classes = classes[np.digitize(y_test, centers)]  # moves the temp value to the nearest bin
    if DEBUG:
        print('CLASSES:', classes)
        print('TEST TEMP:', y_test)
        print('TEST TEMP BINNED:', y_test_classes)
        print('')

    # fitting the naive_bayes to the data
    clf = GaussianNB()
    clf = clf.fit(X_train, y_train_classes)

    # make the naive_bayes prediction for each test sample and start computing its accuracy the prediction should be
    # considered correct if the output value is [-15%,+15%] distant from the real output values to calculate the
    # % difference between the prediction and the real output values use:
    # 100*(|predicted_value - real_value|)/real_value))
    # --> add your Python code here
    counter = 0
    correct_predictions = 0
    for x_test_sample, y_test_sample in zip(X_test, y_test_classes):
        temp_predict = clf.predict([x_test_sample])[0]
        predict_percentage = 100 * (abs((temp_predict - y_test_sample) / y_test_sample))
        if predict_percentage <= 15.0:
            correct_predictions += 1
        if DEBUG:
            print('TEST SAMPLE:', counter + 1)
            print('PREDICTED TEMP:', temp_predict)
            print('REAL TEMP:', y_test_sample)
            print('PREDICTION PERCENTAGE: {}%'.format(round(predict_percentage, 3)))
            print('')
        counter += 1

    # print the naive_bayes accuracy
    # --> add your Python code here
    accuracy = correct_predictions / num_test_samples
    print("naive_bayes accuracy: " + str(accuracy))


if __name__ == '__main__':
    main('weather_training.csv', 'weather_test.csv')



