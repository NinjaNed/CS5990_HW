# -------------------------------------------------------------------------
# AUTHOR: Nythi (Ned) Udomkesmalee
# FILENAME: knn.py
# SPECIFICATION: KNN Regression Prediction for weather data
# FOR: CS 5990 - Assignment #3
# TIME SPENT: 1 hour
# -------------------------------------------------------------------------

# importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

DEBUG = False


def main(training_file, test_file):
    # defining the hyperparameter values of KNN
    k_values = [i for i in range(1, 20)]
    p_values = [1, 2]
    w_values = ['uniform', 'distance']

    # reading the training data
    df_train = pd.read_csv(training_file, sep=',', header=0)  # reading a dataset eliminating the header
    X_train = np.array(df_train.values)[:, 1:-1].astype('f')  # creating a training matrix without the id or Temp
    y_train = np.array(df_train.values)[:, -1].astype('f')  # creating a training label array

    # reading the test data
    df_test = pd.read_csv(test_file, sep=',', header=0)  # reading a dataset eliminating the header
    X_test = np.array(df_test.values)[:, 1:-1].astype('f')  # creating a test matrix without the id or Temp
    y_test = np.array(df_test.values)[:, -1].astype('f')  # creating a test label array
    num_test_samples = len(y_test)

    # loop over the hyper-parameter values (k, p, and w) ok KNN
    # --> add your Python code here
    best_model_accuracy = 0
    for k in k_values:
        for p in p_values:
            for weight in w_values:
                # fitting the knn to the data
                # --> add your Python code here
                if DEBUG:
                    print('Training with k = {}, p = {}, weight = {}'.format(k, p, weight))
                clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=weight)
                clf.fit(X_train, y_train)

                # make the KNN prediction for each test sample and start computing its accuracy
                # hint: to iterate over two collections simultaneously, use zip()
                # Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                # to make a prediction do: clf.predict([x_testSample])
                # the prediction should be considered correct if the output value is [-15%,+15%] distant from the real
                # output values to calculate the % difference between the prediction and the real output values use:
                # 100*(|predicted_value - real_value|)/real_value))
                # --> add your Python code here

                correct_predictions = 0
                for x_test_sample, y_test_sample in zip(X_test, y_test):
                    temp_predict = clf.predict([x_test_sample])[0]
                    predict_percentage = 100 * (abs((temp_predict - y_test_sample) / y_test_sample))
                    if predict_percentage <= 15.0:
                        correct_predictions += 1

                # check if the calculated accuracy is higher than the previously one calculated. If so, update the
                # highest accuracy and print it together with the KNN hyper-parameters.
                # Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
                # --> add your Python code here
                accuracy = correct_predictions / num_test_samples
                if DEBUG:
                    print('Model Accuracy:', accuracy)
                    print('')
                if accuracy > best_model_accuracy:
                    best_model_accuracy = accuracy
                    print('Highest KNN accuracy so far: {}'.format(accuracy))
                    print('Parameters: k = {}, p = {}, weight = {}'.format(k, p, weight))
                    print('')


if __name__ == '__main__':
    main('weather_training.csv', 'weather_test.csv')
