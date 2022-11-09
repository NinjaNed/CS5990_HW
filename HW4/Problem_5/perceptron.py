# -------------------------------------------------------------------------
# AUTHOR: Nythi (Ned) Udomkesmalee
# FILENAME: perceptron.py
# SPECIFICATION: Classification of data using Perceptron and MLP
# FOR: CS 5990- Assignment #4
# TIME SPENT: 30 minutes
# -------------------------------------------------------------------------

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  # pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd
import os


def main(path_to_data='.'):
    n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    r = [True, False]

    df = pd.read_csv(os.path.join(path_to_data, 'optdigits.tra'), sep=',', header=None)
    X_training = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for training
    y_training = np.array(df.values)[:, -1]  # getting the last field to form the class label for training

    df = pd.read_csv(os.path.join(path_to_data, 'optdigits.tes'), sep=',', header=None)
    X_test = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for test
    y_test = np.array(df.values)[:, -1]  # getting the last field to form the class label for test
    num_test_samples = len(X_test)

    best_perceptron_accuracy = 0
    best_mlp_accuracy = 0
    for w in n:  # iterates over n
        for b in r:  # iterates over r
            for a in range(2):  # iterates over the algorithms
                # Create a Neural Network classifier
                if a == 0:
                    # eta0 = learning rate
                    # shuffle = shuffle the training data
                    clf = Perceptron(eta0=w, shuffle=b, max_iter=1000)
                else:
                    # learning_rate_init = learning rate
                    # hidden_layer_sizes = number of neurons in the ith hidden layer
                    # shuffle = shuffle the training data
                    clf = MLPClassifier(activation='logistic', learning_rate_init=w,
                                        hidden_layer_sizes=(25,), shuffle=b, max_iter=1000)

                # Fit the Neural Network to the training data
                clf.fit(X_training, y_training)

                # make the classifier prediction for each test sample and start computing its accuracy
                # hint: to iterate over two collections simultaneously with zip() Example:
                # for (x_testSample, y_testSample) in zip(X_test, y_test):
                # to make a prediction do: clf.predict([x_testSample])
                # --> add your Python code here
                num_accurate_predictions = 0
                for x_testSample, y_testSample in zip(X_test, y_test):
                    predict = clf.predict([x_testSample])[0]
                    if predict == y_testSample:
                        num_accurate_predictions += 1
                accuracy = round(num_accurate_predictions / num_test_samples, 2)

                # check if the calculated accuracy is higher than the previously one calculated for each classifier.
                # If so, update the highest accuracy and print it together with the network hyperparameters
                # Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
                # Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
                # --> add your Python code here
                if a == 0:  # perceptron
                    if accuracy > best_perceptron_accuracy:
                        best_perceptron_accuracy = accuracy
                        print('Highest Perceptron accuracy so far: {}, Parameters: learning rate={}, shuffle={}'.format(
                            accuracy, w, b
                        ))
                else:
                    if accuracy > best_mlp_accuracy:
                        best_mlp_accuracy = accuracy
                        print('Highest MLP accuracy so far: {}, Parameters: learning rate={}, shuffle={}'.format(
                            accuracy, w, b
                        ))


if __name__ == '__main__':
    optdigits_dir = '..'
    main(optdigits_dir)
