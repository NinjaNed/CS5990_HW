# -------------------------------------------------------------------------
# AUTHOR: Nythi (Ned) Udomkesmalee
# FILENAME: decision_tree.py
# SPECIFICATION: Trains a decision trees based on training data and averages training accuracies
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']
YES_ENUM = 1
NO_ENUM = 2

DEBUG = False


def main(data_test, data_test_labels):
    for ds in dataSets:
        print('')
        print('USING DATA SET: {}'.format(ds))
        X = []
        Y = []

        df = pd.read_csv(ds, sep=',', header=0)    # reading a dataset eliminating the header (Pandas library)
        data_training = np.array(df.values)[:, 1:]  # creating a training matrix without the id (NumPy library)

        if DEBUG:
            print('TRAINING DATA ({}):\n'.format(ds), data_training)

        # transform the original training features to numbers and add them to the 5D array X.
        # For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
        # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]].
        # The feature Marital Status must be one-hot-encoded and Taxable Income must be converted to a float.
        # transform the original training classes to numbers and add them to the vector Y.
        # For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        for row in data_training:
            new_row = transform_row(row)
            X.append(new_row)
            Y.append(yes_no_to_number(row[-1]))

        if DEBUG:
            print("X:", X)
            print("Y:", Y)
            print('')

        # loop your training and test tasks 10 times here
        model_accuracies = []
        for i in range(10):
            if DEBUG:
                print('TRAINING LOOP {}'.format(i+1))
            # fitting the decision tree to the data by using Gini index and no max_depth
            clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
            clf = clf.fit(X, Y)

            # plotting the decision tree
            tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
                           class_names=['Yes', 'No'], filled=True, rounded=True)
            plt.show()

            correct_predictions = 0
            num_test_samples = len(data_test)
            for index, data in enumerate(data_test):
                # transform the features of the test instances to numbers following the same strategy done during
                # training, and then use the decision tree to make the class prediction. For instance:
                # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the
                # predicted class label so that you can compare it with the true label
                # --> add your Python code here
                class_predicted = clf.predict([data])[0]

                # compare the prediction with the true label (located at data[3]) of the test instance to start
                # calculating the model accuracy.
                # --> add your Python code here
                if class_predicted == data_test_labels[index]:
                    correct_predictions += 1

                if DEBUG:
                    print('')
                    print('TEST DATA {}:'.format(index), data)
                    print('PREDICTED CLASS:', class_predicted)
                    print('TRUE LABEL:', data_test_labels[index])

            accuracy = correct_predictions / num_test_samples
            model_accuracies.append(accuracy)

            if DEBUG:
                print('Training Loop {} Model Accuracy Using {}: {}'.format(i + 1, ds, accuracy))
                print('')

        # find the average accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        average_model_accuracy = sum(model_accuracies) / len(model_accuracies)

        # print the accuracy of this model during the 10 runs (training and test set).
        # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
        # --> add your Python code here
        print('final accuracy when training on {}: {}'.format(ds, round(average_model_accuracy, 3)))


def transform_row(data_row):
    refund, marital_status, income, _ = data_row

    new_row = [yes_no_to_number(refund), 0, 0, 0, float(income.strip('k'))]
    marital_status = marital_status.lower().strip()
    if marital_status == "single":
        new_row[1] = 1
    elif marital_status == 'divorced':
        new_row[2] = 1
    else:
        new_row[3] = 1
    return new_row


def yes_no_to_number(entry):
    if entry.strip().lower() == 'yes':
        return YES_ENUM
    else:
        return NO_ENUM


if __name__ == '__main__':
    # get test data into np for future use
    raw_test_data = np.array(pd.read_csv('cheat_test.csv', sep=',', header=0).values)[:, 1:]
    test_data = [transform_row(row) for row in raw_test_data]
    test_labels = []
    for row in raw_test_data:
        test_labels.append(yes_no_to_number(row[-1]))
    if DEBUG:
        print('TEST DATA:', test_data)
        print('TEST LABELS:', test_labels)

    main(test_data, test_labels)
