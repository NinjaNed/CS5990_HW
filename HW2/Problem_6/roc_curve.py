# -------------------------------------------------------------------------
# AUTHOR: Nythi Udomkesmalee
# FILENAME: roc_curve.py
# SPECIFICATION: create a roc curve of cheat_data.csv
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 15 minutes
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

YES_ENUM = 1
NO_ENUM = 0

DEBUG = False


def main():
    # read the dataset cheat_data.csv and prepare the data_training numpy array
    # --> add your Python code here
    raw_cheat_data = np.array(pd.read_csv('cheat_data.csv', sep=',', header=0).values)
    cheat_data = [transform_row(row) for row in raw_cheat_data]
    test_labels = []
    for row in raw_cheat_data:
        test_labels.append(yes_no_to_number(row[-1]))

    # transform the original training features to numbers and add them to the 5D array X. For instance,
    # Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]].
    # The feature Marital Status must be one-hot-encoded and Taxable Income must be converted to a float.
    # --> add your Python code here
    X = cheat_data

    # transform the original training classes to numbers and add them to the vector y.
    # For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
    # --> add your Python code here
    y = test_labels

    if DEBUG:
        print('RAW DATA:\n', raw_cheat_data)
        print('')
        print('X:', X)
        print('y:', y)
        print('')

    # split into train/test sets using 30% for test
    # --> add your Python code here
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=.3)

    # generate a no skill prediction (random classifier - scores should be all zero)
    # --> add your Python code here
    ns_probs = np.zeros(len(testy))

    # fit a decision tree model by using entropy with max depth = 2
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
    clf = clf.fit(trainX, trainy)

    # predict probabilities for all test samples (scores)
    dt_probs = clf.predict_proba(testX)

    # keep probabilities for the positive outcome only
    # --> add your Python code here
    dt_probs = dt_probs[:, 1]

    # calculate scores by using both classifiers (no skilled and decision tree)
    ns_auc = roc_auc_score(testy, ns_probs)
    dt_auc = roc_auc_score(testy, dt_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % ns_auc)
    print('Decision Tree: ROC AUC=%.3f' % dt_auc)

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # show the legend
    pyplot.legend()

    # show the plot
    pyplot.show()


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
    main()
