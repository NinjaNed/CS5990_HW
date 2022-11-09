# -------------------------------------------------------------------------
# AUTHOR: Nythi (Ned) Udomkesmalee
# FILENAME: bagging_random_forest.py
# SPECIFICATION: Classify data set using single decision tree, ensemble classifier, and random forest
# FOR: CS 5990- Assignment #4
# TIME SPENT: 2 hours
# -------------------------------------------------------------------------

# importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

DEBUG = False


def main(path_to_data='.'):
    # dbTraining = []
    # dbTest = []
    X_training = []
    y_training = []
    classVotes = []  # this array will be used to count the votes of each classifier

    # reading the training data from a csv file and populate dbTraining
    # --> add your Python code here
    dbTraining = pd.read_csv(os.path.join(path_to_data, 'optdigits.tra'),
                             sep=',', header=None).values.tolist()  # reading a dataset with no header
    if DEBUG:
        print('Num Training Samples:', len(dbTraining))
        print('Num Columns in Training Sample:', len(dbTraining[0]))
        print('')

    # reading the test data from a csv file and populate dbTest
    # --> add your Python code here
    dbTest = pd.read_csv(os.path.join(path_to_data, 'optdigits.tes'),
                         sep=',', header=None).values.tolist()  # reading a dataset with no header

    # initializing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
    # --> add your Python code here
    num_test_samples = len(dbTest)
    for i in range(num_test_samples):
        classVotes.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    print("Started my base and ensemble classifier ...")

    for k in range(20):  # create 20 bootstrap samples here (k = 20). One classifier for each bootstrap sample
        bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

        # populate the values of X_training and y_training by using the bootstrapSample
        # --> add your Python code here
        X_training = [sample[:-1] for sample in bootstrapSample]
        y_training = [sample[-1] for sample in bootstrapSample]
        if DEBUG:
            if k == 0:
                print('')
                print('Num Bootstrap Samples:', len(bootstrapSample))
                print('Num Columns in Bootstrap Sample:', len(bootstrapSample[0]))
                print('Num X_training Samples:', len(X_training))
                print('Num Columns in X_training:', len(X_training[0]))
                print('')

        # fitting the decision tree to the data
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)  # use a single decision tree w/o pruning
        clf = clf.fit(X_training, y_training)

        num_accurate_predictions = 0
        for i, testSample in enumerate(dbTest):
            # make the classifier prediction for each test sample and update the corresponding index value in classVotes
            # For instance, if your first base classifier predicted 2 for the first test sample, then
            # classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
            # Later, if your second base classifier predicted 3 for the first test sample, then
            # classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
            # Later, if your third base classifier predicted 3 for the first test sample,
            # then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
            # this array will consolidate the votes of all classifier for all test samples
            # --> add your Python code here
            x_sample = [testSample[:-1]]
            predict = clf.predict(x_sample)[0]
            classVotes[i][predict] += 1

            # for only the first base classifier,
            # compare the prediction with the true label of the test sample here to start calculating its accuracy
            if k == 0:
                # --> add your Python code here
                y_sample = testSample[-1]
                if predict == y_sample:
                    num_accurate_predictions += 1

        if k == 0:  # for only the first base classifier, print its accuracy here
            # --> add your Python code here
            accuracy = round(num_accurate_predictions / num_test_samples, 2)
            print("Finished my base classifier (fast but relatively low accuracy) ...")
            print("My base classifier accuracy: " + str(accuracy))
            print("")

    # now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground
    # truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
    # --> add your Python code here
    num_accurate_predictions = 0
    for i, testSample in enumerate(dbTest):
        y_sample = testSample[-1]
        class_vote_predict = classVotes[i].index(max(classVotes[i]))
        if y_sample == class_vote_predict:
            num_accurate_predictions += 1
    accuracy = round(num_accurate_predictions / num_test_samples, 2)

    # printing the ensemble accuracy here
    print("Finished my ensemble classifier (slow but higher accuracy) ...")
    print("My ensemble accuracy: " + str(accuracy))
    print("")

    print("Started Random Forest algorithm ...")

    # Create a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=20)

    # Fit Random Forest to the training data
    clf.fit(X_training, y_training)

    # make the Random Forest prediction for each test sample. Ex: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
    # compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    # --> add your Python code here
    num_accurate_predictions = 0
    for i, testSample in enumerate(dbTest):
        x_sample = [testSample[:-1]]
        y_sample = testSample[-1]
        predict = clf.predict(x_sample)[0]
        if y_sample == predict:
            num_accurate_predictions += 1
    accuracy = round(num_accurate_predictions / num_test_samples, 2)

    # printing Random Forest accuracy here
    print("Random Forest accuracy: " + str(accuracy))

    print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")


if __name__ == '__main__':
    optdigits_dir = '..'
    main(optdigits_dir)
