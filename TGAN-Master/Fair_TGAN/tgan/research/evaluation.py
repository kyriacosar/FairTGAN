"""This module contains functions to evaluate the training results."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def _proc_data(df, continuous_cols, label_name=None):
    """Transform dataframe into matrix of features and its labels.

    Args:
        df(pandas.DataFrame): Dataframe to transform.
        continous_cols(list[str]): Name of columns with continous values.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: First element is the feature matrix,
        second the labels.

    """
    features = []
    sens_feature = []
    num_cols = df.shape[1]
    df.columns = list(range(num_cols))

    for i in range(num_cols - 1):

        if i == sensitive_column:
            sens_feature.append(df[i].values)

        if i in continuous_cols:
            features.append(df[i].values.reshape([-1, 1]))

        else:
            features.append(pd.get_dummies(df[i]).values)

    features = np.concatenate(features, axis=1)
    if label_name is None:
        label_name = num_cols - 1

    labels = df.iloc[:, label_name].values

    return features, sens_feature, labels


def evaluate_classification(
    train_data, test_data, continuous_cols, sensitive_column,
    classifier=DecisionTreeClassifier(max_depth=20), metric=accuracy_score
):
    """Score a model with the given data.

    Args:
        train_csv(pandas.DataFrame): Path to the train csv file.
        test_csv(pandas.DataFrame): Path to the test csv file.
        continous_cols(list[str]): List of labels of continous columns.
        classifier(object): Classifier to evaluate the classification. It have to implement
           :meth:`fit` and :meth:`predict` methods.
        metric(callable): Metric to score the classifier results.

    Returns:
        float: score for the given data, classifier and metric.

    """
    n_train = len(train_data)
    dataset = pd.concat([train_data, test_data])

    features, sens_feature, labels = _proc_data(dataset, continuous_cols, sensitive_column)

    train_set = features[:n_train], sens_feature[:n_train], labels[:n_train]
    test_set = features[n_train:], sens_feature[n_train:], labels[n_train:]

    classifier.fit(train_set[0], train_set[2])

    pred = classifier.predict(test_set[0])


    train_prot_pos = 0
    train_unprot_pos = 0

    for i in range(len(train_set)):
        if train_set[1][i] == 'Female' and train_set[2][i] == '>50':
            train_prot_pos += 1
            print('Check for protected sample in positive class in training set.')
        elif train_set[1][i] == 'Male' and train_set[2][i] == '>50':
            train_unprot_pos += 1
            print('Check for unprotected sample in positive class in training set.')


    test_prot_pos = 0
    test_unprot_pos = 0

    for i in range(len(test_set)):
        if test_set[1][i] == 'Female' and test_set[2][i] == '>50':
            test_prot_pos += 1
            print('Check for protected sample in positive class in test set.')
        elif test_set[1][i] == 'Male' and test_set[2][i] == '>50':
            test_unprot_pos += 1
            print('Check for unprotected sample in positive class in test set.')

    total_prot_pos = train_prot_pos + test_prot_pos
    total_unprot_pos = train_unprot_pos + test_unprot_pos

    train_p_rule = (train_prot_pos / train_unprot_pos) * 100
    test_p_rule = (test_prot_pos / test_unprot_pos) * 100
    total_p_rule = (total_prot_pos / total_unprot_pos) * 100

    return metric(test_set[1], pred), [train_p_rule, test_p_rule, total_p_rule]
