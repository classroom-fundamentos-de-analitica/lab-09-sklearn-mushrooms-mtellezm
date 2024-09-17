"""GitHub Classroom autograding script."""


import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def load_estimator():
    """Load trained model from disk."""

    if not os.path.exists("model.pkl"):
        return None
    with open("model.pkl", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def load_datasets():
    """Load train and test datasets."""

    train_dataset = pd.read_csv("train_dataset.csv")
    test_dataset = pd.read_csv("test_dataset.csv")

    encoder = OneHotEncoder()

    x_train = train_dataset.drop("type", axis=1)
    x_train = encoder.fit_transform(x_train)
    y_train = train_dataset["type"]

    x_test = test_dataset.drop("type", axis=1)
    x_test = encoder.fit_transform(x_test)
    y_test = test_dataset["type"]

    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):
    """Evaluate model performance."""

    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def compute_metrics():
    """Compute model metrics."""

    estimator = load_estimator()
    assert estimator is not None, "Model not found"
    

def run_grading(accuracy_train, accuracy_test):
    """Run grading script."""

    assert accuracy_train > 0.99
    assert accuracy_test > 0.99


if __name__ == "__main__":
    estimator = load_estimator()
    x_train, x_test, y_train, y_test = load_datasets()

    y_pred_train = estimator.predict(x_train)
    y_pred_test = estimator.predict(x_test)

    accuracy_train = eval_metrics(y_train, y_pred_train)
    accuracy_test = eval_metrics(y_test, y_pred_test)
    print(accuracy_test, accuracy_test)

    run_grading(accuracy_train, accuracy_test)
