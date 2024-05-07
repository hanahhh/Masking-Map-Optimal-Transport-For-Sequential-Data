import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from MaskingMap.MaskingMapLinear import (
    masking_map_linear,
    masking_map_linear_sequence,
    masking_map_linear_partial,
    masking_map_linear_sub_sequence,
    masking_map_linear_sub_sequence_multivariate
)
from MaskingMap.MaskingMapNonLinear import masking_map_non_linear
from MaskingMap.MaskingMapAutoWeighted import masking_map_auto_weighted
import ot


def knn_classifier_from_distance_matrix(distance_matrix, k, labels):
    knn_clf = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="precomputed"
    )
    n_train_samples = distance_matrix.shape[1]
    knn_clf.fit(np.random.rand(n_train_samples, n_train_samples), labels)
    predicted_labels = knn_clf.predict(distance_matrix)
    return predicted_labels


def knn_masking_map_linear(X_train, X_test, y_train, y_test, ratio=0.1, k=1):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = masking_map_linear(
                X_train[train_idx], X_test[test_idx], ratio=ratio
            )
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_linear_short(X_train, X_test, y_train, y_test, lamb=5, k=1):
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric=masking_map_linear,
        metric_params={"lamb": lamb},
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_linear_subsequence(
    X_train, X_test, y_train, y_test, lamb=5, sub_length=25, k=1
):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = masking_map_linear_sequence(
                X_train[train_idx], X_test[test_idx], sub_length=sub_length, lamb=lamb
            )
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_linear_subsequence_short(
    X_train, X_test, y_train, y_test, lamb=5, sub_length=25, k=1
):
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric=masking_map_linear_sequence,
        metric_params={"lamb": lamb, "sub_length": sub_length},
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_linear_subsequence_sklearn(
    X_train, X_test, y_train, y_test, ratio=0.1, sub_ratio=0.1, k=1
):

    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric=masking_map_linear_sub_sequence,
        metric_params={"ratio": ratio, "sub_ratio": sub_ratio},
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_linear_subsequence_multivariate(X_train, X_test, y_train, y_test, ratio=0.1, sub_ratio=0.1, k=1):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = masking_map_linear_sub_sequence_multivariate(
                X_train[train_idx], X_test[test_idx], ratio=ratio, sub_ratio=sub_ratio
            )
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_linear_partial(X_train, X_test, y_train, y_test, lamb=5, k=1):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = masking_map_linear_partial(
                X_train[train_idx], X_test[test_idx], lamb=lamb
            )
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_linear_partial_short(X_train, X_test, y_train, y_test, lamb=5, k=1):
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric=masking_map_linear_partial,
        metric_params={"lamb": lamb},
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_non_linear(X_train, X_test, y_train, y_test, ratio=0.1, k=1):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = masking_map_non_linear(
                X_train[train_idx], X_test[test_idx], ratio=ratio
            )
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_non_linear_short(X_train, X_test, y_train, y_test, ratio=0.1, k=1):
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric=masking_map_non_linear,
        metric_params={"ratio": ratio},
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_auto_weighted(
    X_train,
    X_test,
    y_train,
    y_test,
    ratio=0.1,
    lamb=3,
    k=1,
    algorithm="linear_programming",
):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = masking_map_auto_weighted(
                X_train[train_idx],
                X_test[test_idx],
                ratio=ratio,
                lamb=lamb,
                algorithm=algorithm,
            )
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_masking_map_auto_weighted_short(
    X_train, X_test, y_train, y_test, ratio=0.1, lamb=3, k=1
):
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric=masking_map_auto_weighted,
        metric_params={"ratio": ratio, "lamb": lamb},
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
