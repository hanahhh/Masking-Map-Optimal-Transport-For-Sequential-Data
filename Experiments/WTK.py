from sklearn.preprocessing import scale
from sklearn.metrics import pairwise
import numpy as np
import argparse
import random
import ot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from GetData.GetDataMultiDimensions import getDataMultiVariate
from GetData.GetDataOneDimension import getData
import json
from tqdm import tqdm

np.random.seed(42)
random.seed(42)
sklearn_seed = 0


def subsequences_multivariate(time_series, k):
    time_series = np.asarray(time_series)
    n = len(time_series)
    shape = (n - k + 1, k) + time_series.shape[1:]
    if time_series.ndim == 1:
        strides = time_series.strides * 2
    elif time_series.ndim == 2:
        strides = time_series.strides[0], time_series.strides[0], time_series.strides[1]
    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)


def cost_matrix_aw(x, y, subsequence=True):
    if subsequence == True:
        x = np.array(x).reshape(np.array(x).shape[0], -1)
        y = np.array(y).reshape(np.array(y).shape[0], -1)
        C = ot.dist(x, y, metric="euclidean", p=2)
        return C
    else:
        x = np.array(x).reshape(np.array(x).shape[0], -1)
        y = np.array(y).reshape(np.array(y).shape[0], -1)
        C = ot.dist(x, y, metric="euclidean", p=2)
        return C


def wtk(xs, xt, sub_ratio, normalized=False):
    sub_length = int(np.floor(min(len(xs), len(xt)) * sub_ratio))
    s_i = subsequences_multivariate(xs, sub_length)
    s_j = subsequences_multivariate(xt, sub_length)
    if normalized:
        s_i = scale(s_i, axis=1)
        s_j = scale(s_j, axis=1)
    C = ot.dist(s_i, s_j, metric="euclidean")
    return ot.emd2([], [], C)
    # dist = wasserstein_kernel(s_i, s_j)
    # return dist


def wtk_multivariate(xs, xt, sub_ratio=0.1, normalized=False, ot_algo="emd"):
    sub_length = int(np.floor(min(len(xs), len(xt)) * sub_ratio))
    subs_xs = subsequences_multivariate(xs, sub_length)
    subs_xt = subsequences_multivariate(xt, sub_length)
    if normalized:
        subs_xs = scale(subs_xs, axis=1)
        subs_xt = scale(subs_xt, axis=1)
    p = np.ones(len(subs_xs)) / len(subs_xs)
    q = np.ones(len(subs_xt)) / len(subs_xt)
    C = cost_matrix_aw(subs_xs, subs_xt)
    T = 0
    if ot_algo == "emd":
        T, logemd = ot.emd(p, q, C, log=True)
    else:
        T = ot.sinkhorn(p, q, C, log=True)
    return np.sum(T * C)


def knn_wtk_short(X_train, X_test, y_train, y_test, sub_ratio=0.1, normalized=False, k=1):
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric=wtk,
        metric_params={"sub_ratio": sub_ratio, "normalized": normalized},
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_classifier_from_distance_matrix(distance_matrix, k, labels):
    knn_clf = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="precomputed"
    )
    n_train_samples = distance_matrix.shape[1]
    knn_clf.fit(np.random.rand(n_train_samples, n_train_samples), labels)
    predicted_labels = knn_clf.predict(distance_matrix)
    return predicted_labels


def knn_wtk(X_train, X_test, y_train, y_test, sub_ratio=0.1, normalized=False, k=1):
    train_size = len(X_train)
    test_size = len(X_test)
    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = wtk_multivariate(np.array(X_train[train_idx]), np.array(
                X_test[test_idx]), sub_ratio=sub_ratio, normalized=normalized)
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="MultiDimensional")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    return args


def main(args):
    method = "wtk"
    with open("Config/AlgorithmWTK.json", "r") as file:
        algorithms = json.load(file)
    with open(args.data_path, "r") as file:
        data = json.load(file)
    for data_set in data[args.data_type]:
        if args.data_type == "MultiDimensional":
            if data_set in ["MSRAction3D", "MSRDailyActivity3D", "Weizmann", "SpokenArabicDigit", "ArabicCut"]:
                X_train, X_test, y_train, y_test = getDataMultiVariate(
                    data_set)
            else:
                X_train, y_train, X_test, y_test = getData(data_set)
        elif args.data_type == "OneDimensional":
            X_train, y_train, X_test, y_test = getData(data_set)
        best_accuracy = -100
        for k in algorithms[method]["k"]:
            with open(args.result_path, "a") as file:
                file.write(f"{data_set}({k}) ")
            for ratio in algorithms[method]["ratio"]:
                accuracy = -1
                if args.data_type == "MultiDimensional":
                    accuracy = knn_wtk(
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        k=k,
                        sub_ratio=ratio,
                    )
                elif args.data_type == "OneDimensional":
                    accuracy = knn_wtk_short(
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        k=k,
                        sub_ratio=ratio,
                    )
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                with open(
                    args.result_path,
                    "a",
                ) as file:
                    if (
                        ratio
                        == algorithms[method]["ratio"][
                            len(algorithms[method]["ratio"]) - 1
                        ]
                    ):
                        file.write(
                            f"&({ratio}) {round(accuracy, 2)}\\\ \n"
                        )
                    else:
                        file.write(
                            f"&({ratio}) {round(accuracy, 2)} ")
        with open(args.result_path, "a") as file:
            file.write(f"{data_set} best result: {best_accuracy} \n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
