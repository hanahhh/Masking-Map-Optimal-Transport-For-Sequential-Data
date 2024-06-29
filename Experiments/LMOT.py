import argparse
import json
import random

import numpy as np

from GetData.GetData import getData, getListMultivariateDataset
from MaskingMap.MOT_MSOT_Knn import knn_LMOT

np.random.seed(42)
random.seed(42)
sklearn_seed = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.algorithm_path, "r") as file:
        algorithms = json.load(file)
    with open(args.data_path, "r") as file:
        data = json.load(file)
    multivariate_datasets = getListMultivariateDataset()
    for data_set in data:
        if (data_set in multivariate_datasets):
            data_type = "multivariate"
        else:
            data_type = "univariate"
        X_train, y_train, X_test, y_test = getData(data_set)
        best_accuracy = -1
        for k in algorithms["k"]:
            with open(args.result_path, "a") as file:
                file.write(f"{data_set}({k}) ")
            for ratio in algorithms["ratio"]:
                accuracy = knn_LMOT(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    type=data_type,
                    k=k,
                    ratio=ratio,
                )
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                with open(
                    args.result_path,
                    "a",
                ) as file:
                    if (
                        ratio
                        == algorithms["ratio"][
                            len(algorithms["ratio"]) - 1
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
