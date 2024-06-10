import argparse
import json
import random
import numpy as np

from GetData.GetDataOneDimension import getData
from GetData.GetDataMultiDimensions import getDataMultiVariate
from MaskingMap.knn import (
    knn_masking_map_non_linear_subsequence,
    knn_masking_map_non_linear_subsequence_short
)

np.random.seed(42)
random.seed(42)
sklearn_seed = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--type", type=str, default="OneDimensional")
    args = parser.parse_args()
    return args


def main(args):
    result_link = args.result_path
    method = "masking_map_nonlinear_subsequence"
    with open("Config/AlgorithmNonlinearSubsequence.json", "r") as file:
        algorithms = json.load(file)
    with open(args.data_path, "r") as file:
        data = json.load(file)
    for data_set in data:
        if data_set in ["MSRAction3D", "MSRDailyActivity3D", "Weizmann", "SpokenArabicDigit", "ArabicCut"]:
            X_train, X_test, y_train, y_test = getDataMultiVariate(data_set)
        else:
            X_train, y_train, X_test, y_test = getData(data_set)
        best_accuracy = -100
        for k in algorithms[method]["k"]:
            with open(result_link, "a") as file:
                file.write(f"{data_set}({k}) ")
            for ratio in algorithms[method]["ratio"]:
                for sub_ratio in algorithms[method]["sub_ratio"]:
                    accuracy = -1
                    if args.type == "OneDimensional":
                        accuracy = knn_masking_map_non_linear_subsequence_short(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            k=k,
                            ratio=ratio,
                            sub_ratio=sub_ratio,
                        )
                    else:
                        accuracy = knn_masking_map_non_linear_subsequence(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            k=k,
                            ratio=ratio,
                            sub_ratio=sub_ratio,
                        )
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                    with open(
                        result_link,
                        "a",
                    ) as file:
                        if (
                            ratio
                            == algorithms[method]["ratio"][
                                len(algorithms[method]["ratio"]) - 1
                            ]
                            and sub_ratio
                            == algorithms[method]["sub_ratio"][
                                len(algorithms[method]["sub_ratio"]) - 1
                            ]
                        ):
                            file.write(
                                f"&({ratio};{sub_ratio}) {round(accuracy, 2)}\\\ \n"
                            )
                        else:
                            file.write(
                                f"&({ratio};{sub_ratio}) {round(accuracy, 2)} ")
        with open(result_link, "a") as file:
            file.write(f"{data_set} best result: {best_accuracy} \n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
