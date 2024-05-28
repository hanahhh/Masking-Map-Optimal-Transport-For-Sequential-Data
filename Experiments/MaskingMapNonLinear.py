import argparse
import json
import random
import numpy as np
from GetData.GetDataMultiDimensions import getDataMultiVariate
from GetData.GetDataOneDimension import getData
from MaskingMap.knn import knn_masking_map_non_linear

np.random.seed(42)
random.seed(42)
sklearn_seed = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    return args


def main(args):
    method = "masking_map_non_linear"
    with open("Config/AlgorithmNonLinear.json", "r") as file:
        algorithms = json.load(file)
    with open(args.data_path, "r") as file:
        data = json.load(file)
    for data_set in data["MultiDimensional"]:
        if data_set in ["MSRAction3D", "MSRDailyActivity3D", "Weizmann", "SpokenArabicDigit", "ArabicCut"]:
            X_train, X_test, y_train, y_test = getDataMultiVariate(data_set)
        else:
            X_train, y_train, X_test, y_test = getData(data_set)
        best_accuracy = -100
        for k in algorithms[method]["k"]:
            with open(args.result_path, "a") as file:
                file.write(f"{data_set}({k}) ")
            for ratio in algorithms[method]["ratio"]:
                accuracy = knn_masking_map_non_linear(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
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
