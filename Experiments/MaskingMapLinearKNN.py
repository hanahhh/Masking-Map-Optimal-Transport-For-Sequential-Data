import argparse
import json
import random
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from GetData.GetDataOneDimension import getData
from GetData.GetDataMultiDimensions import getDataMultiVariate
from MaskingMap.knn import (
    knn_masking_map_linear,
    knn_masking_map_linear_subsequence,
    knn_masking_map_linear_partial,
    knn_masking_map_non_linear,
    knn_masking_map_linear_subsequence_sklearn,
    knn_masking_map_linear_short
)

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
    method = "masking_map_linear"
    with open("Config/AlgorithmLinear.json", "r") as file:
        algorithms = json.load(file)
    with open(args.data_path, "r") as file:
        data = json.load(file)
    for data_set in data["MultiDimensional"]:
        X_train, X_test, y_train, y_test = getDataMultiVariate(data_set)
        best_accuracy = -100
        for k in algorithms[method]["k"]:
            with open(args.result_path, "a") as file:
                file.write(f"{data_set}({k}) ")
            for ratio in algorithms[method]["ratio"]:
                accuracy = knn_masking_map_linear(
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

    # if "masking_map_linear" in algorithms:
    #     for data_type in algorithms["masking_map_linear"]["data"]:
    #         for data_set in data[data_type]:
    #             best_accuracy = -100
    #             for ratio in algorithms["masking_map_linear"]["ratio"]:
    #                 with open(
    #                     "Results/ExperimentMaskingMapLinearOneDimensional.txt", "a"
    #                 ) as file:
    #                     file.write(f"{data_set}({ratio}) ")
    #                 for k in algorithms["masking_map_linear"]["k"]:
    #                     if data_type == "OneDimensional":
    #                         X_train, y_train, X_test, y_test = getData(
    #                             data_set, "Data/OneDimension/"
    #                         )
    #                         accuracy = knn_masking_map_linear(
    #                             X_train=X_train,
    #                             X_test=X_test,
    #                             y_train=y_train,
    #                             y_test=y_test,
    #                             k=k,
    #                             ratio=ratio,
    #                         )
    #                         if accuracy > best_accuracy:
    #                             best_accuracy = accuracy
    #                         with open(
    #                             "Results/ExperimentMaskingMapLinearOneDimensional.txt",
    #                             "a",
    #                         ) as file:
    #                             if (
    #                                 k
    #                                 == algorithms["masking_map_linear"]["k"][
    #                                     len(algorithms["masking_map_linear"]["k"]) - 1
    #                                 ]
    #                             ):
    #                                 file.write(f"&{round(accuracy, 2)}\\\ \n")
    #                             else:
    #                                 file.write(f"&{round(accuracy, 2)} ")
    #             with open(
    #                 "Results/ExperimentMaskingMapLinearOneDimensional.txt", "a"
    #             ) as file:
    #                 file.write(f"{data_set} best result: {best_accuracy} \n")
    # if "masking_map_linear_subsequence" in algorithms:
    #     for data_type in algorithms["masking_map_linear_subsequence"]["data"]:
    #         for data_set in data[data_type]:
    #             if data_type == "OneDimensional":
    #                 X_train, y_train, X_test, y_test = getData(
    #                     data_set, "Data/OneDimension/"
    #                 )
    #             best_accuracy = -100
    #             for k in algorithms["masking_map_linear_subsequence"]["k"]:
    #                 with open(
    #                     "Results/ExperimentMaskingMapLinearSubsequence.txt", "a"
    #                 ) as file:
    #                     file.write(f"{data_set}({k}) ")
    #                 for ratio in algorithms["masking_map_linear_subsequence"]["ratio"]:
    #                     for sub_ratio in algorithms["masking_map_linear_subsequence"][
    #                         "sub_ratio"
    #                     ]:
    #                         accuracy = knn_masking_map_linear_subsequence_sklearn(
    #                             X_train=X_train,
    #                             X_test=X_test,
    #                             y_train=y_train,
    #                             y_test=y_test,
    #                             k=k,
    #                             ratio=ratio,
    #                             sub_ratio=sub_ratio,
    #                         )
    #                         if accuracy > best_accuracy:
    #                             best_accuracy = accuracy
    #                         with open(
    #                             "Results/ExperimentMaskingMapLinearSubsequence.txt",
    #                             "a",
    #                         ) as file:
    #                             if (
    #                                 ratio
    #                                 == algorithms["masking_map_linear_subsequence"][
    #                                     "ratio"
    #                                 ][
    #                                     len(
    #                                         algorithms[
    #                                             "masking_map_linear_subsequence"
    #                                         ]["ratio"]
    #                                     )
    #                                     - 1
    #                                 ]
    #                                 and sub_ratio
    #                                 == algorithms["masking_map_linear_subsequence"][
    #                                     "sub_ratio"
    #                                 ][
    #                                     len(
    #                                         algorithms[
    #                                             "masking_map_linear_subsequence"
    #                                         ]["sub_ratio"]
    #                                     )
    #                                     - 1
    #                                 ]
    #                             ):
    #                                 file.write(
    #                                     f"&({ratio};{sub_ratio}) {round(accuracy, 2)}\\\ \n"
    #                                 )
    #                             else:
    #                                 file.write(
    #                                     f"&({ratio};{sub_ratio}) {round(accuracy, 2)} "
    #                                 )
    #             with open(
    #                 "Results/ExperimentMaskingMapLinearSubsequence.txt", "a"
    #             ) as file:
    #                 file.write(f"{data_set} best result: {best_accuracy} \n")
    # if "masking_map_linear_partial" in algorithms:
    #     for k in algorithms["masking_map_linear_partial"]["k"]:
    #         for lamb in algorithms["masking_map_linear_partial"]["lambda"]:
    #             for data_type in algorithms["masking_map_linear_partial"]["data"]:
    #                 for data_set in data[data_type]:
    #                     if data_type == "OneDimensional":
    #                         X_train, y_train, X_test, y_test = getData(
    #                             data_set, "Data/OneDimension/"
    #                         )
    #                         with open(
    #                             "Results/ExperimentMaskingMapLinearPartial.txt", "a"
    #                         ) as file:
    #                             file.write(f"{data_set}({k}) ")
    #                         accuracy = knn_masking_map_linear_partial(
    #                             X_train=X_train,
    #                             X_test=X_test,
    #                             y_train=y_train,
    #                             y_test=y_test,
    #                             k=k,
    #                             lamb=lamb,
    #                         )
    #                         with open(
    #                             "Results/ExperimentMaskingMapLinearPartial.txt", "a"
    #                         ) as file:
    #                             if (
    #                                 lamb
    #                                 == algorithms["masking_map_linear_partial"][
    #                                     "lambda"
    #                                 ][
    #                                     len(
    #                                         algorithms["masking_map_linear_partial"][
    #                                             "lambda"
    #                                         ]
    #                                     )
    #                                     - 1
    #                                 ]
    #                             ):
    #                                 file.write(f"&{round(accuracy, 2)}\\\ \n")
    #                             else:
    #                                 file.write(f"&{round(accuracy, 2)} ")

    # if "masking_map_non_linear" in algorithms:
    #     for data_type in algorithms["masking_map_non_linear"]["data"]:
    #         for data_set in data[data_type]:
    #             for ratio in algorithms["masking_map_non_linear"]["ratio"]:
    #                 with open(
    #                     "Results/ExperimentMaskingMapNonLinearOneDimensional.txt", "a"
    #                 ) as file:
    #                     file.write(f"{data_set}({ratio}) ")
    #                 for k in algorithms["masking_map_non_linear"]["k"]:
    #                     if data_type == "OneDimensional":
    #                         X_train, y_train, X_test, y_test = getData(
    #                             data_set, "Data/OneDimension/"
    #                         )
    #                         accuracy = knn_masking_map_non_linear(
    #                             X_train=X_train,
    #                             X_test=X_test,
    #                             y_train=y_train,
    #                             y_test=y_test,
    #                             k=k,
    #                             ratio=ratio,
    #                         )
    #                         with open(
    #                             "Results/ExperimentMaskingMapNonLinearOneDimensional.txt",
    #                             "a",
    #                         ) as file:
    #                             if (
    #                                 k
    #                                 == algorithms["masking_map_non_linear"]["k"][
    #                                     len(algorithms["masking_map_non_linear"]["k"])
    #                                     - 1
    #                                 ]
    #                             ):
    #                                 file.write(f"&{round(accuracy, 2)}\\\ \n")
    #                             else:
    #                                 file.write(f"&{round(accuracy, 2)} ")


if __name__ == "__main__":
    args = parse_args()
    main(args)
