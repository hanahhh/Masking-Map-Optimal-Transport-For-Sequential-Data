from aeon.datasets import load_from_arff_file, load_from_tsfile
import numpy as np


def getData(dataset, path="Data/OneDimension/"):
    if dataset in ["ERing", "EyesOpenShut", "JapaneseVowels"]:
        train_path = f"{path}{dataset}/{dataset}_TRAIN.ts"
        test_path = f"{path}{dataset}/{dataset}_TEST.ts"
        X_train, y_train, X_test, y_test = processArffDataFile(
            train_path, test_path, is_ts_file=True)
        return X_train, y_train, X_test, y_test
    else:
        train_path = f"{path}{dataset}/{dataset}_TRAIN.arff"
        test_path = f"{path}{dataset}/{dataset}_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(
            train_path, test_path)
        return X_train, y_train, X_test, y_test


def processArffDataFile(train_path, test_path, is_ts_file=False):
    if is_ts_file == False:
        X_train, y_train = load_from_arff_file(train_path)
        X_test, y_test = load_from_arff_file(test_path)
    else:
        X_train, y_train = load_from_tsfile(train_path)
        X_test, y_test = load_from_tsfile(test_path)
    squeezed_X_train = [np.squeeze(X) for X in X_train]
    squeezed_X_test = [np.squeeze(X) for X in X_test]
    return squeezed_X_train, y_train, squeezed_X_test, y_test
