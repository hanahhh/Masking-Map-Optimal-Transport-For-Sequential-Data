from aeon.datasets import load_from_arff_file, load_from_tsfile
import numpy as np
import joblib
import os


def getData(dataset, path="Data/"):
    if dataset in ["ERing", "EyesOpenShut", "JapaneseVowels"]:
        train_path = f"{path}{dataset}/{dataset}_TRAIN.ts"
        test_path = f"{path}{dataset}/{dataset}_TEST.ts"
        X_train, y_train, X_test, y_test = processArffDataFile(
            train_path, test_path, is_ts_file=True)
        return X_train, y_train, X_test, y_test
    elif dataset in ["MSRAction3D", "MSRDailyActivity3D", "Weizmann", "SpokenArabicDigit", "ArabicCut"]:
        dir_name = os.path.dirname(data_path)
        X_train = joblib.load(os.path.join(dir_name, f"{dataset}/X_train.pkl"))
        y_train = joblib.load(os.path.join(dir_name, f"{dataset}/y_train.pkl"))
        X_test = joblib.load(os.path.join(dir_name, f"{dataset}/X_test.pkl"))
        y_test = joblib.load(os.path.join(dir_name, f"{dataset}/y_test.pkl"))
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


def getListMultivariateDataset():
    with open("Data/ListMultiDimensional.txt", 'r', encoding='utf-8') as file:
        multivariate_datasets = file.readlines()
    multivariate_datasets = [dataset.strip()
                             for dataset in multivariate_datasets]
    return multivariate_datasets


def getListUnivariateDataset():
    with open("Data/ListOneDimensional.txt", 'r', encoding='utf-8') as file:
        univariate_datasets = file.readlines()
    univariate_datasets = [dataset.strip() for dataset in univariate_datasets]
    return univariate_datasets
