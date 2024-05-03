from aeon.datasets import load_from_arff_file
import numpy as np

def getData(dataset, path):
    if dataset == "BME":
        train_path = f"{path}BME/BME_TRAIN.arff"
        test_path = f"{path}BME/BME_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "BeetleFly":
        train_path = f"{path}BeetleFly/BeetleFly_TRAIN.arff"
        test_path = f"{path}BeetleFly/BeetleFly_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "BirdChicken":
        train_path = f"{path}BirdChicken/BirdChicken_TRAIN.arff"
        test_path = f"{path}BirdChicken/BirdChicken_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "Chinatown":
        train_path = f"{path}Chinatown/Chinatown_TRAIN.arff"
        test_path = f"{path}Chinatown/Chinatown_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "Coffee":
        train_path = f"{path}Coffee/Coffee_TRAIN.arff"
        test_path = f"{path}Coffee/Coffee_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "DistalPhalanxOutlineCorrect":
        train_path = f"{path}DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TRAIN.arff"
        test_path = f"{path}DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "DistalPhalanxTW":
        train_path = f"{path}DistalPhalanxTW/DistalPhalanxTW_TRAIN.arff"
        test_path = f"{path}DistalPhalanxTW/DistalPhalanxTW_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "ECG200":
        train_path = f"{path}ECG200/ECG200_TRAIN.arff"
        test_path = f"{path}ECG200/ECG200_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "FaceFour":
        train_path = f"{path}FaceFour/FaceFour_TRAIN.arff"
        test_path = f"{path}FaceFour/FaceFour_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "Fungi":
        train_path = f"{path}Fungi/Fungi_TRAIN.arff"
        test_path = f"{path}Fungi/Fungi_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "GunPoint":
        train_path = f"{path}GunPoint/GunPoint_TRAIN.arff"
        test_path = f"{path}GunPoint/GunPoint_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "Herring":
        train_path = f"{path}Herring/Herring_TRAIN.arff"
        test_path = f"{path}Herring/Herring_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "ItalyPowerDemand":
        train_path = f"{path}ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff"
        test_path = f"{path}ItalyPowerDemand/ItalyPowerDemand_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "MoteStrain":
        train_path = f"{path}MoteStrain/MoteStrain_TRAIN.arff"
        test_path = f"{path}MoteStrain/MoteStrain_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "OliveOil":
        train_path = f"{path}OliveOil/OliveOil_TRAIN.arff"
        test_path = f"{path}OliveOil/OliveOil_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "Plane":
        train_path = f"{path}Plane/Plane_TRAIN.arff"
        test_path = f"{path}Plane/Plane_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "SmoothSubspace":
        train_path = f"{path}SmoothSubspace/SmoothSubspace_TRAIN.arff"
        test_path = f"{path}SmoothSubspace/SmoothSubspace_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "SonyAIBORobotSurface1":
        train_path = f"{path}SonyAIBORobotSurface1/SonyAIBORobotSurface1_TRAIN.arff"
        test_path = f"{path}SonyAIBORobotSurface1/SonyAIBORobotSurface1_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "SonyAIBORobotSurface2":
        train_path = f"{path}SonyAIBORobotSurface2/SonyAIBORobotSurface2_TRAIN.arff"
        test_path = f"{path}SonyAIBORobotSurface2/SonyAIBORobotSurface2_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "ToeSegmentation2":
        train_path = f"{path}ToeSegmentation2/ToeSegmentation2_TRAIN.arff"
        test_path = f"{path}ToeSegmentation2/ToeSegmentation2_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test
    elif dataset == "BasicMotions":
        train_path = f"{path}BasicMotions/BasicMotions_TRAIN.arff"
        test_path = f"{path}BasicMotions/BasicMotions_TEST.arff"
        X_train, y_train, X_test, y_test = processArffDataFile(train_path, test_path)
        return X_train, y_train, X_test, y_test

def processArffDataFile(train_path, test_path):
    X_train, y_train = load_from_arff_file(train_path)
    squeezed_X_train = [np.squeeze(X) for X in X_train]
    X_test, y_test = load_from_arff_file(test_path)
    squeezed_X_test = [np.squeeze(X) for X in X_test]
    return squeezed_X_train, y_train, squeezed_X_test, y_test