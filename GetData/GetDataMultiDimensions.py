import numpy as np
import joblib
import sys
import os

NUM_MFCC = 13
ALL_COEFFS = np.arange(0, NUM_MFCC, 1)
NUM_DIGITS = 10
TRAIN_FILE = os.path.join('../data/UCR/raw_data/arabic/Train_Arabic_Digit.txt')
TEST_FILE = os.path.join('../data/UCR/raw_data/arabic/Test_Arabic_Digit.txt')


def load_data(filepath, coeffs=ALL_COEFFS):
    mask = convert_list_to_mask(coeffs)

    digits = []
    labels = []
    current_digit = []
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            if line.isspace() or line == "\n":
                digits = add_digit(digits, current_digit)
                current_digit = []
            else:
                mfcc = list(map(float, line.split(' ')))
                mfcc = np.asarray(mfcc)
                mfcc_filtered = mfcc[mask]
                current_digit.append(mfcc_filtered)
        if len(current_digit):
            digits = add_digit(digits, current_digit)

    # Should be 660 for train, 220 for test
    num_entries_per_digit = int(len(digits) / 10)
    for digit_value in range(NUM_DIGITS):
        for i in range(num_entries_per_digit):
            labels.append(digit_value)

    return digits, np.asarray(labels)


def add_digit(digits, current_digit):
    """
    :param digits: List of digits, which are each a numpy array
    :param current_digit: Current digit, which is a list of the coefficients for the digit
    :return:
    Formats and reshapes the current_digit list as a numpy array and appends it to digits and returns digits
    """
    digit_matrix = np.asarray(current_digit)
    digit_matrix = np.reshape(digit_matrix, (-1, len(current_digit[0])))
    digits.append(digit_matrix)
    return digits


def get_train_data(train_file, coeffs=ALL_COEFFS):
    return load_data(train_file, coeffs)


def get_test_data(test_file, coeffs=ALL_COEFFS):
    return load_data(test_file, coeffs)


def convert_list_to_mask(list_of_indices):
    mask = [False] * NUM_MFCC
    for index in list_of_indices:
        assert (0 <= index < NUM_MFCC)
        mask[index] = True
    return mask


if __name__ == "__main__":
    coeffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    test_digits, test_labels = get_test_data()
    assert (len(test_digits[0][0]) == len(coeffs))


def getDataMultiVariate(dataset):
    if dataset in ["Weizmann"]:
        directory = f"Data/MultiDimensions/{dataset}/binary"
        X = []
        y = np.empty(80, dtype='<U9')
        index = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                X.append(joblib.load(f))
                y[index] = filename.split('_')[1]
                index = index + 1
        return X[0:40], y[0:40], X[40:80], y[40:80]
    else:
        folder_path = f"Data/MultiDimensions/{dataset}/"
        X_train = joblib.load(os.path.join(folder_path, "X_train.pkl"))
        y_train = joblib.load(os.path.join(folder_path, "X_test.pkl"))
        X_test = joblib.load(os.path.join(folder_path, "y_train.pkl"))
        y_test = joblib.load(os.path.join(folder_path, "y_test.pkl"))
        return X_train, X_test, y_train, y_test
