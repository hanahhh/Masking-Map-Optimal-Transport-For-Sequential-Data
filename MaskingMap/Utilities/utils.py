import torch
import numpy as np
import scipy.io as sio
import math
import ot
import matplotlib.pyplot as plt
import seaborn as sns


def cost_matrix_aw(x, y):
    x = np.array(x).reshape(np.array(x).shape[0], -1)
    y = np.array(y).reshape(np.array(y).shape[0], -1)
    C = ot.dist(x, y, metric="euclidean", p=2)
    return C


def draw_matrix_aw(M, type="seaborn"):
    if type == "seaborn":
        sns.heatmap(M, linewidth=0.5)
    else:
        plt.imshow(M, cmap="viridis")
        plt.colorbar()
        plt.show()


def create_mask_linear_aw(xs, xt, lamb):
    n = len(xs)
    m = len(xt)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i > j * n / m - lamb) & (i < j * n / m + lamb):
                M[i][j] = 1
    return M


def create_neighbor_relationship_aw(x):
    xs = np.array(x).reshape(np.array(x).shape[0], -1)
    xt = np.concatenate((np.array([np.zeros_like(xs[0])]), xs), axis=0)[:-1]
    f = xs - xt
    d = np.linalg.norm(f, axis=1)
    f1 = np.cumsum(d)
    sum_dist = f1[len(f1) - 1]
    return f1 / sum_dist


def create_mask_KL_aw(xs, xt, sigma=1, type=1):
    f1 = create_neighbor_relationship(xs)
    f2 = create_neighbor_relationship(xt)
    n = len(f1)
    m = len(f2)
    mid_para = np.sqrt((1 / (n**2) + 1 / (m**2)))
    M = np.abs(np.subtract.outer(f1, f2)) / mid_para
    return np.exp(-(np.power(M, 2)) / 2 * np.power(sigma, 2)) / (
        sigma * np.sqrt(2 * np.pi)
    )


def create_mask_non_linear_aw(xs, xt, ratio=0.1, sigma=1, type=1):
    f1 = create_neighbor_relationship(xs)
    f2 = create_neighbor_relationship(xt)
    n = len(f1)
    m = len(f2)
    mid_para = np.sqrt((1 / (n**2) + 1 / (m**2)))
    KL = np.abs(np.subtract.outer(f1, f2)) / mid_para
    KL = np.exp(-(np.power(KL, 2)) / 2 * np.power(sigma, 2)) / (
        sigma * np.sqrt(2 * np.pi)
    )
    flattened_list = KL.flatten()
    sorted_list = sorted(flattened_list)
    pivot = sorted_list[math.floor((1 - ratio) * len(sorted_list))]
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if KL[i][j] > pivot:
                M[i][j] = 1
    return M


def create_mask_linear(xs, xt, lamb):
    n = len(xs)
    m = len(xt)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i > j * n / m - lamb) & (i < j * n / m + lamb):
                M[i][j] = 1
    return M


def subsequences(time_series, k):
    time_series = np.asarray(time_series)
    n = time_series.size
    shape = (n - k + 1, k)
    strides = time_series.strides * 2

    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)


def subsequence_2d(matrix_list, k):
    subsequences = [matrix_list[i : i + k] for i in range(0, len(matrix_list) - k)]
    return subsequences


def subsequence_2d_without_overlap(matrix_list, k):
    subsequences = [matrix_list[i : i + k] for i in range(0, len(matrix_list) - k, k)]
    return subsequences


def cost_matrix(x, y):
    x, y = np.array(x), np.array(y)
    Cxy = (
        np.sum(x**2, axis=1).reshape(-1, 1)
        + np.sum(y**2, axis=1).reshape(1, -1)
        - 2 * np.dot(x, y.T)
    )
    return Cxy


def cost_matrix_2d(x, y):
    m = len(x)
    n = len(y)
    Cxy = np.zeros((m, n))
    for row in range(m):
        for col in range(n):
            Cxy[row, col] = np.linalg.norm(x[row] - y[col])
    return Cxy


def create_mask_KL(xs, xt, sigma=1, type=1):
    f1 = create_neighbor_relationship(xs)
    f2 = create_neighbor_relationship(xt)
    n = len(f1)
    m = len(f2)
    mid_para = np.sqrt((1 / (n**2) + 1 / (m**2)))
    M = np.abs(np.subtract.outer(f1, f2)) / mid_para
    # M = np.zeros((n, m))
    # for i in range(0, n):
    #     for j in range(0, m):
    #         if type == 1:
    #             if f1[i] == f2[j]:
    #                 M[i][j] = 1
    #             else:
    #                 M[i][j] = max(f1[i], f2[j])/min(f1[i], f2[j])
    #         else:
    #             M[i][j] = np.abs(f1[i] - f2[j])/mid_para
    return np.exp(-(np.power(M, 2)) / 2 * np.power(sigma, 2)) / (
        sigma * np.sqrt(2 * np.pi)
    )


def create_I(n, m):
    a_n = np.arange(start=1, stop=n + 1)
    b_m = np.arange(start=1, stop=m + 1)
    row_col_matrix = np.meshgrid(a_n, b_m)
    row = row_col_matrix[0].T / n
    col = row_col_matrix[1].T / m
    I = 1 / ((row - col) ** 2 + 1)
    return I


def create_mask_binary(C, k=0.1):
    n, m = C.shape
    flattened_list = C.flatten()
    sorted_list = sorted(flattened_list)
    pivot = sorted_list[math.floor((1 - k) * len(sorted_list))]
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if C[i][j] > pivot:
                M[i][j] = 1
    return M


def cost_matrix_1d(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(1, -1)
    Cxy = (x - y) ** 2
    return Cxy


def create_mask(C, ratio):
    n, m = C.shape
    lamb = ratio * min(n, m)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i > j * n / m - lamb) & (i < j * n / m + lamb):
                M[i][j] = 1
    return M


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)))
    return f_x


def softmax_matrix(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
    return f_x


def KL_matrix(p, q, eps=1e-10):
    return np.sum(p * np.log(p + eps) - p * np.log(q + eps), axis=-1)


def JS_matrix(P, Q, eps=1e-10):
    P = np.expand_dims(P, axis=1)
    Q = np.expand_dims(Q, axis=0)
    kl1 = KL_matrix(P, (P + Q) / 2, eps)
    kl2 = KL_matrix(Q, (P + Q) / 2, eps)
    return 0.5 * (kl1 + kl2)


def create_neighbor_relationship(xs):
    xs = np.array(xs)
    if xs.ndim == 1:
        xt = np.insert(xs, 0, np.zeros_like(xs[0]))[:-1]
        f = xs - xt
        f = f.reshape(-1, 1)
    else:
        xt = np.vstack((np.zeros_like(xs[0]), xs))[:-1]
        f = xs - xt
    d = np.linalg.norm(f, axis=1)
    f1 = np.cumsum(d)
    sum_dist = f1[len(f1) - 1]
    return f1 / sum_dist


def create_mask_DT(xs, xt, lamb):
    f1 = create_neighbor_relationship(xs)
    f2 = create_neighbor_relationship(xt)
    n = len(f1)
    m = len(f2)
    M = np.zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            if (i > j * n / m - lamb) & (i < j * n / m + lamb):
                if f1[i] == f2[j]:
                    M[i][j] = 1
                # elif np.abs(min(f1[i],f2[j])/max(f1[i],f2[j])) > lamb:
                #     M[i][j] = 1
                else:
                    M[i][j] = np.abs(min(f1[i], f2[j]) / max(f1[i], f2[j]))
    return M
