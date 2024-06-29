import math

import matplotlib.pyplot as plt
import numpy as np

from MaskingMap.Utilities.linearprog import lp_partial
from MaskingMap.Utilities.sinkhorn import sinkhorn_log_domain
from MaskingMap.Utilities.utils import (cost_matrix, cost_matrix_1d,
                                        cost_matrix_aw, create_mask_binary,
                                        create_mask_KL,
                                        create_mask_KL_subsequence,
                                        subsequence_2d)


def masking_map_non_linear(xs, xt, ratio=0.1, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", plot=False):
    '''
    Parameters
    ----------
        a: ndarray, (m,d)
           d-dimensional source samples
        b: ndarray, (n,d) 
           d-dimensional target samples
        lamb: lambda, int 
           Adjust the diagonal width. Default is 3
        sub_length: int
                    The number of elements of sub-sequence. Default is 25
        algorithm: str
                   algorithm to solve model. Default is "linear_programming". Choices should be
                   "linear_programming" and "sinkhorn"
        plot: bool
              status for plot the optimal transport matrix or not. Default is "False"
    Returns
    ------- 
        cost: Transportation cost
    '''
    p = np.ones(len(xs))/len(xs)
    q = np.ones(len(xt))/len(xt)

    # mask matrix
    if xs.ndim == 1:
        C = cost_matrix_1d(xs, xt)
    elif xs.ndim == 2:
        C = cost_matrix(xs, xt)
    else:
        raise ValueError("The data must in the form of 1d or 2d array")
    C /= (C.max() + eps)
    KL = create_mask_KL(xs, xt, type=2)
    M_hat = create_mask_binary(KL, ratio)
    # solving model
    if algorithm == "linear_programming":
        pi = lp_partial(p, q, C, M_hat)
    elif algorithm == "sinkhorn":
        pi = sinkhorn_log_domain(
            p, q, C, M_hat, reg, max_iterations, thres)
    else:
        raise ValueError(
            "algorithm must be 'linear_programming' or 'sinkhorn'!")

    cost = np.sum(pi * C)
    if plot:
        plt.imshow(pi, cmap='viridis')
        plt.colorbar()
        plt.show()
        return pi, cost
    return cost


def auto_weighted_masking_map(xs, xt, r1=1, r2=1, eps=1e-10, algorithm="sinkhorn"):
    D1_cost = D1(xs, xt)
    D2_cost = D2(xs, xt, type=1)
    C = D1_cost + D2_cost
    old_w1 = -100
    w1, w2 = 1/2, 1/2
    T = np.zeros((len(xs), len(xt)))
    iteration_num = 0
    while w1 - old_w1 > eps:
        M1 = create_mask_linear(xs=xs, xt=xt, ratio=r1 * w1)
        M2 = create_mask_non_linear(D2_cost, ratio=r2 * w2)
        iteration_num += 1
        M_ = w1*M1 + w2*M2
        M = (M_ >= 1).astype(int)
        # draw_multiple_matrices([M1, M2, M])
        p = np.ones(len(xs))/len(xs)
        q = np.ones(len(xt))/len(xt)
        if algorithm == "linear_programming":
            T = lp(p=p, q=q, C=C, Mask=M)
        elif algorithm == "sinkhorn":
            T = sinkhorn_log_domain(p=p, q=q, C=C, Mask=M)
        else:
            raise ValueError(
                "algorithm must be 'linear_programming' or 'sinkhorn'!")
        old_w1 = w1
        w1 = 1/(2*math.sqrt(np.sum((M1*T)*D1_cost)))
        w2 = 1/(2*math.sqrt(np.sum((M2*T)*D2_cost)))
    cost = w1*np.sum((M1*T)*C) + w2*np.sum((M2*T)*C)
    return cost


def create_neighbor_relationship(x):
    xs = np.array(x).reshape(np.array(x).shape[0], -1)
    xt = np.concatenate((np.array([np.zeros_like(xs[0])]), xs), axis=0)[:-1]
    f = xs - xt
    d = np.linalg.norm(f, axis=1)
    f1 = np.cumsum(d)
    sum_dist = f1[len(f1)-1]
    return f1/sum_dist


def D1(x, y):
    n = len(x)
    m = len(y)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i][j] = 1/(np.square(i/n - j/m) + 1)
    return M


def D2(xs, xt, sigma=1, type=1):
    n = len(xs)
    m = len(xt)
    M = np.zeros((n, m))
    f1 = create_neighbor_relationship(xs)
    f2 = create_neighbor_relationship(xt)
    if (type == 1):
        for i in range(n):
            for j in range(m):
                if (f1[i] == f2[j]):
                    M[i][j] = 1
                else:
                    M[i][j] = min(f1[i], f2[j])/max(f1[i], f2[j])
        return M
    elif (type == 2):
        n = len(f1)
        m = len(f2)
        mid_para = np.sqrt((1/(n**2) + 1/(m**2)))
        M = np.abs(np.subtract.outer(f1, f2)) / mid_para
        return np.exp(-(np.power(M, 2)) / 2 * np.power(sigma, 2)) / (sigma * np.sqrt(2 * np.pi))


def create_mask_linear(xs, xt, ratio):
    n = len(xs)
    m = len(xt)
    lamb = ratio * min(n, m)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i > j*n/m - lamb) & (i < j*n/m + lamb):
                M[i][j] = 1
    return M


def create_mask_non_linear(KL, ratio=0.1, sigma=1, type=1):
    n, m = KL.shape
    flattened_list = KL.flatten()
    sorted_list = sorted(flattened_list)
    pivot = sorted_list[math.floor((1-ratio)*len(sorted_list))]
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if KL[i][j] > pivot:
                M[i][j] = 1
    return M


def masking_map_non_linear_subsequence(xs, xt, ratio=0.1, eps=1e-10, reg=0.0001, max_iterations=100000, thres=1e-5, algorithm="linear_programming", plot=False, sub_ratio=0.1):
    '''
    Parameters
    ----------
        a: ndarray, (m,d)
           d-dimensional source samples
        b: ndarray, (n,d) 
           d-dimensional target samples
        lamb: lambda, int 
           Adjust the diagonal width. Default is 3
        sub_length: int
                    The number of elements of sub-sequence. Default is 25
        algorithm: str
                   algorithm to solve model. Default is "linear_programming". Choices should be
                   "linear_programming" and "sinkhorn"
        plot: bool
              status for plot the optimal transport matrix or not. Default is "False"
    Returns
    ------- 
        cost: Transportation cost
    '''
    sub_length = int(np.floor(min(len(xs), len(xt)) * sub_ratio))
    subs_xs = subsequence_2d(xs, sub_length)
    subs_xt = subsequence_2d(xt, sub_length)
    p = np.ones(len(subs_xs)) / len(subs_xs)
    q = np.ones(len(subs_xt)) / len(subs_xt)

    # mask matrix
    C = cost_matrix_aw(subs_xs, subs_xt)
    C /= C.max() + eps
    KL = create_mask_KL_subsequence(subs_xs, subs_xt, type=2)
    M_hat = create_mask_binary(KL, ratio)
    # solving model
    if algorithm == "linear_programming":
        pi = lp_partial(p, q, C, M_hat)
    elif algorithm == "sinkhorn":
        pi = sinkhorn_log_domain(
            p, q, C, M_hat, reg, max_iterations, thres)
    else:
        raise ValueError(
            "algorithm must be 'linear_programming' or 'sinkhorn'!")

    cost = np.sum(pi * C)
    if plot:
        plt.imshow(pi, cmap='viridis')
        plt.colorbar()
        plt.show()
        return pi, cost
    return cost
