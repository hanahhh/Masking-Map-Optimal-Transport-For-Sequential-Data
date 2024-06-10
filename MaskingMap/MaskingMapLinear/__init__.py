import numpy as np
import torch
import math
import seaborn as sns
from MaskingMap.Utilities.linearprog import lp, lp_partial, lp_sci, lp_aw
from MaskingMap.Utilities.sinkhorn import (
    sinkhorn_log_domain,
    sinkhorn_log_domain_torch,
    sinkhorn_log_domain_aw,
)
from MaskingMap.Utilities.utils import (
    cost_matrix,
    cost_matrix_1d,
    create_mask,
    subsequences,
    subsequenceData,
    subsequence_2d,
    cost_matrix_2d,
    create_mask_DT,
    create_mask_KL,
    create_I,
    create_mask_binary,
    cost_matrix_aw,
    create_mask_linear_aw,
    create_mask_non_linear_aw,
)
from ot.lp import emd_1d_sorted
import matplotlib.pyplot as plt


def masking_map_linear(
    xs,
    xt,
    ratio=0.1,
    eps=1e-10,
    reg=0.0001,
    max_iterations=100000,
    thres=1e-5,
    algorithm="linear_programming",
    plot=False,
):
    """
    Parameters
    ----------
        xs: ndarray, (m,d)
            d-dimensional source samples
        xt: ndarray, (n,d)
            d-dimensional target samples
        lamb: lambda, int
            Adjust the diagonal width. Default is 3
        algorithm: str
            algorithm to solve model. Default is "linear_programming". Choices should be
            "linear_programming" and "sinkhorn"
        plot: bool
            status for plot the optimal transport matrix or not. Default is "False"
    Returns
    -------
        cost: Transportation cost
    """
    C = cost_matrix_aw(xs, xt, subsequence=False)
    C /= C.max() + eps
    p = np.ones(len(xs)) / len(xs)
    q = np.ones(len(xt)) / len(xt)
    # mask matrix
    M = create_mask(C, ratio)

    # solving model
    if algorithm == "linear_programming":
        # pi = lp(p, q, C, M)
        pi = lp_partial(p, q, C, M)
    elif algorithm == "sinkhorn":
        pi = sinkhorn_log_domain(p, q, C, M, reg, max_iterations, thres)
    else:
        raise ValueError(
            "algorithm must be 'linear_programming' or 'sinkhorn'!")

    cost = np.sum(pi * C)
    # cost = np.exp(-gamma * cost)
    if plot:
        sns.heatmap(pi, linewidth=0.5)
        return pi, cost
    return cost


def masking_map_linear_sub_sequence(
    xs,
    xt,
    ratio=0.1,
    sub_ratio=0.1,
    eps=1e-10,
    reg=0.0001,
    max_iterations=100000,
    thres=1e-5,
    algorithm="linear_programming",
):
    sub_length = int(np.floor(min(len(xs), len(xt)) * sub_ratio))
    subs_xs = subsequences(xs, sub_length)
    subs_xt = subsequences(xt, sub_length)
    p = np.ones(len(subs_xs)) / len(subs_xs)
    q = np.ones(len(subs_xt)) / len(subs_xt)
    C = cost_matrix_aw(subs_xs, subs_xt)
    C /= C.max() + eps
    M = create_mask(C, ratio)
    if algorithm == "linear_programming":
        pi = lp(p, q, C, M)
    elif algorithm == "sinkhorn":
        pi = sinkhorn_log_domain(p, q, C, M, reg, max_iterations, thres)
    else:
        raise ValueError(
            "algorithm must be 'linear_programming' or 'sinkhorn'!")
    cost = np.sum(pi * C)
    return cost


def masking_map_linear_sub_sequence_multivariate(
    xs,
    xt,
    ratio=0.1,
    sub_ratio=0.1,
    eps=1e-10,
    reg=0.0001,
    max_iterations=100000,
    thres=1e-5,
    algorithm="linear_programming",
):
    sub_length = int(np.floor(min(len(xs), len(xt)) * sub_ratio))
    subs_xs = subsequenceData(xs, sub_length)
    subs_xt = subsequenceData(xt, sub_length)
    p = np.ones(len(subs_xs)) / len(subs_xs)
    q = np.ones(len(subs_xt)) / len(subs_xt)
    C = cost_matrix_aw(subs_xs, subs_xt)
    C /= C.max() + eps
    M = create_mask(C, ratio)
    if algorithm == "linear_programming":
        pi = lp(p, q, C, M)
    elif algorithm == "sinkhorn":
        pi = sinkhorn_log_domain(p, q, C, M, reg, max_iterations, thres)
    else:
        raise ValueError(
            "algorithm must be 'linear_programming' or 'sinkhorn'!")
    cost = np.sum(pi * C)
    return cost


def masking_map_linear_sequence(
    xs,
    xt,
    lamb=5,
    sub_length=25,
    eps=1e-10,
    reg=0.0001,
    max_iterations=100000,
    thres=1e-5,
    algorithm="linear_programming",
    cost_function="L2",
    plot=False,
):
    """
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
    """
    subs_xs = subsequences(xs, sub_length)
    subs_xt = subsequences(xt, sub_length)
    p = np.ones(len(subs_xs)) / len(subs_xs)
    q = np.ones(len(subs_xt)) / len(subs_xt)

    C = cost_matrix(subs_xs, subs_xt)
    C /= C.max() + eps

    # mask matrix
    M = create_mask(C, lamb)

    # solving model
    if algorithm == "linear_programming":
        pi = lp(p, q, C, M)
    elif algorithm == "sinkhorn":
        pi = sinkhorn_log_domain(p, q, C, M, reg, max_iterations, thres)
    else:
        raise ValueError(
            "algorithm must be 'linear_programming' or 'sinkhorn'!")
    cost = np.sum(pi * C)
    # cost = np.exp(-gamma * cost)
    if plot:
        sns.heatmap(pi, linewidth=0.5)
        return pi, cost
    return cost


def masking_map_linear_partial(
    xs,
    xt,
    ratio=0.1,
    s=0.5,
    xi=None,
    A=None,
    eps=1e-10,
    reg=0.0001,
    max_iterations=100000,
    thres=1e-5,
    algorithm="linear_programming",
    plot=False,
):
    """
    Parameters
    ----------
        xs: ndarray, (m,d)
            d-dimensional source samples
        xt: ndarray, (n,d)
            d-dimensional target samples
        s: int
            The amount of mass wanted to transport through 2 empirical distribution
        lamb: lambda, int
            Adjust the diagonal width. Default is 3
        data: 1d, 2d
            define the type of data
        algorithm: str
            algorithm to solve model. Default is "linear_programming". Choices should be
            "linear_programming" and "sinkhorn"
        plot: bool
            status for plot the optimal transport matrix or not. Default is "False"
    Returns
    -------
        cost: Transportation cost
    """
    p = torch.Tensor(np.ones(len(xs)) / len(xs))
    q = torch.Tensor(np.ones(len(xt)) / len(xt))
    C = cost_matrix_aw(xs, xt, subsequence=False)
    C /= C.max() + eps
    C = torch.Tensor(C)
    M = torch.Tensor(create_mask(C, ratio=ratio))

    # partial cost matrix
    if A is None:
        A = C.max()
    if xi is None:
        xi = 1e2 * C.max()
    C_ = torch.cat((C, xi * torch.ones(C.size(0), 1)), dim=1)
    C_ = torch.cat((C_, xi * torch.ones(1, C_.size(1))), dim=0)
    C_[-1, -1] = 2 * xi + A

    # partial empirical distributions
    p_ = torch.cat((p, (torch.sum(q) - s) * torch.Tensor([1])))
    q_ = torch.cat((q, (torch.sum(p) - s) * torch.Tensor([1])))

    # partial transportation mask
    a = torch.zeros(M.shape[0], 1, dtype=torch.int64)
    b = torch.zeros(M.shape[1] + 1, 1, dtype=torch.int64)
    M_ = torch.cat((M, a), dim=1)
    M_ = torch.cat((M_, b.t()), dim=0)
    pot = M_.shape[1]
    n, m = M_.shape
    lamb = int(ratio * min(n, m))
    for i in range(n - lamb * 2, n):
        if (i > pot * n / m - lamb) & (i < pot * n / m + lamb):
            M_[i][pot - 1] = 1

    pi_ = lp(p_.numpy(), q_.numpy(), C_.numpy(), M_.numpy())
    cost = np.sum(pi_ * C_.numpy())
    if plot:
        sns.heatmap(pi_, linewidth=0.5)
        return pi_, cost
    return cost
