import numpy as np
from MaskingMap.Utilities.linearprog import lp_partial
from MaskingMap.Utilities.sinkhorn import sinkhorn_log_domain
from MaskingMap.Utilities.utils import cost_matrix, cost_matrix_1d, create_mask_KL, create_mask_binary
import matplotlib.pyplot as plt


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
