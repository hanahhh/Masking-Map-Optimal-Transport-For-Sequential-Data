import numpy as np
import math
from MaskingMap.Utilities.linearprog import lp_aw
from MaskingMap.Utilities.sinkhorn import sinkhorn_log_domain_aw
from MaskingMap.Utilities.utils import cost_matrix_aw, create_mask_linear_aw, create_mask_non_linear_aw


def masking_map_auto_weighted(xs, xt, ratio=0.2, lamb=20, eps=1e-10, max_iterations=100000, algorithm="linear_programming"):
    C = cost_matrix_aw(xs, xt)
    old_w1 = -100
    w1, w2 = 1/2, 1/2
    M1 = create_mask_linear_aw(xs=xs, xt=xt, lamb=lamb)
    M2 = create_mask_non_linear_aw(xs=xs, xt=xt, ratio=ratio)
    T = np.zeros_like(C)
    iteration_num = 0
    while w1 - old_w1 > eps:
        iteration_num += 1
        M = w1*M1 + w2*M2
        p = np.ones(len(xs))/len(xs)
        q = np.ones(len(xt))/len(xt)
        if algorithm == "linear_programming":
            T = lp_aw(p=p, q=q, C=C, Mask=M)
        elif algorithm == "sinkhorn":
            T = sinkhorn_log_domain_aw(p=p, q=q, C=C, Mask=M)
        else:
            raise ValueError(
                "algorithm must be 'linear_programming' or 'sinkhorn'!")
        old_w1 = w1
        w1 = 1/(2*math.sqrt(np.sum((M1*T)*C)))
        w2 = 1/(2*math.sqrt(np.sum((M2*T)*C)))
    cost = w1*np.sum((M1*T)*C) + w2*np.sum((M2*T)*C)
    # return T, w1*np.sum((M1*T)*C) + w2*np.sum((M2*T)*C), w1, w2, iteration_num
    return cost
