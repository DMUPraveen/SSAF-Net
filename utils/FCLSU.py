import numpy as np
import scipy as sp
import torch
from scipy.optimize import nnls

def FCLSU(M, Y, sigma=1):
    P = M.shape[1]
    N = Y.shape[1]
    M = sp.vstack((sigma * M, sp.ones((1, P)) ))
    Y = sp.vstack((sigma * Y, sp.ones((1, N)) ))
    A_hat = np.zeros((P, N))

    for i in np.arange(N):
        A_hat[:, i], res = nnls(M, Y[:, i])
    A_hat = torch.tensor(A_hat)

    return A_hat
