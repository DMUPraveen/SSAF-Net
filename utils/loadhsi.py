import numpy as np
import scipy.io as scio

def loadhsi(case):
    if case == 'ridge':
        file = './dataset/JasperRidge2_R198.mat'
        data = scio.loadmat(file)
        Y = data['Y']
        Y = np.reshape(Y,[198,100,100])
        for i,y in enumerate(Y):
            Y[i]=y.T
        Y = np.reshape(Y, [198, 10000])

        GT_file = './dataset/JasperRidge2_end4.mat'
        A_true = scio.loadmat(GT_file)['A']
        M = scio.loadmat(GT_file)['M']
        A_true = np.reshape(A_true, (4, 100, 100))
        for i,A in enumerate(A_true):
            A_true[i]=A.T
        A_true = np.reshape(A_true, (4, 10000))
        if np.max(Y) > 1:
            Y = Y / np.max(Y)

    P = A_true.shape[0]

    Y = Y.astype(np.float32)
    A_true = A_true.astype(np.float32)

    return Y, A_true, P, M