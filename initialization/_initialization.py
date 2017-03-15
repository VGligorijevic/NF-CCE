# -*- coding: utf-8 -*-

import numpy as np
from math import ceil, sqrt
from operator import itemgetter
from numpy import linalg as la

def rnd_init(X, k):
    """"
    Random initialization.
    """
    H = np.mat(np.random.rand(X.shape[1], k))

    return H

def rnda_init(X, k, p = None):
    """
    RandomAcol initialization.
    """
    if p is None:
        p = int(ceil(1./5*X.shape[0]))
    prng = np.random.RandomState()
    H = np.mat(np.zeros((X.shape[0], k)))
    for i in range(k):
        H[:, i] = X[:, prng.randint(low=0, high=X.shape[0], size=p)].mean(axis=1)

    return H

def rndc_init(X, k, p = None, l = None):
    """
    RandomC initialization.
    """
    if p is None:
        p = int(ceil(1./5*X.shape[0]))
    if l is None:
        l = int(ceil(1./2*X.shape[0]))
    prng = np.random.RandomState()
    H = np.mat(np.zeros((X.shape[0], k)))
    top = sorted(enumerate([la.norm(X[i, :], 2) for i in range(X.shape[0])]), key=itemgetter(1), reverse=True)[:l]
    top = np.mat(list(zip(*top))[0])
    for i in range(k):
        H[:, i] = X[:, top[0, prng.randint(low=0, high=l, size=p)].tolist()[0]].mean(axis=1)

    return H

def svd_init(X, k, flag = 0):
    """SVD-based initialization. The implementation is based on paper:

    Boutsidis, C., & Gallopoulos, E. (2008). "SVD based initialization: A
    head start for nonnegative matrix factorization." Pattern Recognition,
    41(4), 1350-1362.
    """
    H = np.mat(np.zeros((X.shape[0], k)))
    U, S, V = la.svd(X)
    H[:, 0] = sqrt(S[0])*abs(U[:,0])

    for i in range(1, k):
        uu = U[:, i]
        uup = _pos(uu)
        uun = _neg(uu)
        n_uup = la.norm(uup, 2)
        n_uun = la.norm(uun, 2)
        termp = n_uup*n_uup
        #termn = n_uun*n_uun
        #if termp >= termn:
        #    H[:, i] = (sqrt(S[i]*termp)/n_uup)*uup
        #else:
        #    H[:, i] = (sqrt(S[i]*termn)/n_uun)*uun
        H[:, i] = sqrt(S[i]*termp)/n_uup*uup
    H[H < 1e-10] = 0

    if flag == 1:
        avg = X.mean()
        H[H == 0] = avg
    if flag == 2:
        avg = X.mean()
        n = len(H[H==0])
        H[H==0] = avg*np.random.uniform(n, 1)/100.0

    return H

def _pos(X):
    """
    Return positive elements of matrix X.
    """
    return np.multiply(X>=0, X)

def _neg(X):
    """
    Return negative elements of matrix X.
    """
    return np.multiply(X<0, -X)
