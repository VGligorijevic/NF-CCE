# -*- coding: utf-8 -*-

from snmf import SNMF
from basenmf import NMF, NMFresult
import numpy as np

class CSNMF(NMF):
    """
    Implementation of the Collective SNMF (CSNMF).
    The implementation is based on the paper:
        V. Gligorijevic, Y. Panagakis, S. Zafeiriou, "Non-Negative Matrix
        Factorizations for Multiplex Network Analysis", 2017, T-PAMI.
    """
    def __init__(self, X, rank, alpha = 0.5, **kwargs):
        NMF.__init__(self, X, rank, **kwargs)
        self.alpha = alpha

    def fit(self):
        """
        Multiplicative update rules for minimizing the objective function:
            sum_i ||X_i - HHT||_F + alpha sum_i ||HHT - H_iH_iT||_F
        """
        N = len(self.X)
        H = []
        for i in range(N):
            print "### Factorizing network [%d]..."%(i+1)
            X = np.mat(self.X[i])
            objSNMF = SNMF(X, self.rank, init = self.init, displ = self.displ)
            res_snmf = objSNMF.fit()
            H.append(np.mat(res_snmf.matrices[0]))
            if res_snmf.converged:
                print "### Converged."
        A_avg = np.mat(np.zeros((self.X[0].shape), dtype=float))
        for i in range(N):
            A_avg += self.X[i] + self.alpha*(H[i]*H[i].T)
        A_avg /= (1.0 + self.alpha)*N
        objSNMF = SNMF(A_avg, self.rank, init = self.init, displ = self.displ)
        res_snmf = objSNMF.fit()

        return res_snmf
