# -*- coding: utf-8 -*-

from basenmf import NMF, NMFresult
from initialization import rnd_init, rndc_init, rnda_init, svd_init
import numpy as np

class SNMF(NMF):
    """
    Implementaiton of the Symmetric NMF (SNMF).
    The implementation is based on the paper:

        C. Ding, X. He, and H. Simon, “On the equivalence of nonnegative
        matrix factorization and spectral clustering,” in Proc. SIAM Int. Conf.
        Data Min., 2005, pp. 606–610.
    """
    def fit(self):
        """
        Multiplicative update rules for minimizing the objective function:
            min||X - HHT||_F
        """
        X = np.mat(self.X)
        if X.min() < 0:
            raise Exception("Negative entries are not allowed in the matrix!")
        if self.init == "rnd":
            H = rnd_init(X, self.rank)
        if self.init == "rndc":
            H = rndc_init(X, self.rank)
        if self.init == "rnda":
            H = rnda_init(X, self.rank)
        if self.init == "svd":
            H = svd_init(X, self.rank, flag=1)

        # flags and tmp variables for checking the convergence
        dist = 0
        pdist = 1e10
        converged = False
        objfun_vals = np.zeros(self.maxiter / 10)
        c = 0

        for it in range(self.maxiter):

            # multiplicative update rule
            grad_neg = X*H
            grad_pos = H*(H.T*H)

            H = np.multiply(H, 0.5 + 0.5*np.divide(grad_neg, grad_pos + 1e-10))

            if it % 10 == 0:
                dist = np.linalg.norm(X-H*H.T, 'fro')
                objfun_vals[it/10] = dist*dist
                if pdist - dist < self.tol:
                    converged = True
                    break
                if self.displ:
                    print "### Iter = %d | ObjF = %.3e | Rel = %.3e" % (it, dist*dist, pdist-dist)
                pdist = dist
                c += 1
        ## l1 row normalization
        H = np.array(H)
        norms = H.sum(axis=1)
        norms[norms==0] = 1.0
        H  /= norms[:, np.newaxis]

        return NMFresult((H,), objfun_vals[:c], dist*dist, converged)
