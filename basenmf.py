# -*- coding: utf-8 -*-

class NMF:
    """
    Base-class for different types of NMF implementations.
    """
    maxiter = 300
    tol = 1e-3
    init = "rnd"
    displ = False

    def __init__(self, X, rank, **kwargs):
        """
        Keyword arguments:

            -init: (str) the initialization technique. May take: {"svd", "rnd", "rndc", "rnda"}
            -maxiter: (int) the maximum number iterations. Default is: 300
            -tol: (float) the convergence criterion for the objective fnction. Default is: 1e-5

            :param X: symmetric matrix
            :param rank: rank parameter
        """
        if kwargs.get("init"):
            self.init = kwargs.get("init")
        if kwargs.get("tol"):
            self.tol = kwargs.get("tol")
        if kwargs.get("maxiter"):
            self.maxiter = kwargs.get("maxiter")
        if kwargs.get("displ"):
            self.displ = kwargs.get("displ")

        self.X = X
        self.rank = rank

class NMFresult:
    """
    Class for storing all the results of the NMF run.
    """
    objfun_vals = None # an array of objective function values
    matrices = None # a list of factorizing matrices
    converged = None # {true, false}

    def __init__(self, matrices, objfun_vals=None, objfun_final=None, converged=None):
        self.matrices = matrices
        self.objfun_vals = objfun_vals
        self.objfun_final = objfun_final
        self.converged = converged
