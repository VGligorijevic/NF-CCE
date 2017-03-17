# -*- coding: utf-8 -*-

from numpy import linalg as la
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


def nmf_clust(H):
    """
    NMF clustering from H matrix.
    """
    H = np.array(H)
    idx = np.argmax(H, axis=1)

    return idx

def spect_clust(H, k):
    """
    Spectral clustering from H matrix.
    """
    ## Similarity matrix
    H = np.array(H)
    S = H.dot(H.T)

    ## Normalized similarity matrix
    rowsum = S.sum(axis=1).flatten()
    D = np.diag(1./np.sqrt(rowsum + 1e-8))
    L = D.dot(S.dot(D))

    ## compute eigenvectors of L
    U, _, _ = la.svd(L, full_matrices=False)


    ## K-means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(U)

    return kmeans.labels_

def clust_eval(true_idx, pred_idx):
    """
    Clustering evaluation measures.
    """

    ARI = metrics.adjusted_rand_score(true_idx, pred_idx)
    NMI = metrics.normalized_mutual_info_score(true_idx, pred_idx)

    return NMI, ARI
