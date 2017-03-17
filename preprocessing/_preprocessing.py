# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio

def nets_from_mat(filename):
    print "### Loading *.mat file..."
    D = sio.loadmat(filename, squeeze_me=True)
    GO_data = D['GO']
    Net_data = D['networks']

    Nets = []
    for i in range(Net_data.shape[0]):
        Nets.append(Net_data[i]['data'].todense())

    goterms = GO_data['collabels'].tolist().tolist()
    goterms = [item.encode('utf-8') for item in goterms]

    genes = GO_data['rowlabels'].tolist().tolist()
    genes = [item.encode('utf-8') for item in genes]

    GO = GO_data['data'].tolist()
    GO = GO.todense()

    return genes, goterms, Nets, GO

def mltplx_from_mat(filename, net_name):
    D = sio.loadmat(filename, squeeze_me=True)
    Nets = []
    ground_idx = []
    if net_name == 'cora':
        print "### Loading CoRA file..."
        Nets = [D['A'][:,:,i] for i in range(D['A'].shape[2])]
        ground_idx = D['C']
        ground_idx = np.reshape(ground_idx, Nets[0].shape[0])
    if net_name == 'mit':
        print "### Loading MIT file..."
        Nets.append(D['celltower_graph'])
        Nets.append(D['phone_graph'])
        Nets.append(D['bt_graph'])
        ground_idx = np.zeros((Nets[0].shape[0], 1), dtype=int)
        for k in range(D['C'].shape[0]):
            ground_idx[D['C'][k]-1] = k+1
        ground_idx = np.reshape(ground_idx, Nets[0].shape[0])

    return Nets, ground_idx

def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print "### Negative entries in the matrix are not allowed!"
        X[X<0] = 0
        print "### Matrix converted to nonnegative matrix."
        print
    if (X.T == X).all():
        pass
    else:
        print "### Matrix not symmetric."
        X = X + X.T - np.diag(np.diag(X))
        print "### Matrix converted to symmetric."

    ### normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X

def net_normalize(Net):
    """
    Normalize Nets or list of Nets.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
    else:
        Net = _net_normalize(Net)

    return Net
