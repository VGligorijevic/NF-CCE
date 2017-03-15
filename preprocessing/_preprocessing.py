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


