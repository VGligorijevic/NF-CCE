# add current directory

import numpy as np
import pylab as pl
#import os, sys
#sys.path.append("/Users/vgligorijevic/Projects/NF-CCE/data/mostafavi/")

from snmf import SNMF
from csnmf import CSNMF
from preprocessing import nets_from_mat


## create symmetric matrix (test)
#X = np.mat(np.random.rand(10, 10))
#X = 0.5*(X + X.T)
#X = X - np.diag(np.diag(X))

## load Mostafavi data
genes, _, Nets, _ = nets_from_mat("data/mostafavi/human_and_go.mat")
Nets = Nets[:3]



#objSNMF = SNMF(Nets[1], 50, init='rnda', displ='true')
#res_snmf = objSNMF.fit()

objCSNMF = CSNMF(Nets, 30, alpha = 0.5, init = 'rnda', displ = True)
res_csnmf = objCSNMF.fit()


H = np.mat(res_csnmf.matrices[0])
J = res_csnmf.objfun_vals

# show results
pl.figure(1)
pl.title("Cluster indicator matrix")
pl.imshow(H[1:500, :], interpolation='nearest')
#pl.imshow(H*H.T, interpolation='nearest')
pl.colorbar()

pl.figure(2)
pl.title("Obj function")
pl.plot(J, 'o-')

pl.show()
