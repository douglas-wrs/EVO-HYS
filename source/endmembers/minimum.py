import matlab.engine
import numpy as np
import helpers.auxiliar as auxiliar
from sklearn.decomposition import PCA
import time
eng = matlab.engine.start_matlab()
eng.addpath(r'../source/endmembers/helpers',nargout=0)
eng.addpath(r'../source/endmembers/helpers/SEDUMI',nargout=0)
eng.addpath(r'../source/endmembers/helpers/YALMIP',nargout=0)

def MVSA(Y, d, q):
    Y_mat = matlab.double(Y.tolist())
    response = eng.mvsa(Y_mat,q,nargout=2)
    M = np.array(response[0])
    duration = response[1]
    return M, duration, [None]

def MVES(Y, d, q):
    Y_mat = matlab.double(Y.tolist())
    response = eng.mves(Y_mat,q,0,nargout=2)
    M = np.array(response[0])
    duration = response[1]
    return M, duration, [None]

def MVCNMF(Y, d, q):
    Y_mat = matlab.double(Y.tolist())
    response = eng.mvcnmf(Y_mat,q,nargout=2)
    W = np.array(response[0])
    duration = np.array(response[1])
    return W, duration, [None]

def SISAL(Y, d, q):
    Y_mat = matlab.double(Y.tolist())
    response = eng.sisal(Y_mat,q,nargout=2)
    M = np.array(response[0])
    duration = np.array(response[1])
    return M,duration, [None]
