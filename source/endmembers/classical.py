import sys
import numpy as np
import scipy as sp
import helpers.auxiliar as auxiliar
import math
import random
from scipy import linalg
import time
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'../source/endmembers/helpers',nargout=0)
# eng.addpath(r'../source/endmembers/helpers/YALMIP',nargout=0)

def PPI(Y, d, q, numSkewers=None, ini_skewers=None):
    start_time = time.time()
    M = Y
    if numSkewers == None:
        numSkewers = 3*q
    M = np.matrix(M, dtype=np.float32)
    # rows are bands
    # columns are signals
    p, N = M.shape
    # Remove mean from data
    u = np.transpose(np.transpose(M).mean(axis=0))
    Mm = M - np.kron(np.ones((1,N)), u)
    #Generate skewers
    if ini_skewers == None:
        skewers = np.random.rand(p, numSkewers)
    else:
        skewers = ini_skewers
    
    votes = np.zeros((N, 1))
    for kk in range(numSkewers):
        # skewers[:,kk] is already a row
        tmp = abs(skewers[:,kk]*Mm)
        idx = np.argmax(tmp)
        votes[idx] = votes[idx] + 1
    max_idx = np.argsort(votes, axis=None)
    # the last right idx..s at the max_idx list are
    # those with the max votes
    end_member_idx = max_idx[-q:][::-1]
    
    U = M[:, end_member_idx]
    M = Y[:,end_member_idx]
    duration = time.time() - start_time
    return M, duration, [end_member_idx]

def NFINDR(Y, d, q, transform=None, maxit=None, ATGP_init=False):    
        data = Y
        # data size
        nsamples, nvariables = data.shape
        if maxit == None:
            maxit = 3*q
        if transform == None:
            # transform as shape (N x p)
            transform = data
            transform = auxiliar._PCA_transform(data, q-1)
        else:
            transform = transform
        # Initialization
        # TestMatrix is a square matrix, the first row is set to 1
        start_time = time.time()
        TestMatrix = np.zeros((q, q), dtype=np.float32, order='F')
        TestMatrix[0,:] = 1
        IDX = None
        if ATGP_init == True:
            induced_em, idx = auxiliar.ATGP(transform, q)
            IDX = np.array(idx, dtype=np.int64)
            for i in range(q):
                TestMatrix[1:q, i] = induced_em[i]
        else:
            IDX =  np.zeros((q), dtype=np.int64)
            for i in range(q):
                idx = int(math.floor(random.random()*nsamples))
                TestMatrix[1:q, i] = transform[idx]
                IDX[i] = idx
        actualVolume = 0
        it = 0
        v1 = -1.0
        v2 = actualVolume
        while it <= maxit and v2 > v1:
            for k in range(q):
                for i in range(nsamples):
                    TestMatrix[1:q, k] = transform[i]
                    volume = math.fabs(linalg._flinalg.sdet_c(TestMatrix)[0])
                    if volume > actualVolume:
                        actualVolume = volume
                        IDX[k] = i
                TestMatrix[1:q, k] = transform[IDX[k]]
            it = it + 1
            v1 = v2
            v2 = actualVolume
        E = np.zeros((len(IDX), nvariables), dtype=np.float32)
        Et = np.zeros((len(IDX), q-1), dtype=np.float32)
        for j in range(len(IDX)):
            E[j] = data[IDX[j]]
            Et[j] = transform[IDX[j]]
        duration = time.time() - start_time
        M = Y[:,IDX]
        return M, duration, [None]

def SGA(Y, d, q):
    imagecube = matlab.double(Y.reshape((d[0],d[1],d[2])).tolist())
    response = eng.sga(imagecube,q,nargout=2)
    indices = response[0]
    duration = response[1]
    indices = np.array(indices).astype(int)
    indices = [i[0] for i in indices]
    M = Y[:,indices]
    return M, duration, [None]

def VCA(Y, d, q, verbose=False, snr_input=0):
    R = q
    # Initializations
    if len(Y.shape)!=2:
        sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')
    [L, N]=Y.shape   # L number of bands (channels), N number of pixels
    R = int(R)
    if (R<0 or R>L):  
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')
    start_time = time.time()
    # SNR Estimates
    if snr_input==0:
        y_m = np.mean(Y,axis=1,keepdims=True)
        Y_o = Y - y_m           # data with zero-mean
        Ud  = linalg.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix 
        x_p = sp.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace
        SNR = auxiliar.estimate_snr(Y,y_m,x_p)
    if verbose:
        print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
    if verbose:
        print("input SNR = {}[dB]\n".format(SNR))
    SNR_th = 15 + 10*sp.log10(R)
    # Choosing Projective Projection or 
    #          projection to p-1 subspace
    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")
            d = R-1
            if snr_input==0: # it means that the projection is already computed
                Ud = Ud[:,:d]
            else:
                y_m = sp.mean(Y,axis=1,keepdims=True)
                Y_o = Y - y_m  # data with zero-mean 
                Ud  = linalg.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:d]  # computes the p-projection matrix 
                x_p =  sp.dot(Ud.T,Y_o)                 # project thezeros mean data onto p-subspace
            Yp =  sp.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L
            x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = sp.amax(sp.sum(x**2,axis=0))**0.5
            y = sp.vstack(( x, c*sp.ones((1,N)) ))
        else:
            if verbose:
                print("... Select the projective proj.")
            d = R
            Ud  = linalg.svd(sp.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix 
            x_p = sp.dot(Ud.T,Y)
            Yp =  sp.dot(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)
            x =  sp.dot(Ud.T,Y)
            u = sp.mean(x,axis=1,keepdims=True)        #equivalent to  u = Ud.T * r_m
            y =  x / sp.dot(u.T,x)
    # VCA algorithm
    indice = sp.zeros((R),dtype=int)
    A = sp.zeros((R,R))
    A[-1,0] = 1
    for i in range(R):
        w = sp.random.rand(R,1);   
        f = w - sp.dot(A,sp.dot(linalg.pinv(A),w))
        f = f / linalg.norm(f)
        v = sp.dot(f.T,y)
        indice[i] = sp.argmax(sp.absolute(v))
        A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))
    Ae = Yp[:,indice]
    duration = time.time() - start_time
    M = Y[:,indice]
    return M, duration, [indice]