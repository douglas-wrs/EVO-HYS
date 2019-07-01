import numpy as np
import math
import random
import scipy.io as sio
import scipy as sp
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class NFINDR(object):

	data = None
	nRow = None
	nCol = None
	nBand = None
	nPixel = None
	p = None
	
	maxite = None

	endmembers = None
	purepixels = None

	genMean = None

	verbose = True

	def __init__(self, argin, verbose):
		self.verbose = verbose
		if (self.verbose):
			print ('---		Initializing VCA algorithm')
		self.data = argin[0]
		self.nRow = argin[1]
		self.nCol = argin[2]
		self.nBand = argin[3]
		self.nPixel = argin[4]
		self.p = argin[5]

		self.maxite = argin[6]

	def _PCA_transform(self,M, n_components):
		pca = PCA(n_components=n_components)
		return pca.fit_transform(M)

	def extract_endmember(self):
		if (self.verbose):
			print('---		Starting endmembers Extracting')
		data = self.data
		q = self.p
		maxit = self.maxite
		nsamples, nvariables = data.shape
		transform = self._PCA_transform(data, q-1)


		TestMatrix = np.zeros((q, q), dtype=np.float32, order='F')
		TestMatrix[0,:] = 1
		IDX = None
	
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
					volume = math.fabs(sp.linalg._flinalg.sdet_c(TestMatrix)[0])
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

		self.endmembers = np.asmatrix(E).T
		self.purepixels = IDX

		return [self.endmembers, self.purepixels]
