import numpy as np
import scipy.io as sio
import numpy.linalg as la
import matplotlib.pyplot as plt

class PPI(object):

	data = None
	nRow = None
	nCol = None
	nBand = None
	nPixel = None
	p = None

	nSkewers = None
	initSkewers = None

	endmembers = None
	purepixels = None

	genMean = None

	verbose = True

	def __init__(self, argin, verbose):
		self.verbose = verbose
		if (self.verbose):
			print ('---		Initializing PPI algorithm')
		self.data = argin[0].T
		self.nRow = argin[1]
		self.nCol = argin[2]
		self.nBand = argin[3]
		self.nPixel = argin[4]
		self.p = argin[5]

		self.nSkewers = argin[6]
		self.initSkewers = argin[7]

	def extract_endmember(self):
		if (self.verbose):
			print('---		Starting endmembers Extracting')

		M = self.data.T
		M = np.matrix(M, dtype=np.float32)
		
		p, N = M.shape

		u = np.transpose(np.transpose(M).mean(axis=0))
		Mm = M - np.kron(np.ones((1,N)), u)

		if self.initSkewers == None:
			skewers = np.random.rand(p, self.nSkewers)
		else:
			skewers = self.initSkewers
		votes = np.zeros((N, 1))

		for kk in range(self.nSkewers):
			tmp = abs(skewers[:,kk]*Mm)
			idx = np.argmax(tmp)
			votes[idx] = votes[idx] + 1

		max_idx = np.argsort(votes, axis=None)
		
		end_member_idx = max_idx[-self.p:][::-1]
		U = M[:, end_member_idx]

		self.endmembers = U
		self.purepixels = end_member_idx

		return [self.endmembers, self.purepixels]