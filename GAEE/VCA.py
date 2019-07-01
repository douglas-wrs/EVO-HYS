import numpy as np
import scipy.io as sio
import numpy.linalg as la
import matplotlib.pyplot as plt

class VCA(object):

	data = None
	nRow = None
	nCol = None
	nBand = None
	nPixel = None
	p = None
	
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

	def extract_endmember(self):
		if (self.verbose):
			print('---		Starting endmembers Extracting')
		R = self.data
		L, N = R.shape
		SNR_th = 15 + 10*np.log10(self.p)
		r_m = R.mean(1)  # mean
		R_m = np.matlib.repmat(r_m,1,N) # mean of each band
		R_o = R-R_m # data with zero-mean
		if (self.verbose):
			print('---		Reducing data dimension (SVD)')
		[Ud, Sd, Vd] = la.svd(R_o*R_o.T/N)
		Ud = Ud[:,:self.p]
		Sd = Sd[:self.p]
		Vd = Vd[:self.p,:]
		x_p = Ud.T*R_o
		P_y = (np.power(R,2)/N).sum()
		P_x = np.asscalar((np.power(x_p,2)/N).sum() + (r_m.T*r_m))
		if (self.verbose):
			print('---		Estimating data SNR')
		SNR = np.asscalar(10*np.log10(np.abs([(P_x - self.p/L*P_y)/(P_y-P_x)])))
		if SNR < SNR_th:
			if (self.verbose):
				print('---		Applying projective projection')
			d = self.p-1
			Ud = Ud[:,:d] 
			R_p = Ud*x_p[:d,:] + R_m
			x = x_p[:d,:]
			c = np.sqrt((np.power(x,2).sum(0)).max())
			y = np.concatenate((x,c*np.matrix(np.ones(N))),axis=0)
		else:
			if (self.verbose):
				print('---		Applying projection to p-1 subspace')
			d = self.p
			[Ud, Sd, Vd] = la.svd((R*R.T)/N)
			Ud = Ud[:,:d]
			Sd = Sd[:d]
			Vd = Vd[:d,:]
			x_p = Ud.T*R
			R_p = Ud*x_p[:d,:]
			x = x_p
			u = x.mean(1)
			aux1 = np.matlib.repmat(u,1,N).T
			aux2 = (x * np.matlib.repmat(u,1,N).T).sum()
			y = x / np.matlib.repmat( (x * np.matlib.repmat(u,1,N).T).sum() ,d,1)
		indice = np.zeros(self.p)
		S = np.matrix(np.zeros((self.p,self.p)))
		S[self.p-1,0] = 1
		if (self.verbose):
			print('---		Starting vertex search')
		for i in range(0,self.p):
			w = np.random.rand(self.p,1)
			aux = S*la.pinv(S)
			f = w - (aux*w)
			aux = np.sqrt(np.sum(np.power(f,2)))
			f = f / aux
			v = f.T*y
			aux = np.abs(v)
			v_max = np.max(aux)
			indice = indice.astype(int)
			indice[i] = np.asscalar(np.argmax(aux))
			S[:,i] = y[:,indice[i]]
		if (self.verbose):
			print('---		Ending endmembers Extracting')
		self.y = R_p
		self.endmembers = R_p[:,indice]
		self.purepixels = indice
		return [self.endmembers, self.purepixels]
