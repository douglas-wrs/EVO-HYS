from VCA import *
from PPI import *
from NFINDR import *
from GAEE import *
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import seaborn as sns; sns.color_palette("pastel",7)
from scipy import stats
from timeit import default_timer as timer

class DEMO(object):

	name = None

	ee = None
	am = None
	
	data = None
	nRow = None
	nCol = None
	nBand = None
	nPixel = None
	selectedBands = None
	p = None 

	thr = None

	groundtruth = None
	num_gtendm = None

	raw_endmembers = None
	abundances = None

	sam_em_max = None
	sam_em_min = None
	
	sam_values_max = None
	sam_values_min = None
	
	sam_max = None
	sam_min = None

	sam_mean = None
	sam_var = None
	sam_std = None

	sid_em_max = None
	sid_em_min = None

	sid_values_max = None
	sid_values_min = None

	sid_max = None
	sid_min = None

	sid_mean = None
	sid_var = None
	sid_std = None

	gen_max = None
	gen_avg = None
	gen_min = None

	sam_all_runs_value = None
	sid_all_runs_value = None

	time_runs = None

	verbose = True

	def __init__(self, argin,verbose):
		if (verbose):
			print('... Initializing DEMO')
		self.verbose = verbose
		self.load_data(argin[0])
		self.load_groundtruth(argin[1])
		self.data = self.convert_2D(self.data)
		self.p = argin[2]
		self.name = argin[3]
		
		if argin[3] == 'VCA':
			if (verbose):
				print('... Selecting VCA endmember extractor')
			self.ee = VCA([self.data,self.nRow,self.nCol,self.nBand,self.nPixel, self.p],self.verbose)
		if argin[3] == 'PPI':
			nSkewers = argin[4]
			initSkewers = argin[5]
			if (verbose):
				print('... Selecting PPI endmember extractor')
			self.ee = PPI([self.data,self.nRow,self.nCol,self.nBand,self.nPixel, self.p, nSkewers, initSkewers],self.verbose)
		if argin[3] == 'NFINDR':
			maxit = argin[4]
			if (verbose):
				print('... Selecting NFINDR endmember extractor')
			self.ee = NFINDR([self.data.T,self.nRow,self.nCol,self.nBand,self.nPixel, self.p, maxit],self.verbose)

		if argin[3] == 'GAEE' or argin[3] == 'GAEE-IVFm' or argin[3] == 'GAEE-VCA' or argin[3] == 'GAEE-IVFm-VCA':
			npop = argin[4]
			ngen = argin[5]
			cxpb = argin[6]
			mutpb = argin[7]
			initPurePixels = argin[8]
			ivfm = argin[9]
			if (verbose):
				print('... Selecting GAEE endmember extractor')
			self.ee = GAEE([self.name,self.data,self.nRow,self.nCol,self.nBand,self.nPixel, self.p, npop,
				ngen,cxpb,mutpb,initPurePixels,ivfm],self.verbose)		

	def extract_endmember(self):
		if (verbose):
			print('... Extracting endmembers')
		self.raw_endmembers = self.ee.extract_endmember()[0]

	def map_abundance(self):
		if (verbose):
			print('... Mapping abundances')
		self.am.endmembers = self.raw_endmembers
		self.abundances = self.am.map_abundance()

	def load_data(self,data_loc):
		if (verbose):
			print('... Loading data')

		pkg_data = sio.loadmat(data_loc)
		self.data = pkg_data['X']
		
		self.nRow = self.data.shape[0]
		self.nCol = self.data.shape[1]
		self.nBand = self.data.shape[2]
		self.selectedBands = pkg_data['BANDS'][0]

	def load_groundtruth(self,gt_loc):
		if (verbose):
			print('... Loading groundtruth')
		pkg_gt = sio.loadmat(gt_loc)
		self.groundtruth = pkg_gt['Y']
		self.num_gtendm = self.groundtruth.shape[1]

	def convert_2D(self,data):
		if (verbose):
			print('... Converting 3D data to 2D')
		self.nPixel = self.nRow*self.nCol
		data_2D = np.asmatrix(data.reshape((self.nRow*self.nCol,self.nBand))).T
		return data_2D

	def convert_3D(self,data):
		if (verbose):
			print('... Converting 2D data to 3D')
		data_3D = np.asarray(data)
		data_3D = data_3D.reshape((self.nRow,self.nCol,self.p))
		return data_3D

	def print_purepixels(self):
		if (verbose):
			print('... Printing pure pixels')
		print(self.ee.purepixels)

	def plot_abundance(self,i):
		if (verbose):	
			print('... Plotting abundance')
		plt.matshow(self.abundances[:,:,i])

	def plot_groundtruth(self,i):
		if (verbose):
			print('... Plotting groundtruth')
		plt.plot(self.groundtruth[:,i])
		plt.title(str(i))
		plt.xlabel('wavelength (µm)')
		plt.ylabel('reflectance (%)')
		plt.tight_layout()

	def plot_groundtruth_ab(self):
		if (verbose):
			print('... Plotting groundtruth')
		plt.matshow(self.groundtruth[:,:])

	def plot_endmember(self,i):
		if (verbose):
			print('... Plotting endmember')
		plt.plot(self.new_endmember[:,i])
		plt.title(str(i))
		plt.xlabel('wavelength (µm)')
		plt.ylabel('reflectance (%)')
		plt.tight_layout()

	def SAM(self,a,b):
		if (verbose):
			print('... Applying SAM metric')
		[L, N] = a.shape
		errRadians = np.zeros(N)
		b = np.asmatrix(b)
		for k in range(0,N):
			tmp = np.asmatrix(a[:,k])
			s1 = tmp.T
			s2 = b
			s1_norm = la.norm(s1)
			s2_norm = la.norm(s2)
			sum_s1_s2 = s1*s2.T
			aux = sum_s1_s2 / (s1_norm * s2_norm)
			aux[aux>1.0] = 1
			angle = math.acos(aux)
			errRadians[k] = angle
		return errRadians

	def SID(self,s1,s2):
		if (verbose):
			print('... Applying SID metric')
		[L, N] = s1.shape
		errRadians = np.zeros(N)
		s1 = np.asarray(s1)
		s2 = np.asarray(s2)
		for k in range(0,N):
			tmp = s1[:,k]
			p = (tmp / np.sum(tmp)) + np.spacing(1)
			q = (s2 / np.sum(s2)) + np.spacing(1)
			angle = np.sum(p.T * np.log(p / q) + q * np.log(q / p))

			if np.isnan(angle):
				errRadians[k] = 0
			else:
				errRadians[k] = angle

		return errRadians

	def best_sam_match(self): # Best Estimated endmember for the groundtruth
		if (verbose):
				print('... Matching best endmember and groundtruth pair')
		sam = np.zeros((self.num_gtendm,self.p))
		sid = np.zeros((self.num_gtendm,self.p))

		for i in range(self.raw_endmembers.shape[1]):
			sam[i,:] = self.SAM(self.raw_endmembers,self.groundtruth[self.selectedBands,:][:,i])
			sid[i,:] = self.SID(self.raw_endmembers,self.groundtruth[self.selectedBands,:][:,i])

		idxs = [list(x) for x in np.argsort(sam,axis=1)]
		values = [list(x) for x in np.sort(sam,axis=1)]
		pidxs = list(range(self.p))
		aux = []

		for i in range(self.num_gtendm):
			for j in range(self.p):
				aux.append([pidxs[i],idxs[i][j],values[i][j]])

		aux = sorted(aux, key=lambda x: x[2])
		new_idx = [0]*self.p
		new_value = [0]*self.p

		for i in range(self.p):
			a_idx, b_idx, c_value = aux[0]
			new_idx[a_idx] = b_idx
			new_value[a_idx] = c_value
			aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

		self.sam_endmembers = self.raw_endmembers[:,new_idx]

		return [new_idx, new_value]

	def best_sid_match(self):
		if (verbose):
				print('... Matching best endmember and groundtruth pair')
		sid = np.zeros((self.num_gtendm,self.p))
		for i in range(self.raw_endmembers.shape[1]):
			sid[i,:] = self.SID(self.raw_endmembers,self.groundtruth[self.selectedBands,:][:,i])
	
		idxs = [list(x) for x in np.argsort(sid,axis=1)]
		values = [list(x) for x in np.sort(sid,axis=1)]
		pidxs = list(range(self.p))
		aux = []

		for i in range(self.num_gtendm):
			for j in range(self.p):
				aux.append([pidxs[i],idxs[i][j],values[i][j]])

		aux = sorted(aux, key=lambda x: x[2])
		new_idx = [0]*self.p
		new_value = [0]*self.p

		for i in range(self.p):
			a_idx, b_idx, c_value = aux[0]
			new_idx[a_idx] = b_idx
			new_value[a_idx] = c_value
			aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

		self.sid_endmembers = self.raw_endmembers[:,new_idx]

		return [new_idx, new_value]

	def best_run(self,mrun):
		if (verbose):
				print('... Starting Monte Carlo set of runs')
		sam_min_run = 9999
		sam_min_values = None
		sam_min_idx = None
		sam_min_data = None

		sam_max_run = -9999
		sam_max_values = None
		sam_max_idx = None
		sam_max_data = None

		gen_max = None
		gen_avg = None
		gen_min = None

		self.sam_all_runs_value = np.zeros((mrun,self.p))

		sid_min_run = 9999
		sid_min_values = None
		sid_min_idx = None
		sid_min_data = None

		sid_max_run = -9999
		sid_max_values = None
		sid_max_idx = None
		sid_max_data = None

		self.sid_all_runs_value = np.zeros((mrun,self.p))
		#print(self.name)
		self.time_runs = []
		for i in range(mrun):
			start = timer()
			self.extract_endmember()
			end = timer()
			#print('START',start)
			#print('END',end)
			self.time_runs.append(end-start)

			#print(self.time_runs)
			# self.extract_endmember()
			[sam_idx, sam_value] = self.best_sam_match()
			[sid_idx, sid_value] = self.best_sid_match()
			
			self.sam_all_runs_value[i,:] = sam_value
			self.sid_all_runs_value[i,:] = sid_value

			if(np.mean(sam_value) <= sam_min_run):
				sam_min_run = np.mean(sam_value)
				sam_min_values = sam_value
				sam_min_idx = sam_idx
				sam_min_data = self.sam_endmembers
				gen_min = self.ee.genMean

			if(np.mean(sam_value) >= sam_max_run):
				sam_max_run = np.mean(sam_value)
				sam_max_values = sam_value
				sam_max_idx = sam_idx
				sam_max_data = self.sam_endmembers
				gen_max = self.ee.genMean

			if(np.mean(sid_value) <= sid_min_run):
				sid_min_run = np.mean(sid_value)
				sid_min_values = sid_value
				sid_min_idx = sid_idx
				sid_min_data = self.sid_endmembers

			if(np.mean(sid_value) <= sid_max_run):
				sid_max_run = np.mean(sid_value)
				sid_max_values = sid_value
				sid_max_idx = sid_idx
				sid_max_data = self.sid_endmembers

		self.sam_em_max = sam_max_data
		self.sam_em_min = sam_min_data
		
		self.sam_values_max = sam_max_values
		self.sam_values_min = sam_min_values
		
		self.sam_max = sam_max_run
		self.sam_min = sam_min_run

		self.sam_mean = np.mean(self.sam_all_runs_value,axis=0)
		self.sam_var = np.var(self.sam_all_runs_value,axis=0)
		self.sam_std = np.std(self.sam_all_runs_value,axis=0)

		self.sid_em_max = sid_max_data
		self.sid_em_min = sid_min_data

		self.gen_max = gen_max
		self.gen_min = gen_min
		
		self.sid_values_max = sid_max_values
		self.sid_values_min = sid_min_values
		
		self.sid_max = sid_max_run
		self.sid_min = sid_max_run

		self.sid_mean = np.mean(self.sid_all_runs_value,axis=0)
		self.sid_var = np.var(self.sid_all_runs_value,axis=0)
		self.sid_std = np.std(self.sid_all_runs_value,axis=0)	


	# self.sam_endmembers = sam_best_data
	# self.sam_values = sam_best_values
	# self.sam_mean = np.mean(sam_all_runs_value,axis=0)
	# self.sam_std = np.std(sam_all_runs_value,axis=0)

	# self.sid_endmembers = sid_best_data
	# self.sid_values = sid_best_values
	# self.sid_mean = np.mean(sid_all_runs_value,axis=0)
	# self.sid_std = np.std(sid_all_runs_value,axis=0)

def _plot_range_band(*args, central_data=None, ci=None, data=None, **kwargs):
    upper = data.max(axis=0)
    lower = data.min(axis=0)
    #import pdb; pdb.set_trace()
    ci = np.asarray((lower, upper))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    sns.timeseries._plot_ci_band(*args, **kwargs)

def best_conf(mrun,npop,ngen,cxpb,mutpb):
	if (verbose):
		print('... Searching best configuration for GAEEs algorithms')
	
	vca = DEMO([data_loc,gt_loc,num_endm,'VCA'],verbose)
	initPurePixels = vca.ee.extract_endmember()[1]

	results = {'GAEE':[], 'GAEE-IVFm':[], 'GAEE-VCA':[], 'GAEE-IVFm-VCA':[]}

	for i in npop:
		for j in ngen:
			for k in cxpb:
				for l in mutpb:
					gaee = DEMO([data_loc,gt_loc,num_endm,'GAEE',i,j,k,l,None,False],verbose)
					ivfm = DEMO([data_loc,gt_loc,num_endm,'GAEE-IVFm',i,j,k,l,None,True],verbose)
					gaee_vca = DEMO([data_loc,gt_loc,num_endm,'GAEE-VCA',i,j,k,l,initPurePixels,False],verbose)
					ivfm_vca = DEMO([data_loc,gt_loc,num_endm,'GAEE-IVFm-VCA',i,j,k,l,initPurePixels,True],verbose)

					algo = [gaee, ivfm, gaee_vca, ivfm_vca]
					for m in algo:
						print(i,j,k,l,m.name)
						m.best_run(mrun)
						results[m.name].append([ [i,j,k,l], m , np.mean(m.sam_values_min)])

	algo_bconf = {'GAEE':[], 'GAEE-IVFm':[], 'GAEE-VCA':[], 'GAEE-IVFm-VCA':[]}

	aux = results['GAEE'][0][1].gen_max
	name = []
	gen = []
	subject = []
	value = []

	for i in results:
		aux1 = [j[2] for j in results[i]]
		indx = np.argmin(aux1) 
		algo_bconf[i].append(results[i][indx])

	for l in range(len(aux)):
		for i in results:
			k = 0
			name.append(i)
			gen.append(l)
			value.append(results[i][0][1].gen_max[l][0])
			subject.append(k)
			k+=1
			name.append(i)
			gen.append(l)
			value.append(results[i][0][1].gen_min[l][0])
			subject.append(k)
			k+=1

	d = {'Algorithms':name,'Generations':gen,'subject': subject,'Log10(volume)':value}
	df = pd.DataFrame(data=d)
	plt.figure()
	ax = sns.tsplot(time="Generations", value="Log10(volume)",
	                 unit="subject", condition="Algorithms",
	                 data=df)
	plt.title("GAEEs Convergence")
	plt.tight_layout()
	plt.savefig('./IMAGES/Convergence.png', format='png', dpi=200)

	return algo_bconf

def run():
	print("BEST CONFIGURATION")
	conf = best_conf(mrun,npop,ngen,cxpb,mutpb)
	#print(conf['GAEE'])

	parameters_names = ['Population Size','Number of Generations','Crossover Probability','Mutation Probability']
	tab3_conf = pd.DataFrame()
	tab3_conf['Parameters'] = parameters_names
	tab3_conf.set_index('Parameters',inplace=True)
	tab3_conf['GAEE'] = conf['GAEE'][0][0]
	tab3_conf['GAEE-IVFm'] = conf['GAEE-IVFm'][0][0]
	tab3_conf['GAEE-VCA'] = conf['GAEE-VCA'][0][0]
	tab3_conf['GAEE-IVFm-VCA'] = conf['GAEE-IVFm-VCA'][0][0]

	file.write('### Parameters used in each GAEE versions\n\n')
	file.write(tabulate(tab3_conf, tablefmt="pipe", headers="keys")+'\n\n')

	file.write('![alt text](./IMAGES/Convergence.png)\n\n')

	endmember_names = ['Alunite','Andradite','Buddingtonite','Dumortierite','Kaolinite_1','Kaolinite_2','Muscovite',
				'Montmonrillonite','Nontronite','Pyrope','Sphene','Chalcedony']

	print("PPI")
	ppi = DEMO([data_loc,gt_loc,num_endm,'PPI',nSkewers,initSkewers],verbose)
	ppi.best_run(mrun)
	print("NFINDR")
	nfindr = DEMO([data_loc,gt_loc,num_endm,'NFINDR',maxit],verbose)
	nfindr.best_run(mrun)
	print("VCA")
	vca = DEMO([data_loc,gt_loc,num_endm,'VCA'],verbose)
	vca.best_run(mrun)
	
	algo = [ppi, nfindr, vca, conf['GAEE'][0][1], conf['GAEE-IVFm'][0][1], conf['GAEE-VCA'][0][1], conf['GAEE-IVFm-VCA'][0][1]]

	tab1_sam = pd.DataFrame()
	tab1_sam['Endmembers'] = endmember_names
	tab1_sam.set_index('Endmembers',inplace=True)

	tab2_sid = pd.DataFrame()
	tab2_sid['Endmembers'] = endmember_names
	tab2_sid.set_index('Endmembers',inplace=True)

	tab4_sam_stats = pd.DataFrame()
	tab4_sam_stats['Statistics'] = ['_Mean_','_Std_','_p-value_','Gain','_Time_']
	tab4_sam_stats.set_index('Statistics',inplace=True)

	tab5_sid_stats = pd.DataFrame()
	tab5_sid_stats['Statistics'] = ['_Mean_','_Std_','_p-value_','Gain','_Time_']	
	tab5_sid_stats.set_index('Statistics',inplace=True) 

	best_sam_gaee = None
	best_sam_mean_gaee = 9999
	best_sid_gaee = None
	best_sid_mean_gaee = 9999

	algo2 = []

	for k in range(0,3):
		algo2.append(algo[k])

	for k in algo:
		if k.name != 'PPI' and k.name != 'NFINDR' and k.name != 'VCA':
			if np.mean(k.sam_mean) <= best_sam_mean_gaee :
				best_sam_gaee = k
				best_sam_mean_gaee = np.mean(k.sam_mean)
			if np.mean(k.sid_mean) <= best_sid_mean_gaee :
				best_sid_gaee = k
				best_sid_mean_gaee = np.mean(k.sid_mean)

	algo2.append(best_sam_gaee)


	for l in algo:

		p = stats.ttest_ind(np.mean(vca.sam_all_runs_value,axis=1),np.mean(l.sam_all_runs_value,axis=1))
		tab1_sam[l.name] = l.sam_values_min
		tab4_sam_stats[l.name] = [np.mean(l.sam_mean), np.mean(l.sam_std), p[0], 100-(100*np.mean(best_sam_gaee.sam_mean)/np.mean(l.sam_mean)), np.mean(l.time_runs)]

		p = stats.ttest_ind(np.mean(vca.sid_all_runs_value,axis=1),np.mean(l.sid_all_runs_value,axis=1))
		tab2_sid[l.name] = l.sid_values_min
		tab5_sid_stats[l.name] = [np.mean(l.sid_mean), np.mean(l.sid_std), p[0],100-(100*np.mean(best_sid_gaee.sam_mean)/np.mean(l.sam_mean)) ,np.mean(l.time_runs)]


	file.write('### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using SAM for the Cuprite Dataset.\n\n')
	table_fancy = tabulate(tab1_sam, tablefmt="pipe", floatfmt=".4f", headers="keys")
	for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
		line = table_fancy.split('\n')[2:][idx]
		table_fancy = table_fancy.replace(line, line.replace(' '+mi+' ', ' **'+mi+'** '))
	file.write(table_fancy+'\n\n')

	file.write('### SAM Statistics for Cuprite Dataset. \n\n')
	table_fancy = tabulate(tab4_sam_stats, tablefmt="pipe", floatfmt=".4f", headers="keys")
	for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
		line = table_fancy.split('\n')[2:][idx]
		table_fancy = table_fancy.replace(line, line.replace(' '+mi+' ', ' **'+mi+'** '))
	file.write(table_fancy+'\n\n')


	# file.write(tabulate(tab1_sam, tablefmt="pipe", headers="keys")+'\n\n')
	file.write('### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using SID for the Cuprite Dataset.\n\n')
	# file.write(tabulate(tab2_sid, tablefmt="pipe", headers="keys")+'\n\n')

	table_fancy = tabulate(tab2_sid, tablefmt="pipe", floatfmt=".4f", headers="keys")
	for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
		line = table_fancy.split('\n')[2:][idx]
		table_fancy = table_fancy.replace(line, line.replace(' '+mi+' ', ' **'+mi+'** '))
	file.write(table_fancy+'\n\n')

	file.write('### SID Statistics for Cuprite Dataset. \n\n')
	table_fancy = tabulate(tab5_sid_stats, tablefmt="pipe", floatfmt=".4f", headers="keys")
	for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
		line = table_fancy.split('\n')[2:][idx]
		table_fancy = table_fancy.replace(line, line.replace(' '+mi+' ', ' **'+mi+'** '))
	file.write(table_fancy+'\n\n')

	# fig = plt.figure()
	for i in range(0, 12):
		fig = plt.figure()
		plt.plot(best_sam_gaee.groundtruth[:,i],label='USGS Library')
		
		for idx, k in enumerate(algo2):
			new_endmember = np.empty((k.groundtruth.shape[0],k.p)) 
			new_endmember[:] = np.nan
			new_endmember[k.selectedBands,:] = k.raw_endmembers
			
			plt.plot(new_endmember[:,i],label=k.name)
		
		plt.title(endmember_names[i])
		
		plt.legend()
		plt.xlabel('wavelength (µm)')
		plt.ylabel('reflectance (%)')
		plt.tight_layout()
		
		plt.savefig('./IMAGES/'+endmember_names[i]+'_Endmember.png', format='png', dpi=200)
		file.write('![alt text](./IMAGES/'+endmember_names[i]+'_Endmember.png)\n\n')


if __name__ == '__main__':
	data_loc = "./DATA/cuprite_data.mat"
	gt_loc = "./DATA/cuprite_groundtruth.mat"

	debug = True
	verbose = False
	num_endm = 12
	thr = 0.8
	nSkewers = 1000
	initSkewers = None
	maxit = 3*num_endm

	npop = [100]
	ngen = [1000]
	cxpb = [0.5, 0.7,1]
	mutpb = [0.05, 0.1, 0.3]
	mrun = 50

	# npop = [100]
	# ngen = [250]
	# cxpb = [1]
	# mutpb = [0.3]
	# mrun = 1

	# npop = [10]
	# ngen = [10]
	# cxpb = [1]
	# mutpb = [0.3]
	# mrun = 2
	
	file = open("README.md","w")
	file.write("# Comparison of Vertex Componet Analysis (VCA) and Genetic Algorithm Endmember Extraction (GAEE) algorithms for Endmember Extraction"+"\n\n")
	file.write("## Douglas Winston R. S., Gustavo T. Laureano, Celso G. Camilo Jr.\n\n")
	file.write("Endmember Extraction is a critical step in hyperspectral image analysis and classification. It is an useful method to decompose a mixed spectrum into a collection of spectra and their corresponding proportions. In this paper, we solve a linear endmember extraction problem as an evolutionary optimization task, maximizing the Simplex Volume in the endmember space. We propose a standard genetic algorithm and a variation with In Vitro Fertilization module (IVFm) to find the best solutions and compare the results with the state-of-art Vertex Component Analysis (VCA) method and the traditional algorithms Pixel Purity Index (PPI) and N-FINDR. The experimental results on real and synthetic hyperspectral data confirms the overcome in performance and accuracy of the proposed approaches over the mentioned algorithms.\n\n")

	file.write('**Envirionment Setup:**\n\n')
	file.write('Monte Carlo runs: %s \n\n' % mrun)
	file.write('Number of endmembers to estimate: %s \n\n' % num_endm)
	file.write('Number of skewers (PPI): %s \n\n' % nSkewers)
	file.write('Maximum number of iterations (N-FINDR): %s \n\n' % maxit)

	if debug:
		run()

	file.close()

	# conf = best_conf(mrun,npop,ngen,cxpb,mutpb)

	# gaee_conf = conf['GAEE'][0]
	# ivfm_conf = conf['GAEE-IVFm'][0]
	# gaee_vca_conf = conf['GAEE-VCA'][0]
	# ivfm_vca_conf = conf['GAEE-IVFm-VCA'][0]

	# print(gaee_conf)
	# print(ivfm_conf)
	# print(gaee_vca_conf)
	# print(ivfm_vca_conf)

	# ppi = DEMO([data_loc,gt_loc,num_endm,'PPI',nSkewers,initSkewers],verbose)
	# nfindr = DEMO([data_loc,gt_loc,num_endm,'NFINDR',maxit],verbose)
	# vca = DEMO([data_loc,gt_loc,num_endm,'VCA'],verbose)
	# gaee = DEMO([data_loc,gt_loc,num_endm,'GAEE',gaee_conf[0],gaee_conf[1],
	# 		gaee_conf[2],gaee_conf[3],None,False],verbose)
	# ivfm = DEMO([data_loc,gt_loc,num_endm,'GAEE-IVFm',ivfm_conf[0],ivfm_conf[1],
	# 		ivfm_conf[2],ivfm_conf[3],None,True],verbose)
	# initPurePixels = vca.ee.extract_endmember()[1]
	# gaee_vca = DEMO([data_loc,gt_loc,num_endm,'GAEE-VCA',gaee_vca_conf[0],gaee_vca_conf[1],
	# 		gaee_vca_conf[2],gaee_vca_conf[3],initPurePixels,False],verbose)
	# ivfm_vca = DEMO([data_loc,gt_loc,num_endm,'GAEE-IVFm-VCA',ivfm_vca_conf[0],ivfm_vca_conf[1],
	# 		ivfm_vca_conf[2],ivfm_vca_conf[3],initPurePixels,True],verbose)

	# endmember_names = ['Alunite','Andradite','Buddingtonite','Dumortierite','Kaolinite_1','Kaolinite_2','Muscovite',
	# 			'Montmonrillonite','Nontronite','Pyrope','Sphene','Chalcedony','**Mean**','**Std**']
	# algo = [ppi, nfindr, vca, gaee, ivfm, gaee_vca, ivfm_vca]

	# tab1_sam = pd.DataFrame()
	# tab1_sam['Endmembers'] = endmember_names
	# tab1_sam.set_index('Endmembers',inplace=True)

	# tab2_sid = pd.DataFrame()
	# tab2_sid['Endmembers'] = endmember_names
	# tab2_sid.set_index('Endmembers',inplace=True)

	# parameters_names = ['Population Size','Number of Generations','Crossover Probability','Mutation Probability']

	# tab3_conf = pd.DataFrame()
	# tab3_conf['Parameters'] = parameters_names
	# tab3_conf.set_index('Parameters',inplace=True)
	# tab3_conf['GAEE'] = gaee_conf
	# tab3_conf['GAEE-IVFm'] = ivfm_conf
	# tab3_conf['GAEE-VCA'] = gaee_vca_conf
	# tab3_conf['GAEE-IVFm-VCA'] = ivfm_vca_conf

	# for l in algo:
	# 	l.best_run(mrun)
	# 	tab1_sam[l.name] = np.append(l.sam_values, [np.mean(l.sam_mean), np.mean(l.sam_std)])
	# 	tab2_sid[l.name] = np.append(l.sid_values, [np.mean(l.sid_mean), np.mean(l.sid_std)])

	# print(tab1_sam)
	# print(tab2_sid)

	# file = open("README.md","w")
	# file.write("# Comparison of Vertex Componet Analysis (VCA) and Genetic Algorithm Endmember Extraction (GAEE) algorithms for Endmember Extraction"+"\n\n")
	# file.write("## Douglas Winston R. S., Gustavo T. Laureano, Celso G. Camilo Jr.\n\n")
	# file.write("Endmember Extraction is a critical step in hyperspectral image analysis and classification. It is an useful method to decompose a mixed spectrum into a collection of spectra and their corresponding proportions. In this paper, we solve a linear endmember extraction problem as an evolutionary optimization task, maximizing the Simplex Volume in the endmember space. We propose a standard genetic algorithm and a variation with In Vitro Fertilization module (IVFm) to find the best solutions and compare the results with the state-of-art Vertex Component Analysis (VCA) method and the traditional algorithms Pixel Purity Index (PPI) and N-FINDR. The experimental results on real and synthetic hyperspectral data confirms the overcome in performance and accuracy of the proposed approaches over the mentioned algorithms.\n\n")

	# file.write('**Envirionment Setup:**\n\n')
	# file.write('Monte Carlo runs: %s \n\n' % mrun)
	# file.write('Number of endmembers to estimate: %s \n\n' % num_endm)
	# file.write('Number of skewers (PPI): %s \n\n' % nSkewers)
	# file.write('Maximum number of iterations (N-FINDR): %s \n\n' % maxit)

	
	# file.write('### Parameters used in each GAEE versions\n\n')
	# file.write(tabulate(tab3_conf, tablefmt="pipe", headers="keys")+'\n\n')

	# file.write('### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using SAM for the Cuprite Dataset.\n\n')
	# file.write(tabulate(tab1_sam, tablefmt="pipe", headers="keys")+'\n\n')
	# file.write('### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using SID for the Cuprite Dataset.\n\n')
	# file.write(tabulate(tab2_sid, tablefmt="pipe", headers="keys")+'\n\n')
