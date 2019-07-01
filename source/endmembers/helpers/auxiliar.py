from sklearn.decomposition import PCA
import numpy as np
import numpy
from scipy import linalg
import scipy as sp
import itertools
import matplotlib.pyplot as plt
import math
from functools import reduce
import spectral.io.envi as envi
import scipy.io as sio
import scipy.stats as ss
from tqdm import tqdm
import seaborn as sns
import pandas as pd


def _PCA_transform(M, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(M)

def estimate_snr(Y,r_m,x):
    [L, N] = Y.shape
    [p, N] = x.shape
    P_y     = sp.sum(Y**2)/float(N)
    P_x     = sp.sum(x**2)/float(N) + sp.sum(r_m**2)
    snr_est = 10*sp.log10( (P_x - p/L*P_y)/(P_y - P_x) )
    return snr_est

def ATGP(data, q):
    nsamples, nvariables = data.shape
    max_energy = -1
    idx = 0
    for i in range(nsamples):
        r = data[i]
        val = np.dot(r, r)
        if val > max_energy:
          max_energy = val
          idx = i
    E = np.zeros((q, nvariables), dtype=np.float32)
    E[0] = data[idx] # the first endmember selected
    I = np.eye(nvariables)
    IDX = np.zeros(q, dtype=np.int)
    IDX[0] = idx
    for i in range(q-1):
        UC = E[0:i+1]
        PU = I - np.dot(UC.T,np.dot(linalg.pinv(np.dot(UC,UC.T)),UC))
        max_energy = -1
        idx = 0
        for j in range(nsamples):
            r = data[j]
            result = np.dot(PU, r)
            val = np.dot(result.T, result)
            if val > max_energy:
                max_energy = val
                idx = j
        E[i+1] = data[idx]
        IDX[i+1] = idx
    return E, IDX

def plot_endmembers_comparison(M, G, q, algo_name, names, color_m, color_g, dimensions):
    k = 0
    f, axarr = plt.subplots(dimensions[0], dimensions[1])
    for i in range(0,dimensions[0]):
        for j in range(0,dimensions[1]):
            axarr[i,j].axis('off')
            if k < q:
                l1,=axarr[i,j].plot(M[:,k], color_m)
                l2,=axarr[i,j].plot(G[:,k], color_g)
                axarr[i,j].set_title(names[k])
                axarr[i,j].legend([l1,l2], [algo_name, "USGS Library"])           
                axarr[i,j].axis('off')
            k+=1
    f.tight_layout()

def plot_envi_image(img, name, band):
    plt.matshow(img[:,:,band],cmap=plt.get_cmap('Spectral_r'))
    plt.axis('off')
    plt.title(name+" "+ str(band)+'th band')

def plot_convergence(generations_fitness,number_generation):
    plt.figure()
    plt.plot( list(range(0,number_generation)),generations_fitness)
    plt.title("GAEE Convergence")
    plt.ylabel('log10(simplex_volume)')
    plt.xlabel('number of generations')
    plt.tight_layout()

def plot_dual_convergence(generations_fitness_1,generations_fitness_2,number_generation):
    normalized_generations_fitness_1 = (generations_fitness_1-min(generations_fitness_1))/(max(generations_fitness_1)-min(generations_fitness_1))
    normalized_generations_fitness_2 = (generations_fitness_2-min(generations_fitness_2))/(max(generations_fitness_2)-min(generations_fitness_2))
    fig, ax1 = plt.subplots()
    ax1.set_title("GAEE Convergence")
    ax1.plot(generations_fitness_1, 'b')
    ax1.set_xlabel('number of generations')
    ax1.set_ylabel('log10(simplex_volume)', color='b')
    ax2 = ax1.twinx()
    ax2.plot(generations_fitness_2, 'r')
    ax2.set_ylabel('log(SID)', color='r')
    fig.tight_layout()

def plot_genes(envi, population, number_rows, band, number_generations):
    # k = 0
    # step = int(number_generations/4)
    # f, axarr = plt.subplots(r, c)
    # for i in range(0,r):
    #     for j in range(0,c):
    #         population_genes = list(reduce(list.__add__,population[k]))
    #         axarr[i,j].set_title('Generation'+str(k))
    #         axarr[i,j].imshow(envi[:,:,band].T,cmap=plt.get_cmap('Spectral_r'))
    #         x = list(map(lambda z: math.floor(z/number_rows),population_genes))
    #         y = list(map(lambda z: math.floor(z%number_rows),population_genes))
    #         axarr[i,j].plot(y,x,'.b')
    #         k+=step
    # f.tight_layout()
    for k in range(0,number_generations):    
        population_genes = list(reduce(list.__add__,population[k]))
        plt.figure()
        plt.title('Generation '+str(k))
        plt.imshow(envi[:,:,band].T,cmap=plt.get_cmap('Spectral_r'))
        x = list(map(lambda z: math.floor(z/number_rows),population_genes))
        y = list(map(lambda z: math.floor(z%number_rows),population_genes))
        plt.plot(y,x,'.b')

def SID(s1, s2):
    p = (s1 / numpy.sum(s1)) + numpy.spacing(1)
    q = (s2 / numpy.sum(s2)) + numpy.spacing(1)
    return numpy.sum(p * numpy.log(p / q) + q * numpy.log(q / p))

def SAM (s1, s2):
    try:
        s1_norm = math.sqrt(numpy.dot(s1, s1))
        s2_norm = math.sqrt(numpy.dot(s2, s2))
        sum_s1_s2 = numpy.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        return 0.0
    return angle

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def chebyshev(s1, s2):
    return numpy.amax(numpy.abs(s1 - s2))

def NormXCorr(s1, s2):
    s = s1.shape[0]
    corr = numpy.sum((s1 - numpy.mean(s1)) * (s2 - numpy.mean(s2))) / (ss.tstd(s1) * ss.tstd(s2))
    return corr * (1./(s-1))

def load_cuprite_dataset():
    envi_data = envi.open('../data/real/cuprite/cuprite.hdr')
    img = envi_data[:,:,:]
    hsi_path = '../data/real/cuprite/cuprite.mat'
    ground_truth_path = '../data/real/cuprite/cuprite.gt'
    hsi = sio.loadmat(hsi_path)
    Y = hsi['Y'][1:,:]
    ground_truth = sio.loadmat(ground_truth_path)['M']
    ground_truth = ground_truth[1:,:]
    endmembers_names = ['Alunite', 'Andradite', 'Buddingtonite', 'Dumortierite',
     'Kaolinite_1', 'Kaolinite_2', 'Muscovite', 'Montmorillonite',
     'Nontronite', 'Pyrope', 'Sphene', 'Chalcedony']
    number_rows = hsi['nRow'][0][0]
    number_columns = hsi['nCol'][0][0]
    number_pixels = int(number_rows)*int(number_columns)
    number_endmembers = ground_truth.shape[1]
    number_bands = Y.shape[0]
    dimensions = [number_rows, number_columns, number_bands, number_pixels]
    selected_bands = hsi['SlectBands'].T[0]
    selected_bands = selected_bands[1:]
    null_bands = list(set(list(range(0,len(ground_truth)))) - set(selected_bands))
    ground_truth_plot =  ground_truth.copy()
    ground_truth_plot[null_bands,:] = None
    bandsSelection = [selected_bands, null_bands]
    ground_truth = ground_truth[selected_bands]
    dimensions_plot = [4, 3]
    ground_truth = normalize_data(ground_truth)
    return Y, img, dimensions, number_endmembers, ground_truth, ground_truth_plot, endmembers_names, dimensions_plot, bandsSelection, 'Cuprite'

def load_samson_dataset():
    envi_data = envi.open('../data/real/samson/samson.hdr')
    img = envi_data[:,:,:]
    hsi_path = '../data/real/samson/samson.mat'
    ground_truth_path = '../data/real/samson/samson.gt'
    hsi = sio.loadmat(hsi_path)
    Y = hsi['V']
    ground_truth = sio.loadmat(ground_truth_path)['M']
    endmembers_names = ['Rock', 'Tree', 'Water']
    number_rows = hsi['nRow'][0][0]
    number_columns = hsi['nCol'][0][0]
    number_pixels = int(number_rows)*int(number_columns)
    number_endmembers = ground_truth.shape[1]
    number_bands = Y.shape[0]
    dimensions = [number_rows, number_columns, number_bands, number_pixels]
    dimensions_plot = [2, 2]
    selected_bands = list(range(0,number_bands))
    bandsSelection = [selected_bands, []]
    ground_truth = normalize_data(ground_truth)
    return Y, img, dimensions, number_endmembers, ground_truth, ground_truth, endmembers_names, dimensions_plot, bandsSelection, 'Samson'

def load_jasper_dataset():
    envi_data = envi.open('../data/real/jasper/jasper.hdr')
    img = envi_data[:,:,:]
    hsi_path = '../data/real/jasper/jasper.mat'
    ground_truth_path = '../data/real/jasper/jasper.gt'
    hsi = sio.loadmat(hsi_path)
    Y = hsi['Y'][:,:]
    ground_truth = sio.loadmat(ground_truth_path)['M']
    endmembers_names = ['Tree', 'Water', 'Dirt', 'Road']
    number_rows = hsi['nRow'][0][0]
    number_columns = hsi['nCol'][0][0]
    number_pixels = int(number_rows)*int(number_columns)
    number_endmembers = ground_truth.shape[1]
    number_bands = Y.shape[0]
    dimensions = [number_rows, number_columns, number_bands, number_pixels]
    selected_bands = hsi['SlectBands'].T[0]
    null_bands = list(set(list(range(0,len(ground_truth)))) - set(selected_bands))
    ground_truth_plot =  ground_truth.copy()
    ground_truth_plot[null_bands,:] = None
    bandsSelection = [selected_bands, null_bands]
    dimensions_plot = [2, 2]
    ground_truth = normalize_data(ground_truth)
    return Y, img, dimensions, number_endmembers, ground_truth, ground_truth_plot, endmembers_names, dimensions_plot, bandsSelection, 'Jasper'

def load_urban_dataset():
    envi_data = envi.open('../data/real/urban/urban.hdr')
    img = envi_data[:,:,:]
    hsi_path = '../data/real/urban/urban.mat'
    ground_truth_path = '../data/real/urban/urban.gt'
    hsi = sio.loadmat(hsi_path)
    Y = hsi['Y'][:,:]
    ground_truth = sio.loadmat(ground_truth_path)['M']
    endmembers_names = ['Asphalt Road', 'Grass', 'Tree', 'Roof', 'Metal', 'Dirt']
    number_rows = hsi['nRow'][0][0]
    number_columns = hsi['nCol'][0][0]
    number_pixels = int(number_rows)*int(number_columns)
    number_endmembers = ground_truth.shape[1]
    number_bands = Y.shape[0]
    dimensions = [number_rows, number_columns, number_bands, number_pixels]
    selected_bands = hsi['SlectBands'].T[0]
    null_bands = list(set(list(range(0,len(ground_truth)))) - set(selected_bands))
    ground_truth_plot =  ground_truth.copy()
    ground_truth_plot[null_bands,:] = None
    bandsSelection = [selected_bands, null_bands]
    dimensions_plot = [2, 3]
    ground_truth = normalize_data(ground_truth)
    return Y, img, dimensions, number_endmembers, ground_truth, ground_truth_plot, endmembers_names, dimensions_plot, bandsSelection, 'Urban'

def load_legendre_dataset(noise):
    hsi_path = '../data/synthetic/legendre/legendre'+str(noise)+'.mat'
    hsi = sio.loadmat(hsi_path)
    img = hsi['syntheticImage'][:,:]
    ground_truth = hsi['endmembersGT'].T
    endmembers_names = ['Brick', 'Sheet Metal', 'Asphalt', 'Fiberglass', 'Vinyl Plastic']
    number_rows = img.shape[0]
    number_columns = img.shape[1]
    number_pixels = int(number_rows)*int(number_columns)
    number_endmembers = ground_truth.shape[1]
    number_bands = img.shape[2]
    dimensions = [number_rows, number_columns, number_bands, number_pixels]
    Y = img.reshape((number_pixels,number_bands))
    selected_bands = list(range(0,number_bands))
    bandsSelection = [selected_bands, []]
    dimensions_plot = [2, 3]
    ground_truth = normalize_data(ground_truth)
    return Y.T, img, dimensions, number_endmembers, ground_truth, ground_truth, endmembers_names, dimensions_plot, bandsSelection, 'Legendre_'+str(noise)

def normalize_data (matrix):
    matrix_norm = []
    for i in range(0,matrix.shape[1]):
        a = matrix[:,i].copy()
        normalized = (a-min(a))/(max(a)-min(a))
        matrix_norm.append(normalized)
    return np.vstack(matrix_norm).T

def best_match_execution(M, G):
    sam_matrix = []
    for k in range(0,M.shape[1]):
        sam = []
        m_0 = M[:,k]
        # plt.plot(m_0)
        for i in range(0,G.shape[1]):
            g_0 = G[:,i]
            sam.append(NormXCorr(m_0, g_0))
        sam_matrix.append(sam)
    best_index = np.argmax(np.matrix(sam_matrix), axis=0).tolist()[0]
    best_values = np.max(np.matrix(sam_matrix), axis=0).tolist()[0]
    return M[:,best_index]

def monte_carlo (results, Y, dimensions, number_endmembers, algo, parameter,algo_name, ground_truth, endmembers_names, dataset_name, n):
    Y = normalize_data(Y)
    df_mc = []
    M_list = []
    ground_truth = normalize_data(ground_truth)
    sam_matrix = []
    # plt.figure()
    # plt.title(algo_name)
    for i in tqdm(range(0,n)):
        M, duration, other = algo(Y, dimensions, number_endmembers, parameter)
        M_norm = normalize_data(M)
        M_best = best_match_execution(M_norm,ground_truth)
        M_list.append(M_best)
        sam_list = []
        for k in range(0,number_endmembers):
            m_0 = M_best[:,k]
            g_0 = ground_truth[:,k]
            aux_sam = SAM(m_0, g_0)*10**2
            df_mc.append([m_0,aux_sam, 'sam', endmembers_names[k], i, algo_name, dataset_name, duration])
            aux_rmse = RMSE(m_0, g_0)*10**2
            df_mc.append([m_0,aux_rmse, 'rmse', endmembers_names[k], i, algo_name, dataset_name, duration])
            sam_list.append(aux_sam)
        sam_matrix.append(sam_list)
    sam_matrix = np.matrix(sam_matrix)
    mu = sam_matrix.mean(axis = 0)*10**2
    std = sam_matrix.std(axis = 0)*10**2
    M_all = np.hstack(M_list)
    index_best = np.argmin(sam_matrix.mean(axis = 1))
    M_aux = M_list[index_best]
    M_aux = normalize_data(M_aux)
    results.extend(df_mc)
    labels = ['Data', 'Value', 'Metric', 'Endmember', 'Execution', 'Algorithm', 'Dataset', 'Duration']
    results_df = pd.DataFrame.from_records(results, columns=labels)
    return M_aux, results, results_df

def plot_total_mean_accurary(results_df, dataset_name):
    plt.figure()
    sns.barplot(x="Value", y="Algorithm", hue="Metric", data=results_df)
    plt.title(dataset_name+" Dataset - Mean Accuracy 10^-2")

def plot_endmembers_mean_accurary(results_df, dataset_name):
    plt.figure()
    sns.barplot(x="Value", y="Algorithm", hue="Endmember", data=results_df)
    plt.title(dataset_name+" Dataset - Endmember Accuracy 10^-2")

def plot_mean_computing_time(results_df, dataset_name):
    plt.figure()
    sns.barplot(x="Duration", y="Algorithm", data=results_df)
    plt.title(dataset_name+" Dataset - Computing Time (sec)")

def plot_endmembers(M_list, G, algorith_name, number_endmembers, endmembers_names):
    for end_indx in range(0,number_endmembers):
        plt.figure()
        plt.title(endmembers_names[end_indx])
        plt.plot(G[:,end_indx], label="USGS Library", color='black')
        k = 0
        for m in M_list:
            plt.plot(m[:,end_indx], label=algorith_name[k])
            k+=1
        plt.legend()

def convergence_volume_erro(individuals,Y,ground_truth,number_endmembers):
    from numpy import linalg as la

    def princomp(A):
        M = (A-np.mean(A.T, axis=1)).T
        [latent,coeff] = la.eig(np.cov(M))
        score = np.dot(coeff.T,M)
        return coeff, score, latent
    def simplex_volume(indices, data_pca, number_endmembers):
        TestMatrix = np.zeros((number_endmembers,number_endmembers))
        for i in range(0,number_endmembers):
            TestMatrix[0:number_endmembers,i] = data_proj[:,int(indices[i])]
        volume = np.abs(np.linalg.det(TestMatrix))
        return volume

    data = np.asarray(Y)
    _coeff, score, _latent = princomp(data.T)
    data_proj = np.squeeze(score[0:number_endmembers,:])
    
    volume= []
    erro = []
    for result in individuals:
        M = data[:,list(result)]
        M = normalize_data(M)
        ground_truth = normalize_data(ground_truth)
        best_m = best_match_execution(M,ground_truth)
        sam_list = []

        for k in range(0,number_endmembers):
            m_0 = best_m[:,k]
            g_0 = ground_truth[:,k]
            aux_sam = SAM(m_0, g_0)*10**2
            sam_list.append(aux_sam)

        volume.append(np.log10(simplex_volume(result,data_proj,number_endmembers)))
        erro.append(np.array(sam_list).mean())
        
    from scipy.signal import savgol_filter
    def round_to_odd(f):
        f = int(np.floor(f-1))
        return f + 1 if f % 2 == 0 else f
    
    volume_erro = list(zip(volume,erro))
    volume_erro = sorted(volume_erro, key=lambda x: x[0])
    volume = list(zip(*volume_erro))[0]
    erro = list(zip(*volume_erro))[1]
    erro_sg = savgol_filter(erro, round_to_odd(len(volume)), 1)
    
    return volume, erro_sg