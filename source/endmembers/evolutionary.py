import numpy
import scipy.io as sio
import numpy.linalg as la
import matplotlib.pyplot as plt
import random
import time
import math
import tqdm
import itertools
import helpers.auxiliar as auxiliar
from deap import base
from deap import creator
from deap import tools
from functools import reduce
import numpy.matlib
from scipy.sparse.linalg import svds, eigs
from collections import deque
from sklearn.metrics import mean_squared_error
from math import sqrt
import classical

def princomp(A):
    M = (A-numpy.mean(A.T, axis=1)).T
    [latent,coeff] = la.eig(numpy.cov(M))
    score = numpy.dot(coeff.T,M)
    return coeff, score, latent

def generate_individual(creator, number_endmembers, number_pixels, number_rows, number_columns):

	individual = numpy.random.choice(list(range(0,number_pixels)), number_endmembers)

	return creator.individual(individual)

def maximize_simplex_volume(indices, data_pca, number_endmembers):
    TestMatrix = numpy.zeros((number_endmembers,number_endmembers))
    TestMatrix[0,:] = 1
    for i in range(0,number_endmembers):
        TestMatrix[1:number_endmembers,i] = data_pca[:,int(indices[i])]
    volume = numpy.abs(numpy.linalg.det(TestMatrix))
    return volume

def mono_fitness (indices, data_pca, number_endmembers):
	return (maximize_simplex_volume(indices, data_pca, number_endmembers),)

def random_mutation(individual, number_pixels, number_endmembers, mutation_probability):
	new_individual =  random.sample(set(individual + random.sample(range(0,number_pixels), number_endmembers)), number_endmembers)
	mutant = [new_individual[i] if random.random() < mutation_probability else individual[i] for i in range(0,number_endmembers)]
	return mutant


def GAEE (data, dimensions, number_endmembers):
	start_time = time.time()
	population_size = 100
	number_generations = 100
	crossing_probability = 0.7
	mutation_probability = 0.5
	stop_criteria_MAX = 20
	random.seed(64)

	number_rows = int(dimensions[0])
	number_columns = int(dimensions[1])
	number_bands = int(dimensions[2])
	number_pixels = number_rows*number_columns

	tournsize = 3

	data = numpy.asarray(data)
	_coeff, score, _latent = princomp(data.T)
	data_pca = numpy.squeeze(score[0:number_endmembers-1,:])
	
	creator.create("max_fitness", base.Fitness, weights=(1.0,))
	creator.create("individual", list, fitness=creator.max_fitness)

	toolbox = base.Toolbox()
	toolbox.register("create_individual", generate_individual, creator, number_endmembers,number_pixels,number_rows, number_columns)
	toolbox.register("initialize_population", tools.initRepeat, list, toolbox.create_individual)
	toolbox.register("evaluate_simplex_volume", mono_fitness, data_pca=data_pca, number_endmembers=number_endmembers)
	
	toolbox.register("cross_twopoints", tools.cxTwoPoint)

	toolbox.register("tournament_select", tools.selTournament, tournsize=tournsize)
	toolbox.register("random_mutation", random_mutation, number_pixels=number_pixels, number_endmembers=number_endmembers, mutation_probability=mutation_probability)

	population = toolbox.initialize_population(n=population_size)

	population_fitnesses = [toolbox.evaluate_simplex_volume(individual) for individual in population]

	for individual, fitness in zip(population, population_fitnesses):
		individual.fitness.values = fitness

	hof = tools.HallOfFame(10)
	hof.update(population)
	
	current_generation = 0
	generations_fitness = []
	generations_population = []
	stop_criteria = deque( maxlen=stop_criteria_MAX)
	stop_criteria.extend(list(range(1,stop_criteria_MAX)))

	while current_generation < number_generations and numpy.var(numpy.array(stop_criteria)) > 0.000001:

		offspring = list(map(toolbox.clone, population))

		# Crossing
		for child_1, child_2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < crossing_probability:
				toolbox.cross_twopoints(child_1, child_2)
				del child_1.fitness.values
				del child_2.fitness.values
		# Mutation
		for mutant in offspring:
			if random.random() < mutation_probability:
				toolbox.random_mutation(mutant)
				del mutant.fitness.values

		# Fitness
		offspring_fitnesses = [toolbox.evaluate_simplex_volume(individual) for individual in offspring]
		for individual, fitness in zip(offspring, offspring_fitnesses):
			individual.fitness.values = fitness

		new_population = population+offspring

		# Selection
		selected = toolbox.tournament_select(new_population, population_size)
		population_selected = list(map(toolbox.clone, selected))

		population[:] = population_selected

		hof.update(population)

		# Statistics
		fits = [ind.fitness.values[0] for ind in population]
		mean_offspring = sum(fits) / len(population)
		generations_fitness.append(numpy.log10(mean_offspring))
		generations_population.append(population.copy())

		stop_criteria.append(numpy.log10(mean_offspring))

		current_generation+=1

	best_individual = tools.selBest(population, 1)[0]

	M = data[:,best_individual]
	duration = time.time() - start_time

	return M, duration, [best_individual, generations_fitness, generations_population, current_generation, hof]

def GAEE_IVFm (data, dimensions, number_endmembers):
	start_time = time.time()

	number_rows = int(dimensions[0])
	number_columns = int(dimensions[1])
	number_bands = int(dimensions[2])
	number_pixels = number_rows*number_columns

	tournsize = 3

	data = numpy.asarray(data)
	_coeff, score, _latent = princomp(data.T)
	data_pca = numpy.squeeze(score[0:number_endmembers-1,:])
	
	creator.create("max_fitness", base.Fitness, weights=(1.0,))
	creator.create("individual", list, fitness=creator.max_fitness)

	toolbox = base.Toolbox()
	toolbox.register("create_individual", generate_individual, creator, number_endmembers,number_pixels,number_rows, number_columns)
	toolbox.register("initialize_population", tools.initRepeat, list, toolbox.create_individual)
	toolbox.register("evaluate_simplex_volume", mono_fitness, data_pca=data_pca, number_endmembers=number_endmembers)
	
	toolbox.register("cross_twopoints", tools.cxTwoPoint)

	toolbox.register("tournament_select", tools.selTournament, tournsize=tournsize)
	toolbox.register("random_mutation", random_mutation, number_pixels=number_pixels, number_endmembers=number_endmembers, mutation_probability=mutation_probability)

	population = toolbox.initialize_population(n=population_size)

	population_fitnesses = [toolbox.evaluate_simplex_volume(individual) for individual in population]

	for individual, fitness in zip(population, population_fitnesses):
		individual.fitness.values = fitness
	
	current_generation = 0
	generations_fitness = []
	generations_population = []
	stop_criteria = deque( maxlen=5)
	stop_criteria.extend([1,2,3,4,5])

	while current_generation < number_generations and numpy.var(numpy.array(stop_criteria)) > 0.000001:

		offspring = list(map(toolbox.clone, population))

		# Crossing
		for child_1, child_2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < crossing_probability:
				toolbox.cross_twopoints(child_1, child_2)
				del child_1.fitness.values
				del child_2.fitness.values
		# Mutation
		for mutant in offspring:
			if random.random() < mutation_probability:
				toolbox.random_mutation(mutant)
				del mutant.fitness.values
		
		# Fitness
		offspring_fitnesses = [toolbox.evaluate_simplex_volume(individual) for individual in offspring]
		for individual, fitness in zip(offspring, offspring_fitnesses):
			individual.fitness.values = fitness

		# In Vitro Fertilization Module
		ivfoffspring = list(map(toolbox.clone, population))
		ivffits = [ind.fitness.values[0] for ind in ivfoffspring]
		fatheridx = numpy.argmax(ivffits)		
		fatherfit = numpy.max(ivffits)
		father = creator.individual(ivfoffspring[fatheridx].copy())

		for ind in ivfoffspring[::2]:
			toolbox.random_mutation(ind)
			del ind.fitness.values

		for child1 in ivfoffspring:
			child2 = creator.individual(father.copy())
			toolbox.cross_twopoints(child1, child2)
			del child1.fitness.values
			del child2.fitness.values

		ivffitnesses = [toolbox.evaluate_simplex_volume(ind) for ind in ivfoffspring]
		for ind, fit in zip(ivfoffspring, ivffitnesses):
			ind.fitness.values = fit
		
		popmax = max(offspring_fitnesses)
		for ind in ivfoffspring:
				if (ind.fitness.values >= popmax):
					population.append(ind)

		new_population = population+offspring

		# Selection
		selected = toolbox.tournament_select(new_population, population_size)
		population_selected = list(map(toolbox.clone, selected))
		
		population[:] = population_selected

		# Statistics
		fits = [ind.fitness.values[0] for ind in population]
		mean_offspring = sum(fits) / len(population)
		generations_fitness.append(numpy.log10(mean_offspring))
		generations_population.append(population.copy())

		stop_criteria.append(numpy.log10(mean_offspring))

		current_generation+=1

	best_individual = tools.selBest(population, 1)[0]
	# print(best_individual)

	M = data[:,best_individual]
	duration = time.time() - start_time

	return M, duration, [generations_fitness, generations_population, current_generation]

def gaussian_mutation(individual, toolbox, number_rows, number_columns):
    gene_x = (numpy.array(individual) / number_rows).astype(int)
    gene_y = (numpy.array(individual) % number_rows)
    
    mut_x = abs(toolbox.gaussian_mutation_op(gene_x.copy())[0] % number_columns-1)
    mut_y = abs(toolbox.gaussian_mutation_op(gene_y.copy())[0] % number_rows-1)
    
    mutant = mut_x*number_rows+mut_y
    return mutant

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

def chebyshev(s1, s2):
    return numpy.amax(numpy.abs(s1 - s2))

def RMSE(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())

def NormXCorr(s1, s2):
    import scipy.stats as ss
    s = s1.shape[0]
    corr = numpy.sum((s1 - numpy.mean(s1)) * (s2 - numpy.mean(s2))) / (ss.tstd(s1) * ss.tstd(s2))
    return corr * (1./(s-1))

def affine_projection(Y,p):
	L, N = Y.shape
	Y = numpy.matrix(Y)
	my = Y.mean(axis=1)
	Y = Y - numpy.matlib.repmat(my, 1, N)
	Up, D, _vt = svds(Y*Y.T/N, k=p-1)
	Up = numpy.matrix(Up)
	Y = Up*Up.T*Y
	Y = Y + numpy.matlib.repmat(my,1,N)
	my_ortho = my-Up*Up.T*my
	aux = my_ortho/numpy.sqrt(sum(numpy.square(my_ortho)))
	aux = aux.reshape((aux.shape[0],1))
	Up = numpy.concatenate((Up, aux), axis=1)
	Y = Up.T*Y
	return Y

def affine_tranform(Y,p):
	from numpy import linalg as la
	L, N = Y.shape
	d = Y.mean(axis=1)
	U = Y-(d*numpy.ones((L,N)).T).T
	R = numpy.dot(U,U.T)
	D,eV = la.eig(R)
	C = eV[:,(L-p+1):]
	Xd = numpy.dot(la.pinv(C), U)
	Xc =  numpy.concatenate((Xd, -1*numpy.ones((1,Xd.shape[1]))), axis=0)
	return Xc

def minimize_simplex_volume(indices, data_proj, number_endmembers):
	TestMatrix = numpy.zeros((number_endmembers,number_endmembers))
	for i in range(0,number_endmembers):
		TestMatrix[0:number_endmembers,i] = data_proj[:,int(indices[i])]
	volume = numpy.abs(numpy.linalg.det(TestMatrix))
	# print("volume", volum÷e)
	return volume

def minimize_similarity(indices, data, number_endmembers):
	half1_indices = indices[:int(len(indices)/2)]
	half2_indices = indices[int(len(indices)/2):]
	combinations_indices = list(zip(half1_indices,half2_indices))
	# combinations_indices = list(itertools.combinations(indices, 2))
	sum_end = 0
	min_list = []
	for comb in combinations_indices:
		spec_a = data[:,comb[0]]
		spec_b = data[:,comb[1]]
		normalized_a = (spec_a-min(spec_a))/(max(spec_a)-min(spec_a))
		normalized_b = (spec_b-min(spec_b))/(max(spec_b)-min(spec_b))
		calc = RMSE(normalized_a,normalized_b)
		# calc = RMSE(spec_a,spec_b)
		min_list.append(calc)
	
	sum_end = numpy.sum(min_list)
	return sum_end

def UCLS(Y, M):
    """
    Performs unconstrained least squares abundance estimation.

    Parameters:
        M: `numpy array`
            2D data matrix (N x p).

        U: `numpy array`
            2D matrix of endmembers (q x p).

    Returns: `numpy array`
        An abundance maps (N x q).
     """
    Uinv = numpy.linalg.pinv(M.T)
    A = numpy.dot(Uinv, Y[0:,:].T).T
    A[A<0] = 0
    return numpy.sum(A,axis=0).tolist()

def multi_fitness (indices, data, data_proj, number_endmembers):
	M = data[:,indices]
	abundances = UCLS(data.T, M.T)
	obj = []
	obj.extend([minimize_similarity(indices, data_proj, number_endmembers)])
	obj.extend([numpy.mean(abundances)])
	return (obj[0],)


def GAEEII (data, dimensions, number_endmembers):
	start_time = time.time()
	population_size = 10
	number_generations = 100
	crossing_probability = 1
	mutation_probability = 0.5
	stop_criteria_MAX = 20
	random.seed(64)	

	number_endmembers = number_endmembers

	number_rows = int(dimensions[0])
	number_columns = int(dimensions[1])
	number_bands = int(dimensions[2])
	
	number_pixels = number_rows*number_columns

	sigma_MAX = max(number_rows, number_generations)
	
	data_proj = numpy.asarray(affine_tranform(data, number_endmembers))

	# data = numpy.asarray(data)
	# _coeff, score, _latent = princomp(data.T)
	# data_proj = numpy.squeeze(score[0:number_endmembers,:])
	
	creator.create("min_fitness", base.Fitness, weights=(1.0,))
	creator.create("individual", list, fitness=creator.min_fitness)

	toolbox = base.Toolbox()
	toolbox.register("create_individual", generate_individual, creator, number_endmembers,number_pixels, number_rows, number_columns)
	toolbox.register("initialize_population", tools.initRepeat, list, toolbox.create_individual)
	toolbox.register("evaluate_individual", multi_fitness, data=data, data_proj=data_proj, number_endmembers=number_endmembers)
	
	toolbox.register("cross_twopoints", tools.cxTwoPoint)
	toolbox.register("selNSGA2", tools.selNSGA2)

	toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=0, indpb=mutation_probability)
	toolbox.register("gaussian_mutation", gaussian_mutation, toolbox=toolbox, number_rows=number_rows, number_columns=number_columns)

	population = toolbox.initialize_population(n=population_size)
	
	ensemble_pop = []
	for i in range(0,5):
		[_, duration, other] = classical.VCA(data, dimensions, number_endmembers)
		# print(other[0])
		# print('Time GAEE:',duration)
		ensemble_pop.append(creator.individual(other[0]))

	population[5:] = ensemble_pop

	population_fitnesses = [toolbox.evaluate_individual(individual) for individual in population]

	for individual, fitness in zip(population, population_fitnesses):
		individual.fitness.values = fitness
	
	hof = tools.HallOfFame(3)
	hof.update(population)

	current_generation = 0
	current_sigma = sigma_MAX
	generations_fitness_1 = []
	generations_fitness_2 = []
	generations_population = []
	stop_criteria = deque( maxlen=stop_criteria_MAX)
	stop_criteria.extend(list(range(1,stop_criteria_MAX)))

	while current_generation < number_generations and numpy.var(numpy.array(stop_criteria)) > 0.000001:

		toolbox.unregister("gaussian_mutation_op")
		toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=current_sigma, indpb=mutation_probability)
		
		offspring = tools.selRandom(population,k=int(population_size/2))
		offspring = list(map(toolbox.clone, offspring))

		# Crossing
		for child_1, child_2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < crossing_probability:
				toolbox.cross_twopoints(child_1, child_2)
				del child_1.fitness.values
				del child_2.fitness.values
		# Mutation
		for mutant in offspring:
			if random.random() < mutation_probability:
				toolbox.gaussian_mutation(mutant)
				del mutant.fitness.values
		
		# Fitness
		offspring_fitnesses = [toolbox.evaluate_individual(individual) for individual in offspring]
		for individual, fitness in zip(offspring, offspring_fitnesses):
			individual.fitness.values = fitness

		new_population = population+offspring

		# Selection
		selected = toolbox.selNSGA2(new_population, population_size)
		selected = list(map(toolbox.clone, selected))

		population[:] = selected

		hof.update(population)

		# Statistics
		fits_1 = [ind.fitness.values[0] for ind in population]
		# fits_2 = [ind.fitness.values[1] for ind in population]

		mean_1_offspring = sum(fits_1) / len(population)
		# mean_2_offspring = sum(fits_2) / len(population)

		generations_fitness_1.append(numpy.log10(mean_1_offspring))
		# generations_fitness_2.append(numpy.log(mean_2_offspring))
		
		generations_population.append(population.copy())
		
		stop_criteria.append(numpy.log10(mean_1_offspring))

		current_generation+=1
		current_sigma = sigma_MAX/((current_generation+1)/4)
		
	best_individual = tools.selBest(population, 1)[0]
	# best_individual = hof[0]

	# print('Result:', multi_fitness(best_individual,data, data_proj,number_endmembers))

	M = data[:,best_individual]
	
	duration = time.time() - start_time

	return M, duration, [generations_fitness_1, generations_fitness_2 ,generations_population, current_generation]


def GAEEII_IVFm (data, dimensions, number_endmembers):
	start_time = time.time()

	number_rows = int(dimensions[0])
	number_columns = int(dimensions[1])
	number_bands = int(dimensions[2])
	
	number_pixels = number_rows*number_columns

	sigma_MAX = max(number_rows, number_generations)
	
	data_proj = numpy.asarray(affine_projection(data, number_endmembers))
	
	creator.create("max_fitness", base.Fitness, weights=(-1.0, -1.0))
	creator.create("individual", list, fitness=creator.max_fitness)

	toolbox = base.Toolbox()
	toolbox.register("create_individual", generate_individual, creator, number_endmembers,number_pixels, number_rows, number_columns)
	toolbox.register("initialize_population", tools.initRepeat, list, toolbox.create_individual)
	toolbox.register("evaluate_individual", multi_fitness, data=data, data_proj=data_proj, number_endmembers=number_endmembers)
	
	toolbox.register("cross_twopoints", tools.cxTwoPoint)
	toolbox.register("selNSGA2", tools.selNSGA2)

	toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=0, indpb=mutation_probability)
	toolbox.register("gaussian_mutation", gaussian_mutation, toolbox=toolbox, number_rows=number_rows, number_columns=number_columns)
	toolbox.register("random_mutation", random_mutation, number_pixels=number_pixels, number_endmembers=number_endmembers, mutation_probability=mutation_probability)

	population = toolbox.initialize_population(n=population_size)

	population_fitnesses = [toolbox.evaluate_individual(individual) for individual in population]

	for individual, fitness in zip(population, population_fitnesses):
		individual.fitness.values = fitness

	hof = tools.HallOfFame(3)
	hof.update(population)
	
	current_generation = 0
	current_sigma = sigma_MAX
	generations_fitness_1 = []
	generations_fitness_2 = []
	generations_population = []
	stop_criteria = deque( maxlen=5)
	stop_criteria.extend([1,2,3,4,5])

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean, axis=0)
	stats.register("std", numpy.std, axis=0)
	stats.register("min", numpy.min, axis=0)
	stats.register("max", numpy.max, axis=0)
    
	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"

	record = stats.compile(population)
	logbook.record(gen=0, evals=len(population), **record)

	while current_generation < number_generations and numpy.var(numpy.array(stop_criteria)) > 0.000001:

		toolbox.unregister("gaussian_mutation_op")
		toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=current_sigma, indpb=mutation_probability)
		
		offspring = tools.selRandom(population,k=int(population_size/2))
		offspring = list(map(toolbox.clone, offspring))

		# Crossing
		for child_1, child_2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < crossing_probability:
				toolbox.cross_twopoints(child_1, child_2)
				del child_1.fitness.values
				del child_2.fitness.values
		# Mutation
		for mutant in offspring:
			if random.random() < mutation_probability:
				toolbox.gaussian_mutation(mutant)
				del mutant.fitness.values
		
		# Fitness
		offspring_fitnesses = [toolbox.evaluate_individual(individual) for individual in offspring]
		for individual, fitness in zip(offspring, offspring_fitnesses):
			individual.fitness.values = fitness

		# In Vitro Fertilization Module
		ivfoffspring = list(map(toolbox.clone, population))
		ivffits = [ind.fitness.values[0] for ind in ivfoffspring]
		fatheridx = numpy.argmax(ivffits)		
		# fatherfit = numpy.max(ivffits)
		father = creator.individual(ivfoffspring[fatheridx].copy())

		for ind in ivfoffspring[::2]:
			toolbox.random_mutation(ind)
			del ind.fitness.values

		for child1 in ivfoffspring:
			child2 = creator.individual(father.copy())
			toolbox.cross_twopoints(child1, child2)
			del child1.fitness.values
			del child2.fitness.values

		ivffitnesses = [toolbox.evaluate_individual(ind) for ind in ivfoffspring]
		for ind, fit in zip(ivfoffspring, ivffitnesses):
			ind.fitness.values = fit
		
		popmax = max(offspring_fitnesses)
		for ind in ivfoffspring:
				if (ind.fitness.values >= popmax):
					population.append(ind)

		new_population = population+offspring

		# Selection
		selected = toolbox.selNSGA2(new_population, population_size)
		selected = list(map(toolbox.clone, selected))

		population[:] = selected

		hof.update(population)

		record = stats.compile(population)
		logbook.record(gen=current_generation, evals=len(population), **record)
		# print(logbook.stream)

		# Statistics
		fits_1 = [ind.fitness.values[0] for ind in population]
		fits_2 = [ind.fitness.values[1] for ind in population]

		mean_1_offspring = sum(fits_1) / len(population)
		mean_2_offspring = sum(fits_2) / len(population)

		generations_fitness_1.append(numpy.log10(mean_1_offspring))
		generations_fitness_2.append(numpy.log(mean_2_offspring))
		
		stop_criteria.append(numpy.log10(mean_1_offspring))

		generations_population.append(population.copy())

		current_generation+=1
		current_sigma = sigma_MAX/((current_generation+1)/1.5)
		
	best_individual = tools.selNSGA2(population, 1)[0]

	# print('Result:', multi_fitness(best_individual,data, data_proj, number_endmembers))

	M = data[:,best_individual]
	duration = time.time() - start_time

	return M, duration, [generations_fitness_1, generations_fitness_2 ,generations_population, current_generation]
