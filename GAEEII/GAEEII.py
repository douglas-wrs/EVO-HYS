def GAEEII (data, dimensions, number_endmembers, parameters):
    from deap import base
    from deap import creator
    from deap import tools
    def princomp(A):
        M = (A-np.mean(A.T, axis=1)).T
        [latent,coeff] = la.eig(np.cov(M))
        score = np.dot(coeff.T,M)
        return coeff, score, latent
    def affine_projection(Y,p):
        from scipy.sparse.linalg import svds, eigs
        L, N = Y.shape
        Y = np.matrix(Y)
        my = Y.mean(axis=1)
        Y = Y - np.matlib.repmat(my, 1, N)
        Up, D, _vt = svds(Y*Y.T/N, k=p-1)
        Up = np.matrix(Up)
        Y = Up*Up.T*Y
        Y = Y + np.matlib.repmat(my,1,N)
        my_ortho = my-Up*Up.T*my
        aux = my_ortho/np.sqrt(sum(np.square(my_ortho)))
        aux = aux.reshape((aux.shape[0],1))
        Up = np.concatenate((Up, aux), axis=1)
        Y = Up.T*Y
        return Y
    def affine_tranform(Y,p):
        from numpy import linalg as la
        L, N = Y.shape
        d = Y.mean(axis=1)
        U = Y-(d*np.ones((L,N)).T).T
        R = np.dot(U,U.T)
        D,eV = la.eig(R)
        C = eV[:,(L-p+1):]
        Xd = np.dot(la.pinv(C), U)
        Xc =  np.concatenate((Xd, -1*np.ones((1,Xd.shape[1]))), axis=0)
        return Xc
    def generate_individual(creator, number_endmembers, number_pixels, number_rows, number_columns):
        individual = np.random.choice(list(range(0,number_pixels)), number_endmembers)
        return creator.individual(individual)
    def gaussian_mutation(individual, toolbox, number_rows, number_columns):
        gene_x = (np.array(individual) / number_rows).astype(int)
        gene_y = (np.array(individual) % number_rows)
        mut_x = abs(toolbox.gaussian_mutation_op(gene_x.copy())[0] % number_columns-1)
        mut_y = abs(toolbox.gaussian_mutation_op(gene_y.copy())[0] % number_rows-1)
        mutant = mut_x*number_rows+mut_y
        return creator.individual(mutant)
    def simplex_volume(indices, data_pca, number_endmembers):
        TestMatrix = np.zeros((number_endmembers,number_endmembers))
        for i in range(0,number_endmembers):
            TestMatrix[0:number_endmembers,i] = data_proj[:,int(indices[i])]
        volume = np.abs(np.linalg.det(TestMatrix))
        return volume
    def mono_fitness (indices, data_proj, number_endmembers):
        return (simplex_volume(indices, data_proj, number_endmembers),)
    def cross_prob(ind_a, ind_b, mate_prob):
        new_ind_a = []
        new_ind_b = []
        for k in range(0,len(ind_a)):
            if rand() < mate_prob:
                new_ind_a.append(ind_b[k])
            else:
                new_ind_a.append(ind_a[k])
            if rand() < mate_prob:
                new_ind_b.append(ind_a[k])
            else:
                new_ind_b.append(ind_b[k])
        return (creator.individual(new_ind_a),creator.individual(new_ind_b)) 
    
    start_time = time.time()
    
    [population_size,
     number_generations,
     crossing_probability,
     mutation_probability,
     stop_criteria_MAX,
     tournsize,
     number_epochs,
     selection] = parameters
    
    random.seed(64)
    
    number_rows = int(dimensions[0])
    number_columns = int(dimensions[1])
    number_bands = int(dimensions[2])
    number_pixels = number_rows*number_columns
    sigma_MAX = max(number_rows, number_generations)
    
    data = np.asarray(data)
    _coeff, score, _latent = princomp(data.T)
    data_proj = np.squeeze(score[0:number_endmembers,:])

    M_list = []
    best_individual_list = []
    epoch_generations_fitness = []
    epoch_generations_population = []
    epoch_current_generation = []
    
    for ep in range(0,number_epochs):
        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", list, fitness=creator.fitness)
        
        toolbox = base.Toolbox()
        
        toolbox.register("create_individual", generate_individual, creator, number_endmembers,number_pixels, number_rows, number_columns)
        toolbox.register("initialize_population", tools.initRepeat, list, toolbox.create_individual)
        toolbox.register("evaluate_individual", mono_fitness, data_proj=data_proj, number_endmembers=number_endmembers)
    
        toolbox.register("cross_prob", cross_prob, mate_prob=crossing_probability)
        
        toolbox.register("tournament_select", tools.selTournament, tournsize=tournsize)
        
        toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=0, indpb=mutation_probability)
        toolbox.register("gaussian_mutation", gaussian_mutation, toolbox=toolbox, number_rows=number_rows, number_columns=number_columns)
        
        population = toolbox.initialize_population(n=population_size)
        population_fitnesses = [toolbox.evaluate_individual(individual) for individual in population]
        for individual, fitness in zip(population, population_fitnesses):
            individual.fitness.values = fitness
            
        hof = tools.HallOfFame(population_size)
        hof.update(population)
        
        current_generation = 0
        current_sigma = sigma_MAX
        
        generations_fitness = []
        generations_population = []
        
        stop_criteria = deque( maxlen=stop_criteria_MAX)
        stop_criteria.extend(list(range(1,stop_criteria_MAX)))
        
        while current_generation < number_generations and np.var(np.array(stop_criteria)) > 0.001: 
            
            toolbox.unregister("gaussian_mutation_op")
            toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=current_sigma, indpb=mutation_probability)
            
            offspring = list(map(toolbox.clone, population))
            crossed_offspring = []
            # Crossing
            
            for child_1, child_2 in zip(offspring[::2], offspring[1::2]):
                child_a, child_b = toolbox.cross_prob(child_1, child_2)
                del child_a.fitness.values
                del child_b.fitness.values
                crossed_offspring.append(child_a)
                crossed_offspring.append(child_b)
                
            # Mutation
            for mu in range(0,len(crossed_offspring)):
                mutant = toolbox.gaussian_mutation(crossed_offspring[mu])
                del mutant.fitness.values
                crossed_offspring[mu] = mutant
                
            # Fitness
            crossed_offspring_fitnesses = [toolbox.evaluate_individual(individual) for individual in crossed_offspring]
            for individual, fitness in zip(crossed_offspring, crossed_offspring_fitnesses):
                individual.fitness.values = fitness
                
            new_population = population+crossed_offspring
            
            # Selection
            selected = toolbox.tournament_select(new_population, population_size)
            population_selected = list(map(toolbox.clone, selected))
            
            population[:] = population_selected
            hof.update(population)
            
            # Statistics
            fits = [ind.fitness.values[0] for ind in population]
            mean_offspring = sum(fits) / len(population)
            
            generations_fitness.append(np.log10(mean_offspring))
            generations_population.append(population.copy())
            
            stop_criteria.append(np.log10(mean_offspring))
            
            current_generation+=1
            current_sigma = sigma_MAX/((current_generation+1)/4)
            
        best_individual = tools.selBest(population, 1)[0]
        
        M = data[:,best_individual]
        best_individual_list.append(best_individual)
        M_list.append(M)
        
        epoch_generations_fitness.append(generations_fitness)
        epoch_generations_population.append(generations_population)
        epoch_current_generation.append(current_generation)
        
    M_all = np.hstack(M_list)
    
    last_volume = list(map( lambda x: x[-1], epoch_generations_fitness))
    
    indx_max = np.argmax(last_volume)
    indx_min = np.argmin(last_volume)
    indx_med = last_volume.index(np.percentile(last_volume,50,interpolation='nearest'))
    
    if selection == 'min':
        M_result = M_list[indx_min]
    if selection == 'avg':
        M_result = M_list[indx_med]
    if selection == 'max':
        M_result = M_list[indx_max]
    if selection == 'all':
        M_result =  M_all
    
    duration = time.time() - start_time
    
    return M_result, duration, [M_list, best_individual_list, epoch_generations_fitness, epoch_generations_population, epoch_current_generation, hof]