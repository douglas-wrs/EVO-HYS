def GAEEII (data, dimensions, number_endmembers, parameters):
    from deap import base
    from deap import creator
    from deap import tools
    def princomp(A):
        M = (A-np.mean(A.T, axis=1)).T
        [latent,coeff] = la.eig(np.cov(M))
        score = np.dot(coeff.T,M)
        return coeff, score, latent
    def generate_bi_individual(creator, number_endmembers, number_pixels, number_rows, number_columns):
        individual = np.random.choice(list(range(0,number_pixels)), number_endmembers)
        return creator.individual(list(zip(individual,individual)))
    def gaussian_bi_mutation(individual, toolbox, number_rows, number_columns):    
        individual_left = list(map(lambda gene: gene[0], individual))
        individual_right = list(map(lambda gene: gene[1], individual))
        gene_x_1 = (np.array(individual_left) / number_rows).astype(int)
        gene_y_1 = (np.array(individual_left) % number_rows)
        gene_x_2 = (np.array(individual_right) / number_rows).astype(int)
        gene_y_2 = (np.array(individual_right) % number_rows)
        mut_x_1 = abs(toolbox.gaussian_mutation_op(gene_x_1.copy())[0] % number_columns-1)
        mut_y_1 = abs(toolbox.gaussian_mutation_op(gene_y_1.copy())[0] % number_rows-1)
        mutant_1 = mut_x_1*number_rows+mut_y_1
        mut_x_2 = abs(toolbox.gaussian_mutation_op(gene_x_2.copy())[0] % number_columns-1)
        mut_y_2 = abs(toolbox.gaussian_mutation_op(gene_y_2.copy())[0] % number_rows-1)
        mutant_2 = mut_x_2*number_rows+mut_y_2
        return list(zip(mutant_1,mutant_2))
    def random_mutation(individual, number_pixels, number_endmembers, mutation_probability):
        new_individual =  random.sample(set(individual + random.sample(range(0,number_pixels), number_endmembers)), number_endmembers)
        mutant = [new_individual[i] if random.random() < mutation_probability else individual[i] for i in range(0,number_endmembers)]
        return mutant
    def simplex_volume_bi(indices, data, pca, number_endmembers):
        gen_data = np.zeros((number_endmembers,data.T.shape[1]))
        k = 0
        for gene in indices:
            if gene[0] == gene[1]:
                gen_data[k,:] = data[:,gene[0]].T
                k+=1
            else:
                gen_data[k,:] = np.multiply(data[:,gene[0]].T,data[:,gene[1]].T)
                k+=1
        gen_data_pca = pca.transform(gen_data)
        volume = np.abs(np.linalg.det(gen_data_pca))
        return volume
    def cross_hadamard(ind_a, ind_b, mate_prob, virtual_prob):
        new_ind_a = []
        new_ind_b = []
        for k in range(0,len(ind_a)):
            if rand() < mate_prob:
                if rand() < virtual_prob:
                    new_ind_a.append((ind_a[k][0],ind_b[k][1]))
                else:
                    new_ind_a.append(ind_b[k])
            else:
                new_ind_a.append(ind_a[k])
            if rand() < mate_prob:
                if rand() < virtual_prob:
                    new_ind_b.append((ind_b[k][0],ind_a[k][1]))
                else:
                    new_ind_b.append(ind_a[k])
            else:
                new_ind_b.append(ind_b[k])
        return (creator.individual(new_ind_a),creator.individual(new_ind_b)) 
    def mono_fitness_bi (indices, data, pca, number_endmembers):
        return (simplex_volume_bi(indices, data, pca, number_endmembers),)
    start_time = time.time()
    [population_size,
     number_generations,
     crossing_probability,
     mutation_probability,
     virtual_probability,
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
    tournsize = 3
    from sklearn.decomposition import PCA
    pca = PCA(n_components=number_endmembers)
    pca.fit(data.T)
    M_list = []
    epoch_generations_fitness = []
    epoch_generations_population = []
    epoch_current_generation = []
    for ep in range(0,number_epochs):
        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", list, fitness=creator.fitness)
        toolbox = base.Toolbox()
        toolbox.register("create_individual", generate_bi_individual, creator, number_endmembers,number_pixels,number_rows, number_columns)
        toolbox.register("initialize_population", tools.initRepeat, list, toolbox.create_individual)
        toolbox.register("evaluate_simplex_volume", mono_fitness_bi, data=data, pca=pca, number_endmembers=number_endmembers)
        toolbox.register("cross_hadamard", cross_hadamard, mate_prob=crossing_probability, virtual_prob=virtual_probability)
        toolbox.register("tournament_select", tools.selTournament, tournsize=tournsize)
        toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=0, indpb=mutation_probability)
        toolbox.register("gaussian_mutation_bi", gaussian_bi_mutation, toolbox=toolbox, number_rows=number_rows, number_columns=number_columns)
        population = toolbox.initialize_population(n=population_size)
        population_fitnesses = [toolbox.evaluate_simplex_volume(individual) for individual in population]
        for individual, fitness in zip(population, population_fitnesses):
            individual.fitness.values = fitness
        hof = tools.HallOfFame(10)
        hof.update(population)
        current_generation = 0
        current_sigma = 2*sigma_MAX
        generations_fitness = []
        generations_population = []
        stop_criteria = deque( maxlen=stop_criteria_MAX)
        stop_criteria.extend(list(range(1,stop_criteria_MAX)))
        while current_generation < number_generations and np.var(np.array(stop_criteria)) > 0.0000001:  
            toolbox.unregister("gaussian_mutation_op")
            toolbox.register("gaussian_mutation_op", tools.mutGaussian, mu=0, sigma=current_sigma, indpb=mutation_probability)
#             offspring = tools.selRandom(population,k=int(population_size/2))
            offspring = list(map(toolbox.clone, population))
            crossed_offspring = []
            # Crossing
            for child_1, child_2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossing_probability:
                    child_a, child_b = toolbox.cross_hadamard(child_1, child_2)
                    del child_a.fitness.values
                    del child_b.fitness.values
                    crossed_offspring.append(child_a)
                    crossed_offspring.append(child_b)
            # Mutation
            for mutant in crossed_offspring:
                if random.random() < mutation_probability:
                    toolbox.gaussian_mutation_bi(mutant)
                    del mutant.fitness.values
            # Fitness
            crossed_offspring_fitnesses = [toolbox.evaluate_simplex_volume(individual) for individual in crossed_offspring]
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
#         best_individual = hof[0]
        M = np.zeros((number_endmembers,data.T.shape[1]))
        k = 0
        epoch_generations_fitness.append(generations_fitness)
        epoch_generations_population.append(generations_population)
        epoch_current_generation.append(current_generation)
        for gene in best_individual:
            if gene[0] == gene[1]:
                M[k,:] = data[:,gene[0]].T
                k+=1
            else:
                M[k,:] = np.multiply(data[:,gene[0]].T,data[:,gene[1]].T)
                k+=1
        M_list.append(M.T)
        
    M_all = np.hstack(M_list)
#     M, _, _ = MVES(M_all, dimensions, number_endmembers, parameters)
    duration = time.time() - start_time
    M_min_max = []
    last_volume = list(map( lambda x: x[-1], epoch_generations_fitness))
    indx_max = np.argmax(last_volume)
    indx_min = np.argmin(last_volume)
    indx_med = last_volume.index(np.percentile(last_volume,50,interpolation='nearest'))
    if selection == 'min':
        M_result = M_list[indx_min]
    if selection == 'med':
        M_result = M_list[indx_med]
    if selection == 'max':
        M_result = M_list[indx_max]
    if selection == 'all':
        M_result =  M_all
    return M_result, duration, [M_list, best_individual, epoch_generations_fitness, epoch_generations_population, epoch_current_generation, hof]