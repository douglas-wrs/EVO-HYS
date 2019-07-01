def GAEE (data, dimensions, number_endmembers, parameters):
    from deap import base
    from deap import creator
    from deap import tools
    def princomp(A):
        M = (A-np.mean(A.T, axis=1)).T
        [latent,coeff] = la.eig(np.cov(M))
        score = np.dot(coeff.T,M)
        return coeff, score, latent
    def generate_individual(creator, number_endmembers, number_pixels, number_rows, number_columns):
        individual = np.random.choice(list(range(0,number_pixels)), number_endmembers)
        return creator.individual(individual)
    def random_mutation(individual, number_pixels, number_endmembers, mutation_probability):
        new_individual =  random.sample(set(individual + random.sample(range(0,number_pixels), number_endmembers)), number_endmembers)
        mutant = [new_individual[i] if random.random() < mutation_probability else individual[i] for i in range(0,number_endmembers)]
        return mutant
    def maximize_simplex_volume(indices, data_pca, number_endmembers):
        TestMatrix = np.zeros((number_endmembers,number_endmembers))
        TestMatrix[0,:] = 1
        for i in range(0,number_endmembers):
            TestMatrix[1:number_endmembers,i] = data_pca[:,int(indices[i])]
        volume = np.abs(np.linalg.det(TestMatrix))
        return volume
    def mono_fitness (indices, data_pca, number_endmembers):
        return (maximize_simplex_volume(indices, data_pca, number_endmembers),)
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
    data = np.asarray(data)
    _coeff, score, _latent = princomp(data.T)
    data_pca = np.squeeze(score[0:number_endmembers-1,:])
    M_list = []
    best_individual_list = []
    epoch_generations_fitness = []
    epoch_generations_population = []
    epoch_current_generation = []
    for ep in range(0,number_epochs):
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
        hof = tools.HallOfFame(population_size)
        hof.update(population)
        current_generation = 0
        generations_fitness = []
        generations_population = []
        stop_criteria = deque( maxlen=stop_criteria_MAX)
        stop_criteria.extend(list(range(1,stop_criteria_MAX)))
        while current_generation < number_generations and np.var(np.array(stop_criteria)) > 0.001:
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
            generations_fitness.append(np.log10(mean_offspring))
            generations_population.append(population.copy())
            stop_criteria.append(np.log10(mean_offspring))
            current_generation+=1
        best_individual = tools.selBest(population, 1)[0]
        M = data[:,best_individual]
        best_individual_list.append(best_individual)
        epoch_generations_fitness.append(generations_fitness)
        epoch_generations_population.append(generations_population)
        epoch_current_generation.append(current_generation)
        M_list.append(M)
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