import artificialNeuralNetwork as ANN
from deap import base, creator, tools
import random




class Evolution():
    neural_network = ANN.NeuralNetwork()

# types
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # build type for fitness max problem
creator.create("Individual", list, fitness=creator.FitnessMax) # create representation of a possible solution


# toolbox - This creates functions to initialize populations from individuals that are themselves initialized with random float numbers.
IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# evaluate function
def evaluate(individual):
    return sum(individual),

# operators, or initializers that are already implemented
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)



'''
eaSimple from deaps website
Parameters:
population - a list of individuals
toolbox - contains evolution operators
cxpb - The probability of mating two individuals
mutpb - The probability of mutating an individual
ngen - the number of generation
stats - A statistics object that is updated in place, optional
halloffame - hallOfFame  oject that will contain the best individuals, optional
verbose - whether or not to log the statistics
Returns:
The final population
A class:~deap.tools.Logbook with the statistics of the evolution
'''

'''
evaluate(population)
for g in range(ngen):
    population = select(population, len(population))
    offspring = varAnd(population, toolbox, cxpb, mutpb)
    evaluate(offspring)
    population = offspring
'''
# deap eaSimple algorithim
def main():
    population = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 2

    # evaluates the individuals with an invalld fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # enters the generational loop
    for g in range(NGEN):

        # entierly replace the parental population, stochastic and selects the same individual multiple times
        # do i need more?
        population = toolbox.select(population, len(population))

        # apply the varAnd() function to produce the next generation population
        offspring = varAnd(population = population, # check
                           toolbox = toolbox,
                           cxpb = CXPB,
                           mutpb = MUTPB)

        # evaluates the new individuals and computes the statistics on the population
        evaluate(offspring)
        log = Logbook()
        

        population = offspring

    return population, Logbook()