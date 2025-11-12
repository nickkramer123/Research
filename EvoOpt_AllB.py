import Artificial_Nueral_Network as ANN
from deap import base, creator, tools
import random
from sklearn.feature_selection import mutual_info_regression
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchviz import make_dot
import matplotlib.pyplot as plt
from npeet import entropy_estimators as ee





# Setup for DEAP
# types
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # build type for fitness max problem
creator.create("Individual", list, fitness=creator.FitnessMax) # create representation of a possible solution


# toolbox - This creates functions to initialize populations from individuals that are themselves initialized with random float numbers.
# size = number of parameters in ANN
IND_SIZE = 249


toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# for logbook 
stats = tools.Statistics(key=lambda ind: ind.fitness.values) # may need [0]


stats.register("avg", np.mean)
stats.register("max", np.max)




# using cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# import rossler attractor file and convert to pyTorch Tensor
with open("./Research/data/values.pkl", "rb") as file: # Read as numpy arrays shape [1500, 3] (1500 samples, 3 features)
    values_pkl = pickle.load(file) # 1500 lists of 3 objects [x1, y1, z1]
    




# Classes
# returns V, or f(x) + e
class Network():
    def __init__(self):
        self.model = ANN.NeuralNetwork().to(device)

    def run(self, values):
        V_values = [] # initialize list to store V, of f(x) + e
        for i in range(len(values)): # for every little x [x, y, x] value in big X
            x = values[i].unsqueeze(0)
            with torch.no_grad():  # no gradients needed
                y = self.model(x)
            V_values.append(y.item())
        return V_values





# Functions

# takes weights from genome and gives them to ANN
def load_weights_from_vector(model, vector):
    pointer = 0
    for param in model.parameters(): # for each parameter
        num = param.numel() # number of elements
        slice  = vector[pointer:pointer + num]
        param.data = slice.view_as(param)
        pointer += num




    
# synergy function, used to evaluate
"""
Computes the awnser to our synergy equation (EXPLATIN IN MORE DETAIL)
param X: 2D array of sample values for x, y, z (1500 sample, 3 variables)
param f_x: list of x values after being put through neural network
return final: synergy value, fitness value for evolutionary optimization
"""
def synergy(X, f_x):
    back = 0
    front = ee.mi(X, f_x.ravel()) # finds mutual info for big X
    for j in range(X.shape[1]): # for each column
        x = X[:, j].reshape(-1, 1) # reshape the X vector to fit out function
        back += mutual_info_regression(x, f_x.ravel())[0] # finds sum of mutual info for each individual feature little x
    final = front - np.sum(back)
    return final





# Create ANN, setup logbook
net = Network()





# operators, or initializers that are already implemented
toolbox.register("mate", tools.cxTwoPoint) # crossover, combines the genetic material of two parents to produce offspring
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # randomly alter part of an individual to introduce new variation
toolbox.register("select", tools.selTournament, tournsize=3) # Chooses the individuals from the population to survive and reproduce
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
# deap eaSimple algorithim
# TODO return logbook?
def main(values):
    
    population = toolbox.population(n=100)
    CXPB, MUTPB, NGEN = 0.7, 0.2, 100


    hof = tools.HallOfFame(1) # keeps best individual
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "max", "size", "spam"

    # EVALUATE FUNCTION
    # evaluates a synerygy score for an individual
    def evaluate(individual):
        # check for torch accuracy
        genome = torch.tensor(individual, dtype=torch.float32, device = device)

        # load genome into model
        load_weights_from_vector(net.model, genome)

        # foward pass, runing the model feeding input and getting output
        with torch.no_grad():
            f_x = net.model(values).cpu().numpy()

        # calculate and return synergy
        score = synergy(values, f_x)

        # record one representative f_x value per iteration for plotting
        # vt_values.append(f_x[:, 0])

        return (score, )

    toolbox.register("evaluate", evaluate) # assigns a fitness score to an individual







    # evaluates the individuals with an invalld fitness
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # enters the generational loop
    for g in range(NGEN):

        # select next generation individuals
        offspring = toolbox.select(population, len(population))
        # clone the selected individuals (for independence)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        population[:] = offspring


        # logging for experiment
        # update hall of fame
        hof.update(population)

        # record for logbook
        record = stats.compile(population)
        logbook.record(gen=g, **record)

        # to keep track of loading times
        print(g)



    # visualize best output before return
    best_ind = hof[0]
    load_weights_from_vector(net.model, torch.tensor(best_ind, dtype=torch.float32, device=device))

    dummy_input = torch.randn(1, 3).to(device)
    output = net.model(dummy_input)

    dot = make_dot(output, params=dict(net.model.named_parameters()))
    #dot.render("ANN_graph", format="png", view=True)

    
    return population, logbook, hof




# NOW WE SET UP THE TRAINING LOOP

hof_fits = []
best_scores = []
logbooks = []


gens = []
maxs = []

# run for all 6 points
for i, (b_key, arr) in enumerate(values_pkl.items()):

    b_val = float(b_key)  # 🔧 convert numpy float -> python float
    values_array = arr.astype(np.float32)
    values = torch.tensor(values_array, dtype=torch.float32).to(device)




    print("NOW PRINTING CYCLE ", i, "b = ", b_val)
    pop, log, hof = main(values)
    hof_fits.append(hof)
    logbooks.append(log)
    best_scores.append(hof[0].fitness.values[0])



# write out best fitnesses to text file just in case
with open('best_scores.txt', 'w') as file:
    for score in best_scores:
        row_string = ' '.join(map(str, score))
        file.write(row_string + '\n')




# PLOTS
# TODO FIX
# best fitness plotted against b
b = 0
fig, ax = plt.subplots()
for i, s in enumerate(best_scores):

    # choose color and label
    color = 'red' 
    labels = 'chaotic'   
    if i > 25:
        color = 'green'
        labels = 'bifurcation'
    if color > 55:
        color = 'blue'
        labels = 'periodic'

    p1 = ax.scatter(b, s, c = color, alpha = .3)
    b += .002

ax.set_xlabel('b value')
ax.set_ylabel('best fitness')

plt.show()
plt.savefig("allB.png")



