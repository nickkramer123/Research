import artificialNeuralNetwork as ANN
from deap import base, creator, tools
import random
from sklearn.feature_selection import mutual_info_regression
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn

import datetime


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
with open("./Research/data/values.pkl", "rb") as file: # Read as numpy arrays shape [3000, 3] (3000 samples, 3 features)
    values_pkl = pickle.load(file) # 3000 lists of 3 objects [x1, y1, z1]




# choose one beta (he first one)
val0 = list(values_pkl.keys())[0]
# converto to tensor
values_array = values_pkl[val0].astype(np.float32)  # shape (3000,3)
values = torch.tensor(values_array, dtype=torch.float32).to(device)

print(values.shape)  # torch.Size([3000,3])


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
    return (score, )
    
# synergy function, used to evaluate
"""
Computes the awnser to our synergy equation (EXPLATIN IN MORE DETAIL)
param X: 2D array of sample values for x, y, z (3000 sample, 3 variables)
param f_x: list of x values after being put through neural network
return final: synergy value, fitness value for evolutionary optimization
"""
def synergy(X, f_x):
    back = 0
    front = mutual_info_regression(X, f_x.ravel())[0] # finds mutual info for big X
    for j in range(X.shape[1]): # for each column
        x = X[:, j].reshape(-1, 1) # reshape the X vector to fit out function
        back += mutual_info_regression(x, f_x.ravel())[0] # finds sum of mutual info for each individual feature little x
    final = front - np.sum(back)
    return final





# Create ANN and setup logbook
net = Network()
logbook = tools.Logbook()



# operators, or initializers that are already implemented
toolbox.register("mate", tools.cxTwoPoint) # crossover, combines the genetic material of two parents to produce offspring
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # randomly alter part of an individual to introduce new variation
toolbox.register("select", tools.selTournament, tournsize=3) # Chooses the individuals from the population to survive and reproduce
toolbox.register("evaluate", evaluate) # assigns a fitness score to an individual
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
def main():
    population = toolbox.population(n=2)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 2
    hof = tools.HallOfFame(1) # keeps best individual

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

        # update hall of fame
        hof.update(population)


        # record for logbook
        record = stats.compile(population)
        logbook.record(gen=g, **record)

    return population, logbook, hof




# NOW WE SET UP THE TRAINING LOOP

# initail run
final_pop, log, hof = main()
best_ind = hof[0]  # best set of weights
best_score = best_ind.fitness.values[0]
print("Best fitness:", best_score)




# NEW STUFF




# To view the graph, launch TensorBoard from your terminal:
# tensorboard --logdir=runs

