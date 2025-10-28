import artificialNeuralNetwork as ANN
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



# for plotting v(t)
vt_values = []

print(len(values_pkl))

# choose our 6 beta values (do this in a loop later)
chaos1_l = list(values_pkl.keys())[100]
chaos2_l = list(values_pkl.keys())[200]
bif1_l = list(values_pkl.keys())[350]
bif2_l = list(values_pkl.keys())[550]
per1_l = list(values_pkl.keys())[700]
per2_l= list(values_pkl.keys())[900]

# conver all to tensor (also will be looped)
c1_array = values_pkl[chaos1_l].astype(np.float32)  # shape (1500,3)
chaos1 = torch.tensor(c1_array, dtype=torch.float32).to(device)
c2_array = values_pkl[chaos2_l].astype(np.float32)  # shape (1500,3)
chaos2 = torch.tensor(c2_array, dtype=torch.float32).to(device)
bif1_array = values_pkl[bif1_l].astype(np.float32)  # shape (1500,3)
bif1 = torch.tensor(bif1_array, dtype=torch.float32).to(device)
bif2_array = values_pkl[bif2_l].astype(np.float32)  # shape (1500,3)
bif2 = torch.tensor(bif2_array, dtype=torch.float32).to(device)
per1_array = values_pkl[per1_l].astype(np.float32)  # shape (1500,3)
per1 = torch.tensor(per1_array, dtype=torch.float32).to(device)
per2_array = values_pkl[per2_l].astype(np.float32)  # shape (1500,3)
per2 = torch.tensor(per2_array, dtype=torch.float32).to(device)

values_list = [
    chaos1, chaos2, bif1, bif2, per1, per2
]

print(chaos1.shape)  # torch.Size([1500,3])



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
        vt_values.append(f_x[:, 0])

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


gens = []
maxs = []
i = 1

# run for all 6 points
for v in values_list:
    final = 6
    print("NOW PRINTING CYCLE ", i, "of ", final)
    pop, log, hof = main(v)
    hof_fits.append(hof)
    i += 1

    # append to lists for plotting
    
    gens.append(log.select("gen"))
    maxs.append(log.select("max"))

j = 1
# print best set of weights for all 6
for h in hof_fits:
    best_ind = h[0]  # best set of weights
    best_score = best_ind.fitness.values[0]
    best_scores.append(best_score)
    print(f"best fitness {j}: {best_score} ")
    j += 1







# PLOTS

# best fitness plotted against b
best_scores = [1.72, 2.5, 1.3, .025, .85, .83]
b_vals = [.2, .4, .7, 1.1, 1.4, 1.8]
colors = ['red', 'red', 'green', 'green', 'blue', 'blue'] # use colormap for real thung
fig, ax = plt.subplots()
p1 = ax.scatter(b_vals[0:2], best_scores[0:2], c = colors[0:2], label='chaotic', alpha = .3)
p2 = ax.scatter(b_vals[2:4], best_scores[2:4], c = colors[2:4], label='bifurcation', alpha=.3)
p3 = ax.scatter(b_vals[4:6], best_scores[4:6], c = colors[4:6], label='periodic', alpha = .3)

ax.legend(handles = [p1, p2, p3])
plt.show()


# facets of 6 charts

# TODO FIX, RESULTS ARE STACKING
fig, axs = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
axs[0, 0].plot(gens[0], maxs[0], "b-", label="Maximum Fitness")
axs[0, 0].set_title('Chaos 1')

axs[0, 1].plot(gens[1], maxs[1], "b-", label="Maximum Fitness")
axs[0, 1].set_title('Chaos 2')


axs[1, 0].plot(gens[2], maxs[2], "b-", label="Maximum Fitness")
axs[1, 0].set_title('Bifurcation 1')

axs[1, 1].plot(gens[3], maxs[3], "b-", label="Maximum Fitness")
axs[1, 1].set_title('Bifurcation 2')


axs[2, 0].plot(gens[4], maxs[4], "b-", label="Maximum Fitness")
axs[2, 0].set_title('Periodic 1')

axs[2, 1].plot(gens[5], maxs[5], "b-", label="Maximum Fitness")
axs[2, 1].set_title('Periodic 2')



for ax in axs.flat:
    ax.set(xlabel='Generation', ylabel='Fitness')



for ax in axs.flat:
    ax.label_outer()

# # fig.savefig("test.png")
plt.show()

