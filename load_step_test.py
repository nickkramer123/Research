import artificialNeuralNetwork as ANN
from deap import base, creator, tools
import random
from sklearn.feature_selection import mutual_info_regression
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn




# Step 1: with X, gather f(x)

# using cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# import rossler attractor file and convert to pyTorch Tensor
with open("./Research/data/values.pkl", "rb") as file: # Read as numpy arrays shape [3000, 3] (3000 samples, 3 features)
    values = pickle.load(file) # 3000 lists of 3 objects [x1, y1, z1]
values_array = np.array(list(values.values()), dtype=np.float32)
values = torch.tensor(values_array).to(device)


# returns V, or f(x) + e
# TODO this takes a long time, take a sample?
class Network():
    def __init__(self):
        self.model = ANN.NeuralNetwork().to(device)

    # TODO should values be given? probably
    def run(self, values):
        V_values = [] # initialize list to store V, of f(x) + e
        for i in range(len(values)): # for every little x [x, y, x] value in big X
            x = values[i].unsqueeze(0)
            with torch.no_grad():  # no gradients needed
                y = self.model(x)
            V_values.append(y.item())
        return V_values


# TODO move?
# setup network
net = Network()








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





# takes weights from genome and gives them to ANN
def load_weights_from_vector(model, vector):
    pointer = 0
    for param in model.parameters(): # for each parameter
        num = param.numel() # number of elements
        slice  = vector[pointer:pointer + num]
        param.data = slice.view_as(param)
        pointer += num



# TODO need to fix
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
    score = synergy(values_array, f_x)
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


vec = torch.randn(IND_SIZE)
load_weights_from_vector(net.model, vec)
flat = torch.cat([p.data.view(-1) for p in net.model.parameters()])
assert torch.allclose(vec, flat), "Weights not loaded correctly!"
print("Loader verified.")
