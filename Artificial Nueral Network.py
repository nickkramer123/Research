import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_selection import mutual_info_regression

# TODO make the synergy equation
# TODO convert seperate values columns into 1 column
# TODO get X values into this function? maybe make it all one file
"""
Computes the awnser to our synergy equation (EXPLATIN IN MORE DETAIL)
param X: 2D array of sample values for x, y, z (3000 sample, 3 variables)
param f_x: list of x values after being put through neural network
return final: answer to our equation
"""
def synergy(X, f_x):
    back = 0
    front = mutual_info_regression(X, f_x.ravel())[0] # finds mutual info for big X
    for j in range(X.shape[1]): # for each column
        x = X[:, j].reshape(-1, 1) # reshape the X vector to fit out function
        back += mutual_info_regression(x, f_x.ravel())[0] # finds sum of mutual info for each individual feature little x
    final = front - np.sum(back)
    return final











# using cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"





# import rossler attractor file
# Read as numpy arrays shape [3000, 3] (3000 samples, 3 features)
# 3000 lists of 3 objects [x1, y1, z1]
with open("./Research/data/values.pkl", "rb") as file:
    values = pickle.load(file)


# Convert to PyTorch tensor
values_array = np.array(list(values.values()), dtype=np.float32)
values = torch.tensor(values_array).to(device)

# TODO check difference made cus 16
class NeuralNetwork(nn.Module): # define neuralnetwork class that inherits from nn.Module
    # define layers of the function in initialization
    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten() # Initialize the flatten layer to convert into contiguous array of values 
        self.linear_relu_stack = nn.Sequential( # ordered container of modules
            nn.Linear(3, 16), # appies a linear transformation on the input using stored weights and bases
            nn.ReLU(), # applies nonlinearity
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    # specify how data will pass through the network
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

# create an instance of NeuralNetwork, move it to device, print its structure
model = NeuralNetwork().to(device)
# print(model)

# use the model
V_logits = model(values) # calls the model
print(f"Predicted class: {V_logits}")

