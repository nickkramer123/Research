import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression

# TODO make the synergy equation
# TODO get x and f(x) values in here, then test synergy
# TODO get X values into this function? maybe make it all one file
"""
Computes the awnser to our synergy equation (EXPLATIN IN MORE DETAIL)
param X: list of x values
param f_x: list of x values after being put through neural network
return final: answer to our equation
"""
def synergy(X, f_x):
    back = 0
    front = mutual_info_regression(X, f_x)
    for i in range(len(X)):
        x = X[i]
        back += mutual_info_regression(x, f_x)
    syn = front - back
    return syn




















# using cpu
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"




# import rossler attractor file
# Read as numpy arrays
x_values = np.loadtxt("C:/Users/nflan/OneDrive/Documents/Research/data/x_values.txt")
y_values = np.loadtxt("C:/Users/nflan/OneDrive/Documents/Research/data/y_values.txt")
z_values = np.loadtxt("C:/Users/nflan/OneDrive/Documents/Research/data/z_values.txt")

# Stack arrays into shape [3000, 3] (3000 samples, 3 features)
# 3000 lists of 3 objects [x1, y1, z1]
values = np.column_stack((x_values, y_values, z_values))

# Convert to PyTorch tensor
values = torch.tensor(values, dtype=torch.float32).to(device)


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

