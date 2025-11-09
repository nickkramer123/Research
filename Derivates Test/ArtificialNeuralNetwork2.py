import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor






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
    


